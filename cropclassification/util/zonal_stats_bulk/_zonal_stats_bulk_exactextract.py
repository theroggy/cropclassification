import logging
import multiprocessing
import os
import signal
import sys
from concurrent import futures
from datetime import datetime
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import psutil

from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util import io_util

from . import _general_helper as general_helper
from . import _processing_util as processing_util
from . import _raster_helper as raster_helper
from . import _vector_helper as vector_helper

logger = logging.getLogger(__name__)


def zonal_stats(
    vector_path: Path,
    columns: list[str],
    rasters_bands: list[tuple[Path, list[str]]],
    output_dir: Path,
    stats: list[str],
    nb_parallel: int = -1,
    force: bool = False,
):
    """Calculate zonal statistics.

    Args:
        vector_path (Path): _description_
        columns (list[str]): _description_
        rasters_bands (list[tuple[Path, list[str]]]): _description_
        output_dir (Path): _description_
        stats (list[str]): _description_
        nb_parallel (int, optional): the number of parallel processes to use.
             Defaults to -1: use all available processors.
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
    # Add extra columns to the columns list
    include_cols = ["index", "UID", "x_ref"]
    include_cols.extend(columns)
    # Make the list unique
    columns = list(set(include_cols))

    # Some checks on the input parameters
    if len(rasters_bands) == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = io_util.create_tempdir("zonal_stats_exactextract")

    start_time = datetime.now()
    nb_todo = 0
    nb_done_total = 0
    if nb_parallel < 1:
        nb_parallel += multiprocessing.cpu_count()

    # Loop over all images and bands to calculate zonal stats in parallel...
    calc_queue = {}
    pool = futures.ProcessPoolExecutor(
        max_workers=nb_parallel, initializer=processing_util.initialize_worker()
    )
    try:
        for raster_path, bands in rasters_bands:
            raster_info = raster_helper.get_image_info(raster_path)
            # Create base output filename
            # TODO: hoort hier niet echt thuis
            orbit = None
            if raster_info.imagetype.lower() == "s1-grd-sigma0-asc":
                orbit = "ASCENDING"
            elif raster_info.imagetype.lower() == "s1-grd-sigma0-desc":
                orbit = "DESCENDING"
            output_base_path = general_helper._format_output_path(
                vector_path,
                raster_path,
                output_dir,
                orbit,
                band=None,
            )
            output_base_busy_path = tmp_dir / f"BUSY_{output_base_path.name}"

            # Check for which bands there is a valid output file already
            if bands is None:
                bands = raster_info.bands
                assert bands is not None
            output_paths = {}

            for band in bands:
                # Prepare the output paths...
                output_band_path = general_helper._format_output_path(
                    vector_path, raster_path, output_dir, orbit, band
                )
                output_band_busy_path = tmp_dir / f"BUSY_{output_band_path.name}"

                # If a busy output file exists, remove it, otherwise we can get
                # double data in it...
                output_band_busy_path.unlink(missing_ok=True)

                # Check if the output file exists already
                if output_band_path.exists():
                    if not force:
                        logger.debug(f"Output file for band exists {output_band_path}")
                        continue
                    else:
                        output_band_path.unlink()
                output_paths[band] = output_band_path

            # If all bands already processed, skip image...
            if len(output_paths) == 0:
                logger.info(
                    "SKIP image: output files for all bands exist already for "
                    f"{output_base_path}"
                )
                continue

            nb_todo += 1
            future = pool.submit(
                zonal_stats_band_tofile,
                vector_path=vector_path,
                raster_path=raster_path,
                bands=bands,
                stats=stats,
                tmp_dir=tmp_dir,
                include_cols=columns,
                output_paths=output_paths,
            )
            calc_queue[future] = {
                "vector_path": vector_path,
                "raster_path": raster_path,
                "bands": bands,
                "bands_todo": output_paths,
                "prepare_calc_future": future,
                "image_info": raster_info,
                "prepare_calc_starttime": datetime.now(),
                "output_base_path": output_base_path,
                "output_base_busy_path": output_base_busy_path,
                "status": "IMAGE_PREPARE_CALC_BUSY",
            }

        # Already write progress to indicate we have started
        logger.info(
            general_helper.format_progress_message(nb_todo, nb_done_total, start_time)
        )

        # Keep looping until all futures are done.
        # The loop is used instead of using as_completed keeps UI responsive, allows
        # for e.g. to log memory usage evolution and allows to catch CTRL+C.
        min_available = 4 * 1024 * 1024 * 1024
        while True:
            # Log memory availability if it becomes low
            virtual_available = psutil.virtual_memory().available
            if virtual_available < min_available:
                virtual_available_str = general_helper.formatbytes(virtual_available)
                logger.info(f" {virtual_available_str=}")
            done, not_done = futures.wait(
                calc_queue, timeout=10, return_when=futures.FIRST_COMPLETED
            )

            for future in done:
                try:
                    _ = future.result()
                except Exception as ex:
                    raise Exception(
                        f"Error calculating {calc_queue[future]}: {ex}"
                    ) from ex
                del calc_queue[future]
                nb_done_total += 1

                logger.info(
                    general_helper.format_progress_message(
                        nb_todo, nb_done_total, start_time
                    )
                )

            # If no more futures are not_done, we're done
            if len(not_done) == 0:
                break
    except KeyboardInterrupt:  # pragma: no cover
        # If CTRL+C is used, shut down pool and kill children
        print("You pressed Ctrl+C")
        print("Worker processes are being stopped, followed by exit!")

        # Stop process pool + kill children + exit
        try:
            pool.shutdown(wait=False)
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for process_pid in children:
                print(f"Kill child with pid {process_pid}")
                process_pid.send_signal(signal.SIGTERM)
        finally:
            sys.exit(1)
    finally:
        pool.shutdown()


def zonal_stats_band(
    vector_path,
    raster_path: Path,
    tmp_dir: Path,
    stats: list[str],
    include_cols: list[str],
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    # Get the image info
    image_info = raster_helper.get_image_info(raster_path)

    # Reproject the vector data
    tmp_dir.mkdir(exist_ok=True, parents=True)
    vector_proj_path = vector_helper.reproject_synced(
        path=vector_path,
        columns=[*include_cols, "geometry"],
        target_epsg=image_info.image_epsg,
        dst_dir=tmp_dir,
    )

    try:
        import exactextract

        stats_df = exactextract.exact_extract(
            rast=raster_path,
            vec=vector_proj_path,
            ops=stats,
            include_geom=False,
            output="pandas",
            include_cols=include_cols,
        )

    except Exception:
        raise

    return stats_df


def zonal_stats_band_tofile(
    vector_path,
    raster_path: Path,
    output_paths: dict[str, Path],
    bands: list[str],
    tmp_dir: Path,
    stats: list[str],
    include_cols: list[str],
    force: bool = False,
) -> dict[str, Path]:
    # Init
    if all(output_path.exists() for output_path in output_paths.values()):
        if force:
            for output_path in output_paths.values():
                output_path.unlink(missing_ok=True)
        return output_paths

    # In stats replace 'std' with 'stdev' for exactextract
    stats = [stat.replace("std", "stdev") for stat in stats]

    # Add the operational arguments to the stats
    min_coverage_frac = 0.8
    coverage_weight = "none"
    operation_arguments = (
        f"(min_coverage_frac={min_coverage_frac},coverage_weight={coverage_weight})"
    )
    stats = [stat + operation_arguments for stat in stats]

    stats_df = zonal_stats_band(
        vector_path=vector_path,
        raster_path=raster_path,
        stats=stats,
        tmp_dir=tmp_dir,
        include_cols=include_cols,
    )

    # Split stats_df in different dataframes for each band index
    raster_info = raster_helper.get_image_info(raster_path)
    for band in bands:
        index = raster_info.bands[band].band_index
        band_columns = include_cols.copy()
        if len(bands) == 1:
            band_columns.extend(stats)
            band_stats_df = stats_df[band_columns].copy()
        else:
            band_columns.extend(
                [
                    f"band_{index}_{stat}"
                    for stat in [stat.split("(")[0] for stat in stats]
                ]
            )
            band_stats_df = stats_df[band_columns].copy()
            band_stats_df.rename(
                columns={
                    f"band_{index}_{stat}": stat
                    for stat in [stat.split("(")[0] for stat in stats]
                },
                inplace=True,
            )
        # Add fid column to the beginning of the dataframe
        band_stats_df.insert(0, "fid", range(len(band_stats_df)))

        if band in output_paths:
            logger.info(
                f"Write data for {len(band_stats_df.index)} parcels found to {output_paths[band]}"  # noqa: E501
            )
            if not output_paths[band].exists():
                # Write the info table to the output file
                pdh.to_file(df=band_stats_df, path=output_paths[band], index=False)

                # Write the parameters table to the output file
                spatial_aggregation_args_df = pd.DataFrame(
                    data=stats, columns=["stats"]
                )
                pdh.to_file(
                    df=spatial_aggregation_args_df,
                    path=output_paths[band],
                    table_name="params",
                    index=False,
                    append=True,
                )

    return output_paths
