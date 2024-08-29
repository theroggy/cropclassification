import logging
import multiprocessing
import os
import signal
import sys
from concurrent import futures
from datetime import datetime
from pathlib import Path
from typing import Union

import exactextract
import geopandas as gpd
import pandas as pd
import psutil
import rasterio

from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util import io_util

from . import Statistic
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
    stats: list[Statistic],
    nb_parallel: int = -1,
    force: bool = False,
):
    """
    Calculate zonal statistics.

    Args:
        features_path (Path): _description_
        id_column (str): _description_
        images_bands (List[Tuple[Path, List[str]]]): _description_
        stats (List[Statistic]): statistics to calculate.
        output_dir (Path): _description_
        nb_parallel (int, optional): the number of parallel processes to use.
            Defaults to -1: use all available processors.
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
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
            bands_todo = {}

            for band in bands:
                # Prepare the output paths...
                output_band_path = general_helper._format_output_path(
                    vector_path, raster_path, output_dir, orbit, band
                )
                output_band_busy_path = tmp_dir / f"BUSY_{output_band_path.name}"

                # If a busy output file exists, remove it, otherwise we can get
                # double data in it...
                if output_band_busy_path.exists():
                    output_band_busy_path.unlink()

                # Check if the output file exists already
                if output_band_path.exists():
                    if not force:
                        logger.debug(f"Output file for band exists {output_band_path}")
                        continue
                    else:
                        output_band_path.unlink()
                bands_todo[band] = output_band_path

                # If all bands already processed, skip image...
                if len(bands_todo) == 0:
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
                    band=band,
                    stats=stats,
                    tmp_dir=tmp_dir,
                    columns=columns,
                    output_band_path=output_band_path,
                )
                calc_queue[future] = {
                    "vector_path": vector_path,
                    "raster_path": raster_path,
                    "bands": bands,
                    "bands_todo": bands_todo,
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
    band: str,
    tmp_dir: Path,
    stats: list[Statistic],
    columns: list[str],
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    # Init
    stats_mask = None
    for stat in stats:
        if stats_mask is None:
            stats_mask = stat_to_exactextract_stat(stat)
        else:
            stats_mask |= stat_to_exactextract_stat(stat)

    # Get the image info
    image_info = raster_helper.get_image_info(raster_path)

    # Reproject the vector data
    tmp_dir.mkdir(exist_ok=True, parents=True)
    vector_proj_path = vector_helper.reproject_synced(
        path=vector_path,
        columns=columns + ["geometry"],
        target_epsg=image_info.image_epsg,
        dst_dir=tmp_dir,
    )
    # layer = gfo.get_only_layer(vector_proj_path)

    # Calculates zonal stats with raster
    raster = Path(image_info.bands[band].path)

    try:
        # HACK: split raster in individual bands
        with rasterio.open(raster) as src:
            # Iterate over each band
            # for i in range(1, src.count + 1):  # rasterio bands are 1-indexed
            # Read the band
            band = src.read(image_info.bands[band].band_index)

            # Create output file name
            raster_single_band = tmp_dir / raster.name

            # Define the metadata for the output file
            out_meta = src.meta.copy()
            out_meta.update({"count": 1})

            # Write the band to a new file
            with rasterio.open(raster_single_band, "w", **out_meta) as dest:
                dest.write(band, 1)

        stats_df = exactextract.exact_extract(
            rast=raster_single_band,
            vec=vector_proj_path,
            ops=stats_mask,
            # strategy="raster-sequential",
            include_geom=False,
            output="pandas",
            include_cols=["index", "UID", "x_ref"],
        )
    except Exception:
        raise

    return stats_df


def zonal_stats_band_tofile(
    vector_path,
    raster_path: Path,
    output_band_path: Path,
    band: str,
    tmp_dir: Path,
    stats: list[Statistic],
    columns: list[str],
    force: bool = False,
) -> Path:
    if output_band_path.exists():
        if force:
            output_band_path.unlink()
        else:
            return output_band_path

    stats_df = zonal_stats_band(
        vector_path=vector_path,
        raster_path=raster_path,
        band=band,
        stats=stats,
        tmp_dir=tmp_dir,
        columns=columns,
    )

    # Add fid column
    stats_df["fid"] = range(len(stats_df))
    # Reorder columns
    columns = ["fid", "index", "UID", "x_ref"]
    columns.extend(stats)
    stats_df = stats_df[columns]

    # Remove rows with empty data
    # stats_df.dropna(inplace=True)

    logger.info(
        f"Write data for {len(stats_df.index)} parcels found to {output_band_path}"
    )
    pdh.to_file(stats_df, output_band_path, index=False)
    return output_band_path


def stat_to_exactextract_stat(stat: str):
    if stat in [
        "cell_id",
        "center_x",
        "center_y",
        "coefficient_of_variation",
        "count",
        "coverage",
        "frac",
        "majority",
        "max",
        "max_center_x",
        "max_center_y",
        "mean",
        "median",
        "min",
        "min_center_x",
        "min_center_y",
        "minority",
        "quantile",
        "stdev",
        "sum",
        "unique",
        "values",
        "variance",
        "variety",
        "weighted_frac",
        "weighted_mean",
        "weighted_stdev",
        "weighted_sum",
        "weighted_variance",
        "weights",
    ]:
        return stat
    elif stat == "std":
        return "stdev"
    elif stat == "range":
        return "values"
    else:
        raise ValueError(f"unsupported value in stats: {stat}")
