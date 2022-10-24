from concurrent import futures
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
from time import time
from typing import Optional

import geofileops as gfo
import geopandas as gpd
from geofileops.util import _io_util
import psutil
from psutil._common import bytes2human
import rasterstats

logger = logging.getLogger()


def zonal_stats(
    input_vector_path: Path,
    input_raster_path: Path,
    output_path: Path,
    input_vector_layer: Optional[str] = None,
    band: int = 1,
    nodata=None,
    affine=None,
    stats: Optional[list] = None,
    all_touched: bool = False,
    categorical: bool = False,
    category_map=None,
    add_stats=None,
    zone_func=None,
    # raster_out: bool = False,
    prefix: Optional[str] = None,
    boundless: bool = True,
    force: bool = False,
):

    # Init
    if output_path.exists():
        if force is True:
            gfo.remove(output_path)
        else:
            logger.info(f"Output file exists already and force is False: {output_path}")
            return

    tmp_dir = _io_util.create_tempdir("zonal_stats")
    output_temp_path = tmp_dir / f"{output_path.name}"

    info = gfo.get_layerinfo(input_vector_path)
    max_nb_per_batch = 10000
    nb_batches = math.ceil(info.featurecount / max_nb_per_batch)
    nb_per_batch = int(info.featurecount / nb_batches)

    nb_parallel = multiprocessing.cpu_count()
    future_infos = {}

    logger.info(
        f"calculate zonal_stats for {input_vector_path.name}, in {nb_batches} batches, "
        f"with {nb_parallel} workers"
    )
    try:
        with futures.ProcessPoolExecutor(
            max_workers=nb_parallel, initializer=setprocessnice(15)
        ) as calculate_pool:
            # calculate per batch  to manage memory usage
            batch_start = 0
            for batch_id in range(nb_batches):
                # Prepare the slice for this batch
                if batch_id < nb_batches - 1:
                    rows = slice(batch_start, batch_start + nb_per_batch)
                else:
                    rows = slice(batch_start)

                output_temp_partial_path = (
                    output_temp_path.parent / f"{output_temp_path.stem}_{batch_id}.gpkg"
                )
                future = calculate_pool.submit(
                    zonal_stats_ext,
                    vector_path=input_vector_path,
                    raster_path=input_raster_path,
                    output_path=output_temp_partial_path,
                    layer=input_vector_layer,
                    band=band,
                    nodata=nodata,
                    affine=affine,
                    stats=stats,
                    all_touched=all_touched,
                    categorical=categorical,
                    category_map=category_map,
                    add_stats=add_stats,
                    zone_func=zone_func,
                    # raster_out: bool = False,
                    prefix=prefix,
                    geojson_out=True,
                    boundless=boundless,
                    rows=rows,
                )
                future_infos[future] = {
                    "batch_id": batch_id,
                    "output_temp_partial_path": output_temp_partial_path,
                }
                batch_start += nb_per_batch

            # Loop till all parallel processes are ready, but process each one
            # that is ready already
            batch_id = None
            for future in futures.as_completed(future_infos):
                try:
                    future_info = future_infos[future]
                    batch_id = future_info["batch_id"]
                    output_temp_partial_path = future_info["output_temp_partial_path"]
                    _ = future.result()

                    # If the calculate gave results, copy to output
                    if (
                        output_temp_partial_path.exists()
                        and output_temp_partial_path.stat().st_size > 0
                    ):
                        logger.info(f"Write result for batch {future_info['batch_id']}")
                        gfo.append_to(
                            src=output_temp_partial_path,
                            dst=output_temp_path,
                        )
                        gfo.remove(output_temp_partial_path)

                except Exception:
                    # calculate_pool.shutdown()
                    logger.exception(f"Error executing {batch_id}")
                    raise

                logger.info(
                    "Available virtual memory: "
                    f"{bytes2human(psutil.virtual_memory().available)}, "
                    f"pct virtual memory used: {psutil.virtual_memory().percent}"
                )

            # We are ready, so move the result to the permanent location
            if output_temp_path.exists():
                gfo.move(output_temp_path, output_path)
    finally:
        shutil.rmtree(output_temp_path.parent)


def setprocessnice(nice_value: int):
    p = psutil.Process(os.getpid())
    if os.name == "nt":
        p.nice(process_nice_to_priority_class(nice_value))
    else:
        p.nice(nice_value)


def process_nice_to_priority_class(nice_value: int) -> int:
    if nice_value <= -15:
        return psutil.REALTIME_PRIORITY_CLASS
    elif nice_value <= -10:
        return psutil.HIGH_PRIORITY_CLASS
    elif nice_value <= -5:
        return psutil.ABOVE_NORMAL_PRIORITY_CLASS
    elif nice_value <= 0:
        return psutil.NORMAL_PRIORITY_CLASS
    elif nice_value <= 10:
        return psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        return psutil.IDLE_PRIORITY_CLASS


def zonal_stats_ext(
    vector_path: Path,
    raster_path: Path,
    output_path: Path,
    layer: Optional[str] = None,
    band: int = 1,
    nodata=None,
    affine=None,
    stats: Optional[list] = None,
    all_touched: bool = False,
    categorical: bool = False,
    category_map=None,
    add_stats=None,
    zone_func=None,
    # raster_out: bool = False,
    prefix: Optional[str] = None,
    geojson_out: bool = False,
    boundless: bool = True,
    rows: Optional[slice] = None,
) -> bool:
    # If there isn't enough memory available (1 GB), sleep a while
    while True:
        if psutil.virtual_memory().available > 1000000000:
            break
        time.sleep(1)

    # Read the rows we want to process in this batch
    vector_gdf = gfo.read_file(path=vector_path, layer=layer, rows=rows)

    # Calculate
    result = rasterstats.zonal_stats(
        vectors=vector_gdf,
        raster=raster_path,
        band=band,
        nodata=nodata,
        affine=affine,
        stats=stats,
        all_touched=all_touched,
        categorical=categorical,
        category_map=category_map,
        add_stats=add_stats,
        zone_func=zone_func,
        # raster_out=raster_out,
        prefix=prefix,
        geojson_out=geojson_out,
        boundless=boundless,
    )
    result_gdf = gpd.GeoDataFrame.from_features(result)
    gfo.to_file(result_gdf, output_path)

    return True
