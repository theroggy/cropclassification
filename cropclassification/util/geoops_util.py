from concurrent import futures
import datetime
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
from osgeo import gdal

# ... and suppress errors
gdal.PushErrorHandler("CPLQuietErrorHandler")
import rasterio
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
    # TODO: GDAL logging was disabled to get rid of this warning, fix it.
    # Warning 1: TIFFReadDirectory:Sum of Photometric type-related color channels and
    # ExtraSamples doesn't match SamplesPerPixel. Defining non-color channels as
    # ExtraSamples.

    # Init
    if output_path.exists():
        if force is True:
            gfo.remove(output_path)
        else:
            logger.info(f"Output file exists already and force is False: {output_path}")
            return

    start_time = datetime.datetime.now()
    tmp_dir = _io_util.create_tempdir("zonal_stats")
    output_temp_path = tmp_dir / f"{output_path.name}"

    info = gfo.get_layerinfo(input_vector_path)

    max_nb_per_batch = 10000
    nb_batches = math.ceil(info.featurecount / max_nb_per_batch)
    nb_per_batch = math.floor(info.featurecount / nb_batches)
    nb_batches_done = 0

    nb_parallel = multiprocessing.cpu_count()
    future_infos = {}

    logger.info(
        f"calculate zonal_stats in {nb_batches} batches, with {nb_parallel} workers, "
        f"for {input_vector_path.name} on {input_raster_path.name}"
    )
    try:
        with futures.ProcessPoolExecutor(
            max_workers=nb_parallel, initializer=setprocessnice(15)
        ) as calculate_pool:
            # calculate per batch to manage memory usage
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
                        gfo.append_to(
                            src=output_temp_partial_path,
                            dst=output_temp_path,
                        )
                        gfo.remove(output_temp_partial_path)

                except Exception:
                    # calculate_pool.shutdown()
                    logger.exception(f"Error executing {batch_id}")
                    raise

                nb_batches_done += 1
                progress = format_progress(
                    start_time=start_time,
                    nb_done=nb_batches_done,
                    nb_todo=nb_batches,
                    operation="zonal_stats",
                    nb_parallel=nb_parallel,
                )
                print(progress)
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
    result_gdf = gpd.GeoDataFrame.from_features(result, crs=vector_gdf.crs)
    gfo.to_file(result_gdf, output_path)

    return True


def format_progress(
    start_time: datetime.datetime,
    nb_done: int,
    nb_todo: int,
    operation: Optional[str] = None,
    nb_parallel: int = 1,
) -> Optional[str]:
    # Init
    time_passed = (datetime.datetime.now() - start_time).total_seconds()
    pct_progress = 100.0 - (nb_todo - nb_done) * 100 / nb_todo
    nb_todo_str = f"{nb_todo:n}"
    nb_decimal = len(nb_todo_str)

    # If we haven't really started yet, don't report time estimate yet
    if nb_done == 0:
        return (
            f" ?: ?: ? left, {operation} done on {nb_done:{nb_decimal}n} of "
            f"{nb_todo:{nb_decimal}n} ({pct_progress:3.2f}%)    "
        )
    else:
        pct_progress = 100.0 - (nb_todo - nb_done) * 100 / nb_todo
        if time_passed > 0:
            # Else, report progress properly...
            processed_per_hour = (nb_done / time_passed) * 3600
            # Correct the nb processed per hour if running parallel
            if nb_done < nb_parallel:
                processed_per_hour = round(processed_per_hour * nb_parallel / nb_done)
            hours_to_go = (int)((nb_todo - nb_done) / processed_per_hour)
            min_to_go = (int)((((nb_todo - nb_done) / processed_per_hour) % 1) * 60)
            secs_to_go = (int)(
                ((((nb_todo - nb_done) / processed_per_hour) % 1) * 3600) % 60
            )
            time_left_str = f"{hours_to_go:02d}:{min_to_go:02d}:{secs_to_go:02d}"
            nb_left_str = f"{nb_done:{nb_decimal}n} of {nb_todo:{nb_decimal}n}"
            pct_str = f"({pct_progress:3.2f}%)    "
        elif pct_progress >= 100:
            time_left_str = "00:00:00"
            nb_left_str = f"{nb_done:{nb_decimal}n} of {nb_todo:{nb_decimal}n}"
            pct_str = f"({pct_progress:3.2f}%)    "
        else:
            return None
        message = f"{time_left_str} left, {operation} done on {nb_left_str} {pct_str}"
        return message
