# -*- coding: utf-8 -*-
"""
Calculate timeseries data per image.
"""

from concurrent import futures
from datetime import datetime
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal  # To catch CTRL-C explicitly and kill children
import sys
import time
from typing import List, Optional, Tuple, Union

from osgeo import gdal

# Suppress gdal warnings/errors
gdal.PushErrorHandler("CPLQuietErrorHandler")

from affine import Affine
import geopandas as gpd
import geofileops as gfo
import pandas as pd
import psutil  # To catch CTRL-C explicitly and kill children
from rasterstats import zonal_stats

from cropclassification.helpers import pandas_helper as pdh
from cropclassification.helpers import raster_helper
from cropclassification.util import io_util

# General init
logger = logging.getLogger(__name__)


# TODO: performantie van rasterstats vergelijken met wat andere bibs:
#   - rastrio manual: https://pysal.org/scipy2019-intermediate-gds/deterministic/gds2-rasters.html
#   - pygeoprocessing (open?): https://pygeoprocessing.readthedocs.io/en/latest/api/pygeoprocessing.html#pygeoprocessing.zonal_statistics
#   - pyjeo (GPL): https://pyjeo.readthedocs.io/en/latest/2_tutorial.html#tutorial-on-extract-calculating-regional-statistics
#   - pyQGIS (GPL): https://gis.stackexchange.com/questions/421556/performance-problem-with-getting-average-pixel-values-within-buffered-circles
#       -> is snel, maar lastig om te installeren op een server.
#       -> src: https://github.com/qgis/QGIS/blob/d5626d92360efffb4b8085389c8d64072ef65833/src/analysis/vector/qgszonalstatistics.cpp#L266
#   - https://pygis.io/docs/f_rs_extraction.html
#
# Dit zijn bibs die ook gedeeltelijke pixels meenemen op basis van opp overlapping,
# wat niet ideaal is hiervoor...
#   - xagg: https://github.com/ks905383/xagg
#   - exactextract (apache2): https://github.com/isciences/exactextract
def calc_stats_per_image(
    features_path: Path,
    id_column: str,
    images_bands: List[Tuple[Path, List[str]]],
    output_dir: Path,
    temp_dir: Path,
    log_dir: Path,
    log_level: Union[str, int],
    cloud_filter_band: Optional[str] = None,
    calc_bands_parallel: bool = True,
    force: bool = False,
):
    """
    Calculate the statistics.

    Args:
        features_path (Path): _description_
        id_column (str): _description_
        images_bands (List[Tuple[Path, List[str]]]): _description_
        output_dir (Path): _description_
        temp_dir (Path): _description_
        log_dir (Path): _description_
        log_level (Union[str, int]): _description_
        force (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """

    # TODO: probably need to apply some object oriented approach here for "image",
    # because there are to many properties,... to be clean/clear this way.
    # TODO: maybe passing the executor pool to a calc_stats_for_image function can have
    # both the advantage of not creating loads of processes + keep the cleanup logic
    # after calculation together with the processing logic

    # Some checks on the input parameters
    nb_todo = len(images_bands)
    if nb_todo == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = io_util.create_tempdir("calc_stats_per_image", parent_dir=temp_dir)

    # Create process pool for parallelisation...
    nb_parallel_max = multiprocessing.cpu_count()
    nb_parallel = nb_parallel_max

    # Loop over all images to start the data preparation for each of them in
    # parallel...
    start_time = datetime.now()
    nb_errors_max = 10
    nb_errors = 0

    image_dict = {}
    calc_stats_batch_dict = {}
    nb_done_total = 0
    image_idx = 0

    pool = futures.ProcessPoolExecutor(nb_parallel)
    try:
        # Keep looping. At the end of the loop there are checks when to
        # break out of the loop...
        while True:
            # Start preparation for calculation on next image + features combo
            # If not all images aren't prepared for processing yet, and there
            # aren't too many busy yet, prepare the next one
            if (
                image_idx < len(images_bands)
                and len(filter_on_status(image_dict, "IMAGE_PREPARE_CALC_BUSY"))
                < nb_parallel_max
            ):
                # Not too many busy preparing, so get next image_path to start
                # prepare on
                image_path, bands = images_bands[image_idx]
                image_path_str = str(image_path)
                image_idx += 1

                image_info = raster_helper.get_image_info(image_path)
                # Create base output filename
                # TODO: hoort hier niet echt thuis
                orbit = None
                if image_info.imagetype.lower() == "s1-grd-sigma0-asc":
                    orbit = "ASCENDING"
                elif image_info.imagetype.lower() == "s1-grd-sigma0-desc":
                    orbit = "DESCENDING"
                output_base_path = format_output_path(
                    features_path,
                    image_path,
                    output_dir,
                    orbit,
                    band=None,
                )
                output_base_busy_path = temp_dir / f"BUSY_{output_base_path.name}"

                # Check for which bands there is a valid output file already
                if bands is None:
                    bands = image_info.bands
                    assert bands is not None
                bands_done = 0
                for band in bands:
                    # Prepare the output paths...
                    output_band_path = format_output_path(
                        features_path,
                        image_path,
                        output_dir,
                        orbit,
                        band,
                    )
                    output_band_busy_path = temp_dir / f"BUSY_{output_band_path.name}"

                    # If a busy output file exists, remove it, otherwise we can get
                    # double data in it...
                    if output_band_busy_path.exists():
                        output_band_busy_path.unlink()

                    # Check if the output file exists already
                    if output_band_path.exists():
                        if not force:
                            logger.debug(
                                f"Output file for band exists {output_band_path}"
                            )
                            bands_done += 1
                        else:
                            output_band_path.unlink()

                # If all bands already processed, skip image...
                if len(bands) == bands_done:
                    logger.info(
                        "SKIP image: output files for all bands exist already for "
                        f"{output_base_path}"
                    )
                    nb_todo -= 1
                    continue

                # Start the prepare processing assync
                # TODO: possibly it is cleaner to do this per band...
                logger.info(f"Start calculation for image {output_base_path}")
                future = pool.submit(
                    prepare_calc,
                    features_path,
                    id_column,
                    image_path,
                    temp_dir,
                    log_dir,
                    log_level,
                    nb_parallel_max,
                )
                image_dict[image_path_str] = {
                    "features_path": features_path,
                    "image_path": image_path,
                    "bands": bands,
                    "prepare_calc_future": future,
                    "image_info": image_info,
                    "prepare_calc_starttime": datetime.now(),
                    "output_base_path": output_base_path,
                    "output_base_busy_path": output_base_busy_path,
                    "status": "IMAGE_PREPARE_CALC_BUSY",
                }

                # Jump to next image to start the prepare_calc for it...
                continue

            # For images busy preparing: if ready: start real calculation in batches

            # Loop through the busy preparations
            for image_path_str in filter_on_status(
                image_dict, "IMAGE_PREPARE_CALC_BUSY"
            ):
                # If still running, go to next
                image = image_dict[image_path_str]
                if image["prepare_calc_future"].running():
                    continue

                # Extract the result from the preparation
                try:
                    # Get the result from the completed  prepare_inputs
                    prepare_calc_result = image["prepare_calc_future"].result()

                    # If nb_features to be treated is 0... create (empty) output
                    # files and continue with next...
                    if prepare_calc_result["nb_features_to_calc_total"] == 0:
                        for band in image["bands"]:
                            # Prepare the output path...
                            orbit = None
                            if image["image_info"].imagetype.upper() in (
                                "S1-ASC",
                                "S1-DESC",
                            ):
                                orbit = image["image_info"].extra[
                                    "orbit_properties_pass"
                                ]
                            output_band_path = format_output_path(
                                features_path,
                                image["image_path"],
                                output_dir,
                                orbit,
                                band,
                            )

                            # Create output file
                            logger.info(
                                "No features found in image: create done file: "
                                f"{output_band_path}"
                            )
                            io_util.create_file_atomic(output_band_path)
                            image["status"] = "IMAGE_CALC_DONE"

                        # Jump to next image
                        continue

                    # Add info about the result of the prepare_calc to image info
                    image["image_prepared_path"] = prepare_calc_result[
                        "image_prepared_path"
                    ]
                    image["feature_batches"] = prepare_calc_result["feature_batches"]
                    image["nb_features_to_calc_total"] = prepare_calc_result[
                        "nb_features_to_calc_total"
                    ]
                    image["temp_features_dir"] = prepare_calc_result[
                        "temp_features_dir"
                    ]

                    # Set status to calc_is_busy so we know calculation is busy...
                    image["status"] = "IMAGE_CALC_BUSY"
                    image["calc_starttime"] = datetime.now()

                    # Now loop through all prepared feature batches to start the
                    # statistics calculation for each
                    logger.info(f"Start statistics calculation for {image_path_str}")
                    for features_batch in image["feature_batches"]:
                        bands = image["bands"]
                        # If bands shouldn't be calculated in parallel...
                        if not calc_bands_parallel:
                            bands = [image["bands"]]

                        for band in bands:
                            start_time_batch = datetime.now()
                            future = pool.submit(
                                calc_stats_image_gdf,
                                features_batch["path"],
                                id_column,
                                image["image_prepared_path"],
                                band,
                                image["output_base_busy_path"],
                                log_dir,
                                log_level,
                                start_time_batch,
                                cloud_filter_band,
                            )
                            band_id = band if isinstance(band, str) else "-".join(band)
                            batch_id = f"{features_batch['id']}_{band_id}"
                            calc_stats_batch_dict[batch_id] = {
                                "calc_stats_future": future,
                                "image_path": image["image_path"],
                                "image_prepared_path": image["image_prepared_path"],
                                "start_time_batch": start_time_batch,
                                "nb_items_batch": features_batch["nb_items"],
                                "status": "BATCH_CALC_BUSY",
                            }

                except Exception as ex:
                    message = f"Exception getting result of prepare_calc for {image}"
                    logger.exception(message)
                    nb_errors += 1
                    if nb_errors > nb_errors_max:
                        raise Exception(message) from ex

            # For batches with calc busy, check if there are ready ones
            for calc_stats_batch_id in filter_on_status(
                calc_stats_batch_dict, "BATCH_CALC_BUSY"
            ):
                # If it is not done yet, continue
                calc_stats_batch_info = calc_stats_batch_dict[calc_stats_batch_id]
                if calc_stats_batch_info["calc_stats_future"].done() is False:
                    continue

                try:
                    # Get the result
                    result = calc_stats_batch_info["calc_stats_future"].result()
                    if not result:
                        raise Exception("Returned False?")

                    # Set the processed flag to True
                    calc_stats_batch_info["status"] = "BATCH_CALC_DONE"
                    logger.debug(
                        "Ready processing batch of "
                        f"{calc_stats_batch_info['nb_items_batch']} for features "
                        f"image: {calc_stats_batch_info['image_path']}"
                    )

                except Exception as ex:
                    message = (
                        "Exception getting result of calc_stats_image_gdf for "
                        f"{calc_stats_batch_info}"
                    )
                    logger.exception(message)
                    nb_errors += 1
                    if nb_errors > nb_errors_max:
                        raise Exception(message) from ex

            # For images with calc busy, check if they are ready

            # Loop through busy image calculations
            for image_path_str in filter_on_status(image_dict, "IMAGE_CALC_BUSY"):
                # If still batches busy for this image, continue to next image
                batches_busy = False
                for calc_stats_batch_id in filter_on_status(
                    calc_stats_batch_dict, "BATCH_CALC_BUSY"
                ):
                    path = calc_stats_batch_dict[calc_stats_batch_id]["image_path"]
                    if str(path) == image_path_str:
                        batches_busy = True
                        break
                if batches_busy:
                    continue

                # All batches are done, so the image is done
                image = image_dict[image_path_str]
                image["status"] = "IMAGE_CALC_DONE"

                # If the preprocessing created a temp image file, clean it up
                image_prepared_path = image["image_prepared_path"]
                if str(image_prepared_path) != str(image_path_str):
                    logger.info(f"Remove local temp image copy: {image_prepared_path}")
                    if image_prepared_path.is_dir():
                        shutil.rmtree(image_prepared_path, ignore_errors=True)
                    else:
                        image_prepared_path.unlink()

                # If the preprocessing created temp pickle files with features,
                # clean them up
                # shutil.rmtree(image["temp_features_dir"], ignore_errors=True)

                # Move the (completed) output files
                output_base_path = Path(image["output_base_path"])
                output_base_busy_path = Path(image["output_base_busy_path"])
                for band in image["bands"]:
                    # TODO: creating the output paths should probably be cleaner
                    # centralised
                    output_band_path = (
                        output_base_path.parent
                        / f"{output_base_path.stem}_{band}{output_base_path.suffix}"
                    )
                    suffix = output_base_busy_path.suffix
                    output_band_busy_path = (
                        output_base_busy_path.parent
                        / f"{output_base_busy_path.stem}_{band}{suffix}"
                    )

                    # If BUSY output file exists, move it (rename cannot move to
                    # different file volume)
                    if output_band_busy_path.exists():
                        shutil.move(output_band_busy_path, output_band_path)
                    else:
                        # If BUSY output file doesn't exist, create empty file
                        logger.info(
                            "No features found overlapping image after processing, "
                            f"create done file: {output_band_path}"
                        )
                        io_util.create_file_atomic(output_band_path)

                # Log the progress and prediction speed
                logger.info(f"Ready processing image: {image_path_str}")
                nb_done_latestbatch = 1
                nb_done_total += nb_done_latestbatch
                progress_msg = format_progress_message(
                    nb_todo,
                    nb_done_total,
                    nb_done_latestbatch,
                    start_time,
                    image["calc_starttime"],
                )
                logger.info(progress_msg)

            # Check if we are completely ready

            # This is the case if:
            #     - no processing is needed (= empty image_dict)
            #     - OR if all processing is started + is done
            #       (= status IMAGE_CALC_DONE)
            if len(image_dict) == 0 or (
                image_idx == nb_todo
                and len(image_dict)
                == len(filter_on_status(image_dict, "IMAGE_CALC_DONE"))
            ):
                if nb_errors == 0:
                    # Ready, so jump out of unending while loop
                    break
                else:
                    raise Exception(
                        f"Ready processing, but there were {nb_errors} errors!"
                    )

            # Some sleep before starting next iteration...
            time.sleep(0.1)

    except KeyboardInterrupt:
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

    logger.info(
        f"Time taken to calculate data for {nb_todo} images: "
        f"{(datetime.now()-start_time).total_seconds()} sec"
    )


def filter_on_status(dict: dict, status_to_check: str) -> List[str]:
    """
    Function to check the number of images that are being prepared for processing
    """
    keys_with_status = []
    for key in dict:
        if dict[key]["status"] == status_to_check:
            keys_with_status.append(key)
    return keys_with_status


def format_output_path(
    features_path: Path,
    image_path: Path,
    output_dir: Path,
    orbit_properties_pass: Optional[str],
    band: Optional[str],
) -> Path:
    """
    Prepare the output path.
    """
    # Interprete the orbit...
    if orbit_properties_pass is not None:
        if orbit_properties_pass == "ASCENDING":
            orbit = "_ASC"
        elif orbit_properties_pass == "DESCENDING":
            orbit = "_DESC"
        else:
            message = f"Unknown orbit_properties_pass: {orbit_properties_pass}"
            logger.error(message)
            raise Exception(message)
    else:
        orbit = ""

    # Format + return output path
    output_stem = f"{features_path.stem}__{image_path.stem}{orbit}"
    if band is not None:
        output_stem = f"{output_stem}_{band}"
    output_path = output_dir / f"{output_stem}.sqlite"
    return output_path


def format_progress_message(
    nb_todo: int,
    nb_done_total: int,
    nb_done_latestbatch: int,
    start_time: datetime,
    start_time_latestbatch: datetime,
) -> str:
    """
    Returns a progress message based on the input.

    Args
        nb_todo: total number of items that need(ed) to be processed
        nb_done_total: total number of items that have been processed already
        nb_done_latestbatch: number of items that were processed in the latest batch
        start_time: datetime the processing started
        start_time_latestbatch: datetime the latest batch started
    """
    time_passed_s = (datetime.now() - start_time).total_seconds()
    time_passed_latestbatch_s = (
        datetime.now() - start_time_latestbatch
    ).total_seconds()

    # Calculate the overall progress
    large_number = 9999999999
    if time_passed_s > 0:
        nb_per_hour = (nb_done_total / time_passed_s) * 3600
    else:
        nb_per_hour = large_number
    hours_to_go = (int)((nb_todo - nb_done_total) / nb_per_hour)
    min_to_go = (int)((((nb_todo - nb_done_total) / nb_per_hour) % 1) * 60)

    # Calculate the speed of the latest batch
    if time_passed_latestbatch_s > 0:
        nb_per_hour_latestbatch = (
            nb_done_latestbatch / time_passed_latestbatch_s
        ) * 3600
    else:
        nb_per_hour_latestbatch = large_number

    # Return formatted message
    message = (
        f"{hours_to_go}:{min_to_go} left for {nb_todo-nb_done_total} todo at "
        f"{nb_per_hour:0.0f}/h ({nb_per_hour_latestbatch:0.0f}/h last batch)"
    )
    return message


def prepare_calc(
    features_path: Path,
    id_column: str,
    image_path: Path,
    temp_dir: Path,
    log_dir: Path,
    log_level: Union[str, int],
    nb_parallel_max: int = 16,
) -> dict:
    """
    Prepare the inputs for a calculation.

    Returns True if succesfully completed.
    Remark: easiest it returns something, when used in a parallel way:
    concurrent.futures likes it better if something is returned
    """
    # When running in parallel processes, the logging needs to be write to seperate
    # files + no console logging
    if len(logging.getLogger().handlers) == 0:
        log_path = (
            log_dir
            / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_prepare_calc_{os.getpid()}.log"
        )
        filehandler = logging.FileHandler(filename=log_path)
        filehandler.setLevel(log_level)
        log_format = "%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s"
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[filehandler],
        )

    global logger
    logger = logging.getLogger("prepare_calc")
    ret_val = {}

    # Prepare the image
    logger.info(f"Start prepare_image for {image_path} to {temp_dir}")
    image_prepared_path = raster_helper.prepare_image(image_path, temp_dir)
    logger.debug(f"Preparing ready, result: {image_prepared_path}")
    ret_val["image_prepared_path"] = image_prepared_path

    # Get info about the image
    logger.info(f"Start get_image_info for {image_prepared_path}")
    image_info = raster_helper.get_image_info(image_prepared_path)
    logger.info(f"image_info: {image_info}")

    # Load the features that overlap with the image.
    # TODO: passing both bbox and poly is double, or not?
    # footprint epsg should be passed as well, or reproject here first?
    footprint_shape = None
    if image_info.footprint is not None:
        logger.info(f"poly: {image_info.footprint['shape']}")
        footprint_shape = image_info.footprint["shape"]

    if image_info.image_epsg == "NONE":
        raise Exception(f"target_epsg == NONE: {image_info}")

    features_gdf = load_features_file(
        features_path=features_path,
        target_epsg=image_info.image_epsg,
        columns_to_retain=[id_column, "geometry"],
        bbox=image_info.image_bounds,
        polygon=footprint_shape,
    )

    # Check if overlapping features were found, otherwise no use to proceed
    nb_todo = len(features_gdf.index)
    ret_val["nb_features_to_calc_total"] = nb_todo
    if nb_todo == 0:
        logger.info(f"No features found in the bounding box of the image: {image_path}")
        return ret_val

    # Calculate the number per batch, but keep the number between 100 and 50000...
    nb_per_batch = min(max(math.ceil(nb_todo / nb_parallel_max), 100), 50000)

    # The features were already sorted on x coordinate, so the features in the batches
    # are already clustered geographically
    features_gdf_batches = [
        features_gdf.loc[i : i + nb_per_batch - 1, :]
        for i in range(0, nb_todo, nb_per_batch)
    ]

    # Pickle the batches to temporary files
    # Create temp dir to put the pickles in... and clean or create it.
    # TODO: change dir so it is always unique
    temp_features_dir = temp_dir / image_path.stem
    ret_val["temp_features_dir"] = temp_features_dir
    if temp_features_dir.exists():
        logger.info(f"Remove dir {str(temp_features_dir)}{os.sep}")
        shutil.rmtree(f"{str(temp_features_dir)}{os.sep}")
    temp_features_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the batches, pickle them and add the paths to the result...
    ret_val["feature_batches"] = []
    for batch_idx, features_gdf_batch in enumerate(features_gdf_batches):
        batch_info = {}
        pickle_path = temp_features_dir / f"{batch_idx}.pkl"
        logger.info(
            f"Write pkl of {len(features_gdf_batch.index)} features: {pickle_path}"
        )
        try:
            features_gdf_batch.to_pickle(pickle_path)
        except Exception as ex:
            raise Exception(f"Exception writing pickle: {pickle_path}: {ex}") from ex
        batch_info["path"] = pickle_path
        batch_info["nb_items"] = len(features_gdf_batch.index)
        batch_info["id"] = f"{image_path.stem}_{batch_idx}"
        ret_val["feature_batches"].append(batch_info)

    # Ready... return
    return ret_val


def load_features_file(
    features_path: Path,
    columns_to_retain: List[str],
    target_epsg: int,
    bbox=None,
    polygon=None,
) -> gpd.GeoDataFrame:
    """
    Load the features and reproject to the target crs.

    Remarks:
        * Reprojected version is "cached" so on a next call, it can be directly read.
        * Locking and waiting is used to ensure correct results if used in parallel.

    Args
        features_path:
        columns_to_retain:
        target_srs:
        bbox: bounds of the area to be loaded, in the target_epsg
    """
    # Load parcel file and preprocess it: remove excess columns + reproject if needed.
    # By convention, the features filename should end on the projection... so extract
    # epsg from filename
    start_time = datetime.now()
    """
    features_epsg = None
    if features_path.stem.find("_") != -1:
        splitted = features_path.stem.split("_")
        features_epsg = splitted[len(splitted) - 1]
        try:
            features_epsg = int(features_epsg)
        except Exception:
            logger.info(
                f"features_epsg {features_epsg } could not be cast to int for "
                f"{features_path.name}"
            )
            features_epsg = None
    """
    features_epsg = gfo.get_layerinfo(features_path).crs.to_epsg()

    # Determine the correct filename for the input features in the correct projection.
    if features_epsg != target_epsg:
        features_prepr_path = (
            features_path.parent / f"{features_path.stem}_{target_epsg:.0f}.gpkg"
        )
    else:
        features_prepr_path = features_path

    # Prepare filename for a "busy file" to ensure proper behaviour in a parallel
    # processing context
    features_prepr_path_busy = Path(f"{str(features_prepr_path)}_busy")

    # If the file doesn't exist yet in right projection, read original input file to
    # reproject/write to new file with correct epsg
    features_gdf = None
    if not (features_prepr_path_busy.exists() or features_prepr_path.exists()):
        # Create lock file in an atomic way, so we are sure we are the only process
        # working on it. If function returns true, there isn't any other thread/process
        # already working on it
        if io_util.create_file_atomic(features_prepr_path_busy):
            try:
                # Read (all) original features + remove unnecessary columns...
                logger.info(f"Read original file {features_path}")
                start_time = datetime.now()
                logging.getLogger("fiona.ogrext").setLevel(logging.INFO)
                features_gdf = gfo.read_file(features_path)
                logger.info(
                    f"Read ready, found {len(features_gdf.index)} features, "
                    f"crs: {features_gdf.crs}, took "
                    f"{(datetime.now()-start_time).total_seconds()} s"
                )
                for column in features_gdf.columns:
                    if column not in columns_to_retain and column not in [
                        "geometry",
                        "x_ref",
                    ]:
                        features_gdf.drop(columns=column, inplace=True)

                # Reproject them
                logger.info(
                    f"Reproject features from {features_gdf.crs} to epsg {target_epsg}"
                )
                features_gdf = features_gdf.to_crs(epsg=target_epsg)
                logger.info("Reprojected, now sort on x_ref")

                if features_gdf is None:
                    raise Exception("features_gdf is None")
                # Order features on x coordinate
                if "x_ref" not in features_gdf.columns:
                    features_gdf["x_ref"] = features_gdf.geometry.bounds.minx
                features_gdf.sort_values(by=["x_ref"], inplace=True)
                features_gdf.reset_index(inplace=True)

                # Cache the file for future use
                logger.info(
                    f"Write {len(features_gdf.index)} reprojected features to "
                    f"{features_prepr_path}"
                )
                gfo.to_file(
                    features_gdf, features_prepr_path, index=False  # type: ignore
                )
                logger.info("Reprojected features written")

            except Exception as ex:
                # If an exception occurs...
                message = f"Delete possibly incomplete file: {features_prepr_path}"
                logger.exception(message)
                gfo.remove(features_prepr_path)
                raise Exception(message) from ex
            finally:
                # Remove lock file as everything is ready for other processes to use it
                features_prepr_path_busy.unlink()

            # Now filter the parcels that are in bbox provided
            if bbox is not None:
                logger.info(f"bbox provided, so filter features in the bbox of {bbox}")
                xmin, ymin, xmax, ymax = bbox
                features_gdf = features_gdf.cx[xmin:xmax, ymin:ymax]
                logger.info(f"Found {len(features_gdf.index)} features in bbox")

    # If there exists already a file with the features in the right projection, we can
    # just read the data
    if features_gdf is None:
        # If a "busy file" still exists, the file isn't ready yet, but another process
        # is working on it, so wait till it disappears
        wait_secs_max = 600
        wait_start_time = datetime.now()
        while features_prepr_path_busy.exists():
            time.sleep(1)
            wait_secs = (datetime.now() - wait_start_time).total_seconds()
            if wait_secs > wait_secs_max:
                raise Exception(
                    f"Waited {wait_secs} for busy file "
                    f"{features_prepr_path_busy} and it is still there!"
                )

        logger.info(f"Read {features_prepr_path}")
        start_time = datetime.now()
        features_gdf = gfo.read_file(features_prepr_path, bbox=bbox)
        logger.info(
            f"Read ready, found {len(features_gdf.index)} features, crs: "
            f"{features_gdf.crs}, took {(datetime.now()-start_time).total_seconds()} s"
        )

        # Order features on x_ref to (probably) have more clustering of features in
        # further action...
        if "x_ref" not in features_gdf.columns:
            features_gdf["x_ref"] = features_gdf.geometry.bounds.minx
        features_gdf.sort_values(by=["x_ref"], inplace=True)
        features_gdf.reset_index(inplace=True)

        # To be sure, remove the columns anyway...
        for column in features_gdf.columns:
            if column not in columns_to_retain and column not in ["geometry"]:
                features_gdf.drop(columns=column, inplace=True)

    # If there is a polygon provided, filter on the polygon (as well)
    if polygon is not None:
        logger.info("Filter polygon provided, start filter")
        polygon_gdf = gpd.GeoDataFrame(
            geometry=[polygon], crs="EPSG:4326", index=[0]  # type: ignore
        )
        logger.debug(f"polygon_gdf: {polygon_gdf}")
        logger.debug(
            f"polygon_gdf.crs: {polygon_gdf.crs}, features_gdf.crs: {features_gdf.crs}"
        )
        polygon_gdf = polygon_gdf.to_crs(features_gdf.crs)
        assert polygon_gdf is not None
        logger.debug(f"polygon_gdf, after reproj: {polygon_gdf}")
        logger.debug(
            f"polygon_gdf.crs: {polygon_gdf.crs}, features_gdf.crs: {features_gdf.crs}"
        )
        features_gdf = gpd.sjoin(
            features_gdf, polygon_gdf, how="inner", predicate="within"
        )

        # Drop column added by sjoin
        features_gdf.drop(columns="index_right", inplace=True)
        """
        spatial_index = gdf.sindex
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(polygon)]
        """
        logger.info(f"Filter ready, found {len(features_gdf.index)}")

    # Ready, so return result...
    logger.debug(
        f"Loaded {len(features_gdf)} to calculate on in {datetime.now()-start_time}"
    )
    assert isinstance(features_gdf, gpd.GeoDataFrame)
    return features_gdf


def calc_stats_image_gdf(
    features,
    id_column: str,
    image_path: Path,
    bands: Union[List[str], str],
    output_base_path: Path,
    log_dir: Path,
    log_level: Union[str, int],
    future_start_time=None,
    cloud_filter_band: Optional[str] = None,
) -> bool:
    """
    Calculate stats for an image.

    Returns True if succesfully completed.
    Remark: easiest it returns something, when used in a parallel way:
        concurrent.futures likes it better if something is returned
    """

    # TODO: the different bands should be possible to process in parallel as well... so
    # this function should process only one band!
    # When running in parallel processes, the logging needs to be write to seperate
    # files + no console logging
    if len(logging.getLogger().handlers) == 0:
        log_path = (
            log_dir
            / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calc_stats_image_{os.getpid()}.log"
        )
        filehandler = logging.FileHandler(filename=log_path)
        filehandler.setLevel(log_level)
        log_format = "%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s"
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[filehandler],
        )

    global logger
    logger = logging.getLogger("calc_stats_image")

    # Log the time between scheduling the future and acually run...
    if future_start_time is not None:
        logger.info(
            f"Start, {(datetime.now()-future_start_time).total_seconds()} after future "
            "was scheduled"
        )

    # If the features_gdf is a string, use it as file path to unpickle geodataframe...
    if isinstance(features, str) or isinstance(features, Path):
        features_gdf_pkl_path = features
        logger.info(f"Read pickle: {features_gdf_pkl_path}")
        features = pd.read_pickle(features_gdf_pkl_path)
        logger.info(f"Read pickle with {len(features.index)} features ready")

    if isinstance(bands, str):
        bands = [bands]

    # Init some variables
    features_total_bounds = features.total_bounds
    output_base_path_noext, output_ext = os.path.splitext(output_base_path)

    # If the image has a quality band, check that one first so parcels with
    # bad pixels can be removed as the data can't be trusted anyway
    if cloud_filter_band is not None:
        if cloud_filter_band != "SCL-20m":
            raise ValueError(f"cloud_filter_band not supported: {cloud_filter_band}")
        # Specific for Scene Classification (SCL) band, interprete already
        # Folowing values are considered "bad":
        #   -> 0 (no_data), 1 (saturated or defective), 3 (cloud shadows),
        #      8: (cloud, medium proba), 9 (cloud, high proba), 11 (snow)
        # These are considered "OK":
        #   -> 2 (dark area pixels), 4 (vegetation), 5 (not_vegetated),
        #      6 (water), 7 (unclassified), 10 (thin cirrus)
        # List of classes:
        # https://usermanual.readthedocs.io/en/latest/pages/ProductGuide.html#quality-indicator-bands

        # Get the image data and calculate
        logger.info(
            f"Calculate categorical counts for band {cloud_filter_band} on "
            f"{len(features.index)} features"
        )
        category_map = {
            0.0: "nodata",
            1.0: "saturated",
            2.0: "dark",
            3.0: "cloud_shadow",
            4.0: "vegetation",
            5.0: "not_vegetated",
            6.0: "water",
            7.0: "unclassified",
            8.0: "cloud_mediumproba",
            9.0: "cloud_highproba",
            10: "cloud_thincirrus",
            11.0: "snow",
        }
        # Define which columns contain good pixels and which don't
        bad_pixels_cols = [
            "nodata",
            "saturated",
            "dark",
            "snow",
            "cloud_mediumproba",
            "cloud_highproba",
        ]
        image_data = raster_helper.get_image_data(
            image_path,
            bounds=features_total_bounds,
            bands=[cloud_filter_band],
            pixel_buffer=1,
        )
        # Before zonal_stats, do reset_index so index is "clean", otherwise the
        # concat/insert/... later on gives wrong results
        features.reset_index(drop=True, inplace=True)
        features_stats = zonal_stats(
            features,
            image_data[cloud_filter_band]["data"],
            affine=image_data[cloud_filter_band]["transform"],
            prefix="",
            nodata=0,
            categorical=True,
            category_map=category_map,
        )
        features_stats_df = pd.DataFrame(features_stats)
        features_stats_df.fillna(value=0, inplace=True)

        # Make sure the dataframe contains columns for all possible values
        for i, category_key in enumerate(category_map):
            category_column = category_map[category_key]
            if category_column in features_stats_df.columns:
                # Cast to int, otherwise is float
                features_stats_df[category_column] = features_stats_df[
                    category_column
                ].astype("int32")
            else:
                features_stats_df.insert(loc=i, column=category_column, value=0)

        # Add bad pixels column
        nb_bad_pixels_column = "nb_bad_pixels"
        features_stats_df[nb_bad_pixels_column] = features_stats_df[
            bad_pixels_cols
        ].sum(axis=1)

        # Add index and write to file
        features_stats_df.insert(loc=0, column=id_column, value=features[id_column])
        output_band_path = output_base_path.with_stem(
            f"{output_base_path.stem}_{cloud_filter_band}"
        )

        logger.info(
            f"Write data for {len(features_stats_df.index)} parcels found to "
            f"{output_band_path}"
        )
        pdh.to_file(features_stats_df, output_band_path, index=False, append=True)

        # Use the nb_bad_pixels column to filter only parcels without bad pixels
        features.insert(
            loc=0,
            column=nb_bad_pixels_column,
            value=features_stats_df[nb_bad_pixels_column],
        )
        features = features.loc[features[nb_bad_pixels_column] == 0]
        features.drop(columns=[nb_bad_pixels_column], inplace=True)

        # Check if there are still features to be calculated
        if len(features.index) == 0:
            logger.info(
                "After checking quality band, no more features to be calculated: stop"
            )
            return True

    # Loop over image bands
    features_total_bounds = features.total_bounds
    for band in bands:
        # Check if output file exists already...
        output_band_path = output_base_path.with_stem(f"{output_base_path.stem}_{band}")

        # Get the image data and calculate statistics
        logger.info(f"Read band {band} for bounds {features_total_bounds}")
        image_data = raster_helper.get_image_data(
            image_path, bounds=features_total_bounds, bands=[band], pixel_buffer=1
        )

        # Upsample the image to double resolution, so we can use pieces of pixels for
        # small parcels without introducing big errors due to mixels
        upsample_factor = 2
        image_data_upsampled = (
            image_data[band]["data"]
            .repeat(upsample_factor, axis=0)
            .repeat(upsample_factor, axis=1)
        )
        affine_upsampled = image_data[band]["transform"] * Affine.scale(
            1 / upsample_factor
        )

        logger.info(
            f"Calculate zonal statistics for band {band} on "
            f"{len(features.index)} features"
        )
        # Before zonal_stats, do reset_index so index is "clean", otherwise the
        # concat/insert/... later on gives wrong results
        features.reset_index(drop=True, inplace=True)
        features_stats = zonal_stats(
            features,
            image_data_upsampled,
            affine=affine_upsampled,
            prefix="",
            nodata=image_data[band]["nodata"],
            all_touched=False,
            stats=["count", "mean", "median", "std", "min", "max"],
        )
        features_stats_df = pd.DataFrame(features_stats)
        features_stats_df["count"] = features_stats_df["count"].divide(
            upsample_factor * 2
        )

        # Add original id column to statistics dataframe
        features_stats_df.insert(loc=0, column=id_column, value=features[id_column])

        # Remove rows with empty data
        features_stats_df.dropna(inplace=True)
        if len(features_stats_df.index) == 0:
            logger.info(
                f"No data found for band {band}, so no use to process other bands"
            )
            return True
        features_stats_df.set_index(id_column, inplace=True)
        logger.info(
            f"Write data for {len(features_stats_df.index)} parcels found to "
            f"{output_band_path}"
        )
        pdh.to_file(features_stats_df, output_band_path, append=True)

    return True
