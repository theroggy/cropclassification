"""Calculate timeseries data per image."""

import logging
import math
import multiprocessing
import os
import shutil
import signal  # To catch CTRL-C explicitly and kill children
import sys
import time
from concurrent import futures
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psutil  # To catch CTRL-C explicitly and kill children
import rasterstats
from affine import Affine
from osgeo import gdal

from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util import io_util

from . import _general_helper, _raster_helper, _vector_helper
from . import _processing_util as processing_util

# Suppress gdal warnings/errors
gdal.PushErrorHandler("CPLQuietErrorHandler")

# General init
logger = logging.getLogger(__name__)


def zonal_stats(
    vector_path: Path,
    id_column: str,
    rasters_bands: list[tuple[Path, list[str]]],
    output_dir: Path,
    stats: list[str],
    cloud_filter_band: str | None = None,
    calc_bands_parallel: bool = True,
    nb_parallel: int = -1,
    force: bool = False,
):
    """Calculate zonal statistics."""
    # TODO: probably need to apply some object oriented approach here for "image",
    # because there are to many properties,... to be clean/clear this way.
    # TODO: maybe passing the executor pool to a calc_stats_for_image function can have
    # both the advantage of not creating loads of processes + keep the cleanup logic
    # after calculation together with the processing logic

    # Some checks on the input parameters
    nb_todo = len(rasters_bands)
    if nb_todo == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = io_util.create_tempdir("zonal_stats_rs")
    log_dir = tmp_dir
    log_level = logger.level

    # Create process pool for parallelisation...
    if nb_parallel < 1:
        nb_parallel = multiprocessing.cpu_count()

    # Loop over all images to start the data preparation for each of them in
    # parallel...
    start_time = datetime.now()
    nb_errors_max = 10
    nb_errors = 0

    image_dict: dict[str, Any] = {}
    calc_stats_batch_dict = {}
    nb_done_total = 0
    image_idx = 0

    pool = futures.ProcessPoolExecutor(
        max_workers=nb_parallel, initializer=processing_util.initialize_worker()
    )
    try:
        # Keep looping. At the end of the loop there are checks when to
        # break out of the loop...
        while True:
            # Start preparation for calculation on next image + features combo
            # If not all images aren't prepared for processing yet, and there
            # aren't too many busy yet, prepare the next one
            if (
                image_idx < len(rasters_bands)
                and len(_filter_on_status(image_dict, "IMAGE_PREPARE_CALC_BUSY"))
                < nb_parallel
            ):
                # Not too many busy preparing, so get next image_path to start
                # prepare on
                bands: list[Any]
                image_path, bands = rasters_bands[image_idx]
                image_path_str = str(image_path)
                image_idx += 1

                image_info = _raster_helper.get_image_info(image_path)
                # Create base output filename
                # TODO: hoort hier niet echt thuis
                orbit = None
                if image_info.imagetype.lower() == "s1-grd-sigma0-asc":
                    orbit = "ASCENDING"
                elif image_info.imagetype.lower() == "s1-grd-sigma0-desc":
                    orbit = "DESCENDING"
                output_base_path = _general_helper._format_output_path(
                    vector_path, image_path, output_dir, orbit, band=None
                )
                output_base_busy_path = tmp_dir / f"BUSY_{output_base_path.name}"

                # Check for which bands there is a valid output file already
                if bands is None:
                    bands = image_info.bands
                    assert bands is not None
                bands_todo = {}
                for band in bands:
                    # Prepare the output paths...
                    output_band_path = _general_helper._format_output_path(
                        vector_path, image_path, output_dir, orbit, band
                    )
                    output_band_busy_path = tmp_dir / f"BUSY_{output_band_path.name}"

                    # If a busy output file exists, remove it, otherwise we can get
                    # double data in it...
                    output_band_busy_path.unlink(missing_ok=True)

                    # Check if the output file exists already
                    if output_band_path.exists():
                        if not force:
                            logger.debug(
                                f"Output file for band exists {output_band_path}"
                            )
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
                    nb_todo -= 1
                    continue

                # Start the prepare processing assync
                # TODO: possibly it is cleaner to do this per band...
                logger.info(f"Start calculation for image {output_base_path}")
                future = pool.submit(
                    _prepare_calc,
                    features_path=vector_path,
                    id_column=id_column,
                    image_path=image_path,
                    tmp_dir=tmp_dir,
                    log_dir=log_dir,
                    log_level=log_level,
                    nb_parallel_max=nb_parallel,
                )
                image_dict[image_path_str] = {
                    "features_path": vector_path,
                    "image_path": image_path,
                    "bands": bands,
                    "bands_todo": bands_todo,
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
            for image_path_str in _filter_on_status(
                image_dict, "IMAGE_PREPARE_CALC_BUSY"
            ):
                # If still running, go to next
                image = image_dict[image_path_str]
                if image["prepare_calc_future"].running():
                    continue

                # Extract the result from the preparation
                try:
                    # Get the result from the completed prepare_inputs
                    prepare_calc_result = image["prepare_calc_future"].result()

                    # If nb_features to be treated is 0... create (empty) output
                    # files and continue with next...
                    if prepare_calc_result["nb_features_to_calc_total"] == 0:
                        for band in image["bands_todo"]:
                            # Create output file
                            output_band_path = image["bands_todo"][band]
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
                    bands = image["bands_todo"]
                    # If bands shouldn't be calculated in parallel...
                    if not calc_bands_parallel:
                        bands = [bands]

                    for band in bands:
                        for features_batch in image["feature_batches"]:
                            start_time_batch = datetime.now()
                            future = pool.submit(
                                _zonal_stats_image_gdf,  # type: ignore[arg-type]
                                features=features_batch["path"],
                                id_column=id_column,
                                image_path=image["image_prepared_path"],
                                bands=band,
                                output_base_path=image["output_base_busy_path"],
                                stats=stats,
                                log_dir=log_dir,
                                log_level=log_level,
                                future_start_time=start_time_batch,
                                cloud_filter_band=cloud_filter_band,
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
            for calc_stats_batch_id in _filter_on_status(
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
            for image_path_str in _filter_on_status(image_dict, "IMAGE_CALC_BUSY"):
                # If still batches busy for this image, continue to next image
                batches_busy = False
                for calc_stats_batch_id in _filter_on_status(
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
                for band in image["bands_todo"]:
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
                progress_msg = _general_helper.format_progress_message(
                    nb_todo,
                    nb_done_total,
                    start_time,
                    nb_done_latestbatch,
                    start_time_latestbatch=image["calc_starttime"],
                )
                logger.info(progress_msg)

            # Check if we are completely ready

            # This is the case if:
            #     - no processing is needed (= empty image_dict)
            #     - OR if all (possible) processing is started + everything is done
            #       (= status IMAGE_CALC_DONE)
            if len(image_dict) == 0 or (
                image_idx == len(rasters_bands)
                and len(image_dict)
                == len(_filter_on_status(image_dict, "IMAGE_CALC_DONE"))
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
        f"{(datetime.now() - start_time).total_seconds()} sec"
    )


def _filter_on_status(dict: dict, status_to_check: str) -> list[str]:
    """Check the number of images that are being prepared for processing."""
    keys_with_status = []
    for key in dict:
        if dict[key]["status"] == status_to_check:
            keys_with_status.append(key)
    return keys_with_status


def _prepare_calc(
    features_path: Path,
    id_column: str,
    image_path: Path,
    tmp_dir: Path,
    log_dir: Path,
    log_level: int,
    nb_parallel_max: int = 16,
) -> dict:
    """Prepare the inputs for a calculation.

    Returns True if succesfully completed.
    Remark: easiest it returns something, when used in a parallel way:
    concurrent.futures likes it better if something is returned
    """
    # When running in parallel processes, the logging needs to be write to seperate
    # files + no console logging
    if len(logging.getLogger().handlers) == 0:
        log_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_prepare_calc_{os.getpid()}.log"
        log_path = log_dir / log_name
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
    ret_val: dict[str, Any] = {}

    # Prepare the image
    logger.info(f"Start prepare_image for {image_path} to {tmp_dir}")
    image_prepared_path = _raster_helper.prepare_image(image_path, tmp_dir)
    logger.debug(f"Preparing ready, result: {image_prepared_path}")
    ret_val["image_prepared_path"] = image_prepared_path

    # Get info about the image
    logger.info(f"Start get_image_info for {image_prepared_path}")
    image_info = _raster_helper.get_image_info(image_prepared_path)
    logger.info(f"image_info: {image_info}")

    # Load the features that overlap with the image.
    # Reproject the vector data
    tmp_dir.mkdir(exist_ok=True, parents=True)
    features_proj_path = _vector_helper.reproject_synced(
        path=features_path,
        columns=[id_column, "geometry"],
        target_epsg=image_info.image_epsg,
        dst_dir=tmp_dir,
    )
    # TODO: passing both bbox and poly is double, or not?
    # footprint epsg should be passed as well, or reproject here first?
    footprint_shape = None
    if image_info.footprint is not None:
        logger.info(f"poly: {image_info.footprint['shape']}")
        footprint_shape = image_info.footprint["shape"]

    if image_info.image_epsg == "NONE":
        raise Exception(f"target_epsg == NONE: {image_info}")

    features_gdf = _vector_helper._load_features_file(
        features_path=features_proj_path,
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
    temp_features_dir = tmp_dir / image_path.stem
    ret_val["temp_features_dir"] = temp_features_dir
    if temp_features_dir.exists():
        logger.info(f"Remove dir {temp_features_dir!s}{os.sep}")
        shutil.rmtree(f"{temp_features_dir!s}{os.sep}")
    temp_features_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the batches, pickle them and add the paths to the result...
    ret_val["feature_batches"] = []
    for batch_idx, features_gdf_batch in enumerate(features_gdf_batches):
        batch_info: dict[str, Any] = {}
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


def _zonal_stats_image_gdf(
    features,
    id_column: str,
    image_path: Path,
    bands: list[str] | str,
    output_base_path: Path,
    stats: list[str],
    log_dir: Path,
    log_level: str | int,
    future_start_time=None,
    cloud_filter_band: str | None = None,
) -> bool:
    """Calculate stats for an image.

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
            f"Start, {(datetime.now() - future_start_time).total_seconds()} after "
            "future was scheduled"
        )

    # If the features_gdf is a string, use it as file path to unpickle geodataframe...
    if isinstance(features, (str | Path)):
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
        image_data = _raster_helper.get_image_data(
            image_path,
            bounds=features_total_bounds,
            bands=[cloud_filter_band],
            pixel_buffer=1,
        )
        # Before zonal_stats, do reset_index so index is "clean", otherwise the
        # concat/insert/... later on gives wrong results
        features.reset_index(drop=True, inplace=True)
        features_stats = rasterstats.zonal_stats(
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
        image_data = _raster_helper.get_image_data(
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
        features_stats = rasterstats.zonal_stats(
            features,
            image_data_upsampled,
            affine=affine_upsampled,
            prefix="",
            nodata=image_data[band]["nodata"],
            all_touched=False,
            stats=stats,
        )
        features_stats_df = pd.DataFrame(features_stats)

        # If upsampling was applied, correct the pixel count accordingly
        if upsample_factor > 1 and "count" in features_stats_df:
            count_divide = upsample_factor * 2
            features_stats_df["count"] = features_stats_df["count"].divide(count_divide)

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
