# -*- coding: utf-8 -*-
"""
Calaculate the timeseries data per image on DIAS.
"""

import datetime
import glob
import os
from pathlib import Path
import shutil
from typing import List
import dateutil.parser

# Import geofilops here already, if tensorflow is loaded first leads to dll load errors
import geofileops as gfo  # noqa: F401

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.helpers import raster_helper
from cropclassification.preprocess import _timeseries_calc_per_image as calc_ts
from cropclassification.preprocess import _timeseries_util as ts_util


def calc_timeseries_task(config_paths: List[Path], default_basedir: Path):
    """
    Runs a calculation of timeseries with the settings in the config_paths.

    Args:
        config_paths (List[Path]): the config files to load
        default_basedir (Path): the dir to resolve relative paths in the config
            file to.

    Raises:
        Exception: [description]
        Exception: [description]
    """
    # Read the configuration files
    conf.read_config(config_paths, default_basedir=default_basedir)

    test = conf.calc_timeseries_params.getboolean("test")

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_util.get_monday(
        conf.marker["start_date_str"]
    )  # output: vb 2018_2_1 - maandag van week 2 van 2018
    end_date = ts_util.get_monday(conf.marker["end_date_str"])

    calc_year_start = start_date.year
    calc_year_stop = end_date.year
    calc_month_start = start_date.month
    calc_month_stop = end_date.month

    # Init logging
    base_log_dir = conf.dirs.getpath('log_dir')
    log_level = conf.general.get("log_level")
    if test:
        base_log_dir = base_log_dir.parent / f"{base_log_dir.name}_test"
    log_dir = base_log_dir / f"calc_dias_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Clean test log dir if it exist
    if test and base_log_dir.exists():
        shutil.rmtree(base_log_dir)

    global logger
    logger = log_helper.main_log_init(log_dir, __name__, log_level)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    if test:
        logger.info(
            f"As we are testing, clean all test logging and use new log_dir: {log_dir}"
        )

    # Write the consolidated config as ini file again to the run dir
    config_used_path = log_dir / "config_used.ini"
    with open(config_used_path, "w") as config_used_file:
        conf.config.write(config_used_file)

    # TODO: this shouldn't be hardcoded!
    input_parcel_filename = conf.calc_timeseries_params.getpath("input_parcel_filename")
    input_features_filename = Path(
        f"{input_parcel_filename.stem}_bufm5{input_parcel_filename.suffix}"
    )
    input_preprocessed_dir = conf.dirs.getpath("input_preprocessed_dir")
    input_features_path = input_preprocessed_dir / input_features_filename

    # Init output dir
    timeseries_per_image_dir = conf.dirs.getpath("timeseries_per_image_dir")
    if not test:
        output_basedir = timeseries_per_image_dir
    else:
        output_basedir = Path(f"{str(timeseries_per_image_dir)}_test")
        logger.info(f"As we are testing, use test output basedir: {output_basedir}")
    output_dir = output_basedir / input_features_filename.stem
    if test:
        if output_dir.exists():
            logger.info(f"As we are only testing, clean the output dir: {output_dir}")
            # By adding a / at the end, only the contents are recursively deleted
            shutil.rmtree(str(output_dir) + os.sep)

    # Temp dir + clean contents from it.
    temp_dir = conf.dirs.getpath("temp_dir") / "calc_dias"
    logger.info(f"Clean the temp dir {temp_dir}")
    if temp_dir.exists():
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(str(temp_dir) + os.sep)

    # Process S1 GRD images
    input_image_paths = []
    for year in range(calc_year_start, calc_year_stop + 1):
        # TODO: works, but doesn't seem to be the most elegant code...
        if year < calc_year_stop:
            month_stop = 12
        else:
            month_stop = calc_month_stop
        if year > calc_year_start:
            month_start = 1
        else:
            month_start = calc_month_start
        for month in range(calc_month_start, calc_month_stop + 1):
            input_image_searchstr = (
                f"/mnt/NAS*/CARD/FLANDERS/S1*/L1TC/{year}/{month:02d}/*/*.CARD"
            )
            input_image_paths.extend(glob.glob(input_image_searchstr))
    logger.info(f"Found {len(input_image_paths)} S1 GRD images in the time period")

    # Verify which S1 images we actually want to process...
    tmp_input_image_paths = []
    esaSwitchedProcessingMethod = dateutil.parser.isoparse("2021-02-23").timestamp()
    for input_image_path in input_image_paths:
        input_image_path = Path(input_image_path)
        # Get more detailed info about the image
        try:
            image_info = raster_helper.get_image_info(input_image_path)
        except Exception:
            # If not possible to get info for image, log and skip it
            logger.exception(
                f"SKIP image: error getting info for {input_image_path}"
            )
            continue

        # Fast-24h <- 2021-02-23 -> NRT-3H
        productTimelinessCategory = (
            "Fast-24h"
            if dateutil.parser.isoparse(
                image_info["acquisition_date"]
            ).timestamp()
            < esaSwitchedProcessingMethod
            else "NRT-3h"
        )

        # If sentinel1 and wrong productTimelinessCategory, skip: we only
        # want 1 type to evade images used twice
        if (
            image_info["satellite"].startswith("S1")
            and image_info["productTimelinessCategory"]
            != productTimelinessCategory
        ):
            logger.info(
                f"SKIP image, productTimelinessCategory should be "
                f"'{productTimelinessCategory}', but is: "
                f"{image_info['productTimelinessCategory']} for "
                f"{input_image_path}"
            )
            continue

        tmp_input_image_paths.append(input_image_path)

    input_image_paths = tmp_input_image_paths
    logger.info(f"Found {len(input_image_paths)} S1 GRD images to process")

    if test:
        # Take only the x first images found while testing
        input_image_paths = input_image_paths[:10]
        logger.info(
            f"As we are only testing, process only {len(input_image_paths)} test images"
        )

    try:
        calc_ts.calc_stats_per_image(
            features_path=input_features_path,
            id_column=conf.columns["id"],
            image_paths=input_image_paths,
            bands=["VV", "VH"],
            output_dir=output_dir,
            temp_dir=temp_dir,
            log_dir=log_dir,
            log_level=log_level,
        )
    except Exception as ex:
        logger.exception(ex)

    # Process S2 images
    input_image_paths = []
    for year in range(calc_year_start, calc_year_stop + 1):
        # TODO: works, but doesn't seem to be the most elegant code...
        if year < calc_year_stop:
            month_stop = 12
        else:
            month_stop = calc_month_stop
        if year > calc_year_start:
            month_start = 1
        else:
            month_start = calc_month_start
        for month in range(month_start, month_stop + 1):
            input_image_searchstr = (
                f"/mnt/NAS*/CARD/FLANDERS/S2*/L2A/{year}/{month:02d}/*/*.SAFE"
            )
            logger.info(f"Search for {input_image_searchstr}")
            input_image_paths.extend(glob.glob(input_image_searchstr))
    logger.info(f"Found {len(input_image_paths)} S2 images to process")

    if test:
        # Take only the x first images found while testing
        input_image_paths = input_image_paths[:10]
        logger.info(
            f"As we are only testing, process only {len(input_image_paths)} test images"
        )

    # TODO: refactor underlying code so the SCL band is used regardless of it being
    # passed here
    max_cloudcover_pct = conf.timeseries.getfloat("max_cloudcover_pct")
    # Verify which S1 images we actually want to process...
    tmp_input_image_paths = []
    for input_image_path in input_image_paths:
        input_image_path = Path(input_image_path)

        # Get more detailed info about the image
        try:
            image_info = raster_helper.get_image_info(input_image_path)
        except Exception:
            # If not possible to get info for image, log and skip it
            logger.exception(
                f"SKIP image: error getting info for {input_image_path}"
            )
            continue

        # If sentinel2 and cloud coverage too high... skip
        if (
            max_cloudcover_pct >= 0
            and image_info["satellite"].startswith("S2")
            and image_info["Cloud_Coverage_Assessment"] > max_cloudcover_pct
        ):
            logger.info(
                "SKIP image, Cloud_Coverage_Assessment: "
                f"{image_info['Cloud_Coverage_Assessment']:0.2f} > "
                f"{max_cloudcover_pct} for {input_image_path}"
            )
            continue

        tmp_input_image_paths.append(input_image_path)

    try:
        calc_ts.calc_stats_per_image(
            features_path=input_features_path,
            id_column=conf.columns["id"],
            image_paths=input_image_paths,
            bands=conf.timeseries.getlist("s2bands"),
            output_dir=output_dir,
            temp_dir=temp_dir,
            log_dir=log_dir,
            log_level=log_level,
        )
    except Exception as ex:
        logger.exception(ex)

    # Process S1 Coherence images
    input_image_paths = []
    for year in range(calc_year_start, calc_year_stop + 1):
        # TODO: works, but doesn't seem to be the most elegant code...
        if year < calc_year_stop:
            month_stop = 12
        else:
            month_stop = calc_month_stop
        if year > calc_year_start:
            month_start = 1
        else:
            month_start = calc_month_start
        for month in range(calc_month_start, calc_month_stop + 1):
            input_image_searchstr = (
                f"/mnt/NAS*/CARD/FLANDERS/S1*/L1CO/{year}/{month:02d}/*/*.CARD"
            )
            input_image_paths.extend(glob.glob(input_image_searchstr))
    logger.info(f"Found {len(input_image_paths)} S1 Coherence images to process")

    if test:
        # Take only the x first images found while testing
        input_image_paths = input_image_paths[:10]
        logger.info(
            f"As we are only testing, process only {len(input_image_paths)} test images"
        )

    input_image_paths = [
        Path(input_image_path) for input_image_path in input_image_paths
    ]

    try:
        calc_ts.calc_stats_per_image(
            features_path=input_features_path,
            id_column=conf.columns["id"],
            image_paths=input_image_paths,
            bands=["VV", "VH"],
            output_dir=output_dir,
            temp_dir=temp_dir,
            log_dir=log_dir,
            log_level=log_level,
        )
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    raise Exception("Not implemented")
