# -*- coding: utf-8 -*-
"""
This module contains general functions that apply to timeseries data...
"""

import logging
import os
from pathlib import Path
from typing import List

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh
import cropclassification.preprocess._timeseries_util as ts_util

# First define/init some general variables/constants
# -------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

# The real work
# -------------------------------------------------------------


def calc_timeseries_data(
    input_parcel_path: Path,
    start_date_str: str,
    end_date_str: str,
    sensordata_to_get: List[str],
    dest_data_dir: Path,
):
    """
    Calculate timeseries data for the input parcels

    Args:
        input_parcel_path (str): [description]
        start_date_str (str): [description]
        end_date_str (str): [description]
        sensordata_to_get (List[str]): an array with data you want to be calculated:
                check out the constants starting with DATA_TO_GET... for the options.
        dest_data_dir (str): [description]
    """
    # Check some variables...
    if sensordata_to_get is None:
        raise Exception("sensordata_to_get cannot be None")
    if not dest_data_dir.exists():
        dest_data_dir.mkdir(parents=True, exist_ok=True)

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_util.get_monday(
        start_date_str
    )  # output: vb 2018_2_1 - maandag van week 2 van 2018
    end_date = ts_util.get_monday(end_date_str)

    logger.info(
        f"Start date {start_date_str} converted to monday before: {start_date}, end "
        f"date {end_date_str} as well: {end_date}"
    )
    openeo_supported = ["s2-rgb", "s2-ndvi", "s2-agri", "s1-asc", "s1-desc", "s1-coh"]
    sensordata_to_get_onda = [
        sensor for sensor in sensordata_to_get if sensor not in openeo_supported
    ]
    sensordata_to_get_openeo = [
        sensor for sensor in sensordata_to_get if sensor in openeo_supported
    ]

    if len(sensordata_to_get_onda) > 0:
        # Start!
        # TODO: start calculation of per image data on DIAS
        # import cropclassification.preprocess.timeseries_calc_dias_onda_per_image as
        # ts_calc

        # Now all image data is available per image, calculate periodic data
        ts_util.calculate_periodic_data(
            input_parcel_path=input_parcel_path,
            input_base_dir=conf.dirs.getpath("timeseries_per_image_dir"),
            start_date=start_date,
            end_date=end_date,
            sensordata_to_get=sensordata_to_get_onda,
            dest_data_dir=dest_data_dir,
        )
    if len(sensordata_to_get_openeo) > 0:
        # Pepare periodic images + calculate base timeseries on them
        import cropclassification.preprocess._timeseries_calc_openeo as ts_calc_openeo

        ts_calc_openeo.calc_timeseries_data(
            input_parcel_path=input_parcel_path,
            start_date=start_date,
            end_date=end_date,
            sensordata_to_get=sensordata_to_get_openeo,
            dest_image_data_dir=conf.dirs.getpath("images_periodic_dir"),
            dest_data_dir=conf.dirs.getpath("timeseries_per_image_dir"),
        )

        # Now all image data is available per image, calculate periodic data
        ts_util.calculate_periodic_data(
            input_parcel_path=input_parcel_path,
            input_base_dir=conf.dirs.getpath("timeseries_per_image_dir"),
            start_date=start_date,
            end_date=end_date,
            sensordata_to_get=sensordata_to_get_openeo,
            dest_data_dir=dest_data_dir,
        )


def collect_and_prepare_timeseries_data(
    input_parcel_path: Path,
    timeseries_dir: Path,
    base_filename: str,
    output_path: Path,
    start_date_str: str,
    end_date_str: str,
    sensordata_to_use: List[str],
    parceldata_aggregations_to_use: List[str],
    force: bool = False,
):
    """
    Collect all timeseries data to use for the classification and prepare it by applying
    scaling,... as needed.
    """

    # If force == False Check and the output file exists already, stop.
    if force is False and output_path.exists() is True:
        logger.warning(
            f"Output file already exists and force == False, so stop: {output_path}"
        )
        return

    # Init the result with the id's of the parcels we want to treat
    result_df = pdh.read_file(input_parcel_path, columns=[conf.columns["id"]])
    if result_df.index.name != conf.columns["id"]:
        result_df.set_index(conf.columns["id"], inplace=True)
    nb_input_parcels = len(result_df.index)
    logger.info(f"Parceldata aggregations to use: {parceldata_aggregations_to_use}")
    logger.setLevel(logging.DEBUG)

    # Loop over all input timeseries data to find the data we really need
    data_ext = conf.general["data_ext"]
    path_start = timeseries_dir / f"{base_filename}_{start_date_str}{data_ext}"
    path_end = timeseries_dir / f"{base_filename}_{end_date_str}{data_ext}"
    logger.debug(f"path_start_date: {path_start}")
    logger.debug(f"path_end_date: {path_end}")

    ts_data_files = timeseries_dir.glob(f"{base_filename}_*{data_ext}")
    for curr_path in sorted(ts_data_files):

        # Only process data that is of the right sensor types
        sensor_type = curr_path.stem.split("_")[-1]
        if sensor_type not in sensordata_to_use:
            logger.debug(
                f"SKIP: file not needed (only {sensordata_to_use}): {curr_path}"
            )
            continue
        # The only data we want to process is the data in the range of dates
        if (str(curr_path) < str(path_start)) or (str(curr_path) >= str(path_end)):
            logger.debug(f"SKIP: File is not in date range asked: {curr_path}")
            continue
        # An empty file signifies that there wasn't any valable data for that
        # period/sensor/...
        if os.path.getsize(curr_path) == 0:
            logger.info(f"SKIP: file is empty: {curr_path}")
            continue

        # Read data, and check if there is enough data in it
        data_read_df = pdh.read_file(curr_path)
        nb_data_read = len(data_read_df.index)
        data_available_pct = nb_data_read * 100 / nb_input_parcels
        min_parcels_with_data_pct = conf.timeseries.getfloat(
            "min_parcels_with_data_pct"
        )
        if data_available_pct < min_parcels_with_data_pct:
            logger.info(
                f"SKIP: only data for {data_available_pct:.2f}% of parcels, should be "
                f"> {min_parcels_with_data_pct}%: {curr_path}"
            )
            continue

        # Start processing the file
        logger.info(f"Process file: {curr_path}")
        if data_read_df.index.name != conf.columns["id"]:
            data_read_df.set_index(conf.columns["id"], inplace=True)

        # Loop over columns to check if there are columns that need to be dropped.
        for column in data_read_df.columns:

            # If it is the id column, continue
            if column == conf.columns["id"]:
                continue

            # Check if the column is "asked"
            column_ok = False
            for parceldata_aggregation in parceldata_aggregations_to_use:
                if column.endswith("_" + parceldata_aggregation):
                    column_ok = True
            if column_ok is False:
                # Drop column if it doesn't end with something in
                # parcel_data_aggregations_to_use
                logger.debug(
                    f"Drop column as it's column aggregation isn't to be used: {column}"
                )
                data_read_df.drop(column, axis=1, inplace=True)
                continue

            # Check if the column contains data for enough parcels
            valid_input_data_pct = (
                1 - (data_read_df[column].isnull().sum() / nb_input_parcels)
            ) * 100
            if valid_input_data_pct < min_parcels_with_data_pct:
                # If the number of nan values for the column > x %, drop column
                logger.warn(
                    f"Drop column as it contains only {valid_input_data_pct:.2f}% real "
                    f"data compared to input (= not nan) which is "
                    f"< {min_parcels_with_data_pct}%!: {column}"
                )
                data_read_df.drop(column, axis=1, inplace=True)

        # If S2, rescale data
        if sensor_type.startswith("S2"):
            for column in data_read_df.columns:
                logger.info(
                    f"Column with S2 data: scale it by dividing by 10.000: {column}"
                )
                data_read_df[column] = data_read_df[column] / 10000

        # If S1 coherence, rescale data
        if sensor_type == "S1Coh":
            for column in data_read_df.columns:
                logger.info(
                    f"Column with S1 Coherence: scale it by dividing by 300: {column}"
                )
                data_read_df[column] = data_read_df[column] / 300

        # Join the data to the result...
        result_df = result_df.join(data_read_df, how="left")

    # Remove rows with many null values from result
    max_number_null = int(0.6 * len(result_df.columns))
    parcel_many_null_df = result_df[result_df.isnull().sum(axis=1) > max_number_null]
    if len(parcel_many_null_df.index) > 0:
        # Write the rows with empty data to a file
        parcel_many_null_path = Path(f"{str(output_path)}_rows_many_null.sqlite")
        logger.warn(
            f"Write {len(parcel_many_null_df.index)} rows with > {max_number_null} of "
            f"{len(result_df.columns)} columns==null to {parcel_many_null_path}"
        )
        pdh.to_file(parcel_many_null_df, parcel_many_null_path)

        # Now remove them from result
        result_df = result_df[result_df.isnull().sum(axis=1) <= max_number_null]

    # For rows with some null values, set them to 0
    # TODO: first rough test of using interpolation doesn't give a difference, maybe
    # better if smarter interpolation is used (= only between the different types of
    # data: S1_GRD_VV, S1_GRD_VH, S1_COH_VV, S1_COH_VH, ASC?, DESC?, S2
    # result_df.interpolate(inplace=True)
    result_df.fillna(0, inplace=True)

    # Write output file...
    logger.info(f"Write output to file, start: {output_path}")
    pdh.to_file(result_df, output_path)
    logger.info(f"Write output to file, ready (with shape: {result_df.shape})")
