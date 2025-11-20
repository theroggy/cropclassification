"""This module contains general functions that apply to timeseries data..."""

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import geofileops as gfo
import numpy as np
import pyproj

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh
import cropclassification.preprocess._timeseries_calc_openeo as ts_calc_openeo
import cropclassification.preprocess._timeseries_helper as ts_helper

# Get a logger...
logger = logging.getLogger(__name__)


def calc_timeseries_data(
    input_parcel_path: Path,
    roi_bounds: tuple[float, float, float, float],
    roi_crs: pyproj.CRS | None,
    start_date: datetime,
    end_date: datetime,
    images_to_use: dict[str, conf.ImageConfig],
    timeseries_periodic_dir: Path,
    force: bool = False,
):
    """Calculate timeseries data for the input parcels.

    Args:
        input_parcel_path (str): path to the parcel data.
        roi_bounds (tuple[float, float, float, float]): bounds of the region to
            calculate timeseries data for.
        roi_crs (Optional[pyproj.CRS]): crs of the bounds of the region.
        start_date (datetime): the start date for the timeseries to generate, inclusive.
        end_date (datetime): the end date for the timeseries to generate, exclusive.
        images_to_use (List[str]): an array with data you want to be calculated:
            check out the constants starting with DATA_TO_GET... for the options.
        timeseries_periodic_dir (Path): Directory the timeseries will be written to.
        force (bool = False): whether to force recalculation of existing data.
    """
    # Check some variables...
    if images_to_use is None:
        raise Exception("sensordata_to_get cannot be None")
    if not timeseries_periodic_dir.exists():
        timeseries_periodic_dir.mkdir(parents=True, exist_ok=True)

    sensordata_to_get_openeo = [
        sensor for sensor in images_to_use if sensor in conf.image_profiles
    ]

    if len(sensordata_to_get_openeo) > 0:
        # Prepare periodic images + calculate base timeseries on them
        ts_calc_openeo.calculate_periodic_timeseries(
            input_parcel_path=input_parcel_path,
            roi_bounds=roi_bounds,
            roi_crs=roi_crs,
            start_date=start_date,
            end_date=end_date,
            imageprofiles_to_get=sensordata_to_get_openeo,
            imageprofiles=conf.image_profiles,
            images_periodic_dir=conf.paths.getpath("images_periodic_dir"),
            timeseries_periodic_dir=timeseries_periodic_dir,
            nb_parallel=conf.general.getint("nb_parallel", -1),
            on_missing_image=conf.images.get("on_missing_image", "calculate_raise"),
            force=force,
        )


def collect_and_prepare_timeseries_data(
    input_parcel_path: Path,
    timeseries_dir: Path,
    output_path: Path,
    start_date: datetime,
    end_date: datetime,
    images_to_use: dict[str, conf.ImageConfig],
    parceldata_aggregations_to_use: list[str],
    max_fraction_null: float = 0.6,
    force: bool = False,
):
    """Collect and preprocess timeseries data needed for the classification.

    Args:
        input_parcel_path (Path): the path to the parcel file to collect timeseries data
            for.
        timeseries_dir (Path): the directory where the timeseries cache is saved.
        output_path (Path): path to write the final output to.
        start_date (datetime): start date of timeseries data to include (inclusive).
        end_date (datetime): end date of timeseries data to include (exclusive).
        images_to_use (dict[str, conf.ImageConfig]): the image profiles and bands to
            extract the timeseries data for.
        parceldata_aggregations_to_use (list[str]): aggregations on parcel level to use
            in the timeseries data.
        max_fraction_null (float, optional): the maximum fraction of columns that can
            have null as value for a row. If more that that is null, the entire row is
            filtered away. If 1, no row filtering is applied. Defaults to 0.6, so if
            more than 60% of row values is None, the row is dropped.
        force (bool, optional): True to overwrite existing outputs. Defaults to False.
    """
    # If force == False Check and the output file exists already, stop.
    if not force and output_path.exists():
        logger.warning(f"Output file already exists, so return: {output_path}")
        return

    # Init the result with the id's of the parcels we want to treat
    result_df = pdh.read_file(input_parcel_path, columns=[conf.columns["id"]])
    if result_df.index.name != conf.columns["id"]:
        result_df.set_index(conf.columns["id"], inplace=True)
    nb_input_parcels = len(result_df.index)
    logger.info(f"Parceldata aggregations to use: {parceldata_aggregations_to_use}")
    logger.setLevel(logging.DEBUG)

    # Loop over all input timeseries data to find the data we really need
    glob_pattern = f"*{conf.general['data_ext']}"
    ts_data_paths = list(timeseries_dir.glob(glob_pattern))
    if len(ts_data_paths) == 0:
        raise ValueError(f"No timeseries data found for pattern {glob_pattern}")

    min_parcels_with_data_pct = conf.timeseries.getfloat("min_parcels_with_data_pct")
    for curr_path in sorted(ts_data_paths):
        # Skip the pixcount file
        if curr_path.stem.endswith("_pixcount"):
            continue

        # Only process data that is of the right sensor types
        fileinfo = ts_helper.get_fileinfo_timeseries_periods(curr_path)
        image_profile = fileinfo["image_profile"].lower()
        if image_profile not in images_to_use:
            logger.debug(f"SKIP: file not needed (only {images_to_use}): {curr_path}")
            continue
        # The only data we want to process is the data in the range of dates
        if fileinfo["start_date"] < start_date or fileinfo["end_date"] >= end_date:
            logger.debug(f"SKIP: file doesn't match the period asked: {curr_path}")
            continue
        band = fileinfo["band"]
        if band not in images_to_use[image_profile].bands:
            logger.debug(f"SKIP: file doesn't match the bands asked: {curr_path}")
            continue
        time_reducer_asked = images_to_use[image_profile].imageprofile.time_reducer
        if time_reducer_asked is not None:
            time_reducer = fileinfo.get("time_reducer")
            if time_reducer is None:
                logger.warning(
                    f"SKIP: time_reducer {time_reducer_asked} "
                    f"asked, but not known for file: {curr_path}"
                )
                continue
            elif time_reducer != time_reducer_asked:
                logger.debug(
                    f"SKIP: file doesn't match the time reducer asked: {curr_path}"
                )
                continue

        # An empty file signifies that there wasn't any valable data for that
        # period/sensor/...
        if os.path.getsize(curr_path) == 0:
            logger.info(f"SKIP: file is empty: {curr_path}")
            continue

        # Determine the columns to be read from the file and which to rename.
        info = gfo.get_layerinfo(curr_path, layer="info", raise_on_nogeom=False)
        columns = []
        columns_to_rename = {}
        for column in info.columns:
            # The id column should be read
            if column == conf.columns["id"]:
                columns.append(column)
                continue

            # Check if the column is "asked"
            column_ok = False
            for parceldata_aggregation in parceldata_aggregations_to_use:
                if column.endswith("_" + parceldata_aggregation):
                    column_ok = True
                elif column == parceldata_aggregation:
                    curr_start_date_str = fileinfo["start_date"].strftime("%Y%m%d")
                    columns_to_rename[column] = (
                        f"{image_profile}_{curr_start_date_str}_{band}_{column}"
                    )
                    column_ok = True
            if not column_ok:
                # Drop column if it doesn't end with something in
                # parcel_data_aggregations_to_use
                logger.debug(
                    "Drop column as it's column aggregation isn't to be used: "
                    f"{curr_path.stem}.{column}"
                )
                continue

            columns.append(column)

        # Read data, and check if there is enough data in it
        data_read_df = pdh.read_file(curr_path, columns=columns)
        if data_read_df.index.name != conf.columns["id"]:
            data_read_df.set_index(conf.columns["id"], inplace=True)

        # Only retain data read that is in the result_df
        data_read_df = data_read_df[data_read_df.index.isin(result_df.index)]

        if min_parcels_with_data_pct > 0:
            nb_data_read = len(data_read_df.index)
            data_available_pct = nb_data_read * 100 / nb_input_parcels
            if data_available_pct < min_parcels_with_data_pct:
                logger.info(
                    f"SKIP: only data for {data_available_pct:.2f}% of parcels, "
                    f"should be > {min_parcels_with_data_pct}%: {curr_path}"
                )
                continue

        # Start processing the file
        logger.info(f"Process file: {curr_path}")
        for column in data_read_df.columns:
            # If it is the id column, continue
            if column == conf.columns["id"]:
                continue

            # Check if the column contains data for enough parcels
            nb_rows_null = len(data_read_df[data_read_df[column].isnull()])
            valid_input_data_pct = (1 - (nb_rows_null / nb_input_parcels)) * 100
            if valid_input_data_pct < min_parcels_with_data_pct:
                # If the number of nan values for the column > x %, drop column
                logger.warning(
                    f"Drop column as it contains only {valid_input_data_pct:.2f}% real "
                    f"data compared to input (= not nan) which is "
                    f"< {min_parcels_with_data_pct}%!: {curr_path.stem}.{column}"
                )
                data_read_df.drop(column, axis=1, inplace=True)

        # If there are columns that need renaming, do so
        if len(columns_to_rename) > 0:
            data_read_df = data_read_df.rename(columns=columns_to_rename)

        # If S2, rescale data
        if image_profile.startswith("s2-agri"):
            for column in data_read_df.columns:
                logger.info(
                    f"Column with s2 raw band data: /10.000, clip to upper=1: {column}"
                )
                data_read_df[column] = data_read_df[column] / 10000
                data_read_df[column] = data_read_df[column].clip(upper=1)

        # If s1 grd, rescale data
        if image_profile.startswith("s1-grd"):
            for column in data_read_df.columns:
                # Just clipping gives the best results (tested with random forest)
                normalization_steps = ["clip"]
                for step in normalization_steps:
                    if step == "log":
                        logger.info(f"Convert to db, but in range ~0-~1: {column}")
                        data_read_df[column] = (
                            np.log(data_read_df[column]) * 0.21714724095 + 1
                        )
                    if step == "clip":
                        logger.info(f"Column with s1-grd: clip to upper=1: {column}")
                        data_read_df[column] = data_read_df[column].clip(upper=1)
                    if step == "normalize":
                        # normalize all values to be between 0 and 1
                        min = np.min(data_read_df[column])
                        max = np.max(data_read_df[column])
                        data_read_df[column] = (data_read_df[column] - min) / (
                            max - min
                        )
                    if step == "abs":
                        data_read_df[column] = np.abs(data_read_df[column])

        # If s1 coherence, rescale data
        if image_profile.startswith(("s1coh", "s1-coh")):
            for column in data_read_df.columns:
                # Just scale gives same results as log (tested with random forest)
                normalization_steps = ["scale"]
                for step in normalization_steps:
                    if step == "log":
                        logger.info(f"Convert to db, but in range ~0-~1: {column}")
                        data_read_df[column] = (
                            np.log(data_read_df[column]) * 0.21714724095 + 1
                        )
                    if step == "clip":
                        logger.info(f"Column with s1-grd: clip to upper=1: {column}")
                        data_read_df[column] = data_read_df[column].clip(upper=1)
                    if step == "normalize":
                        # normalize all values to be between 0 and 1
                        min = np.min(data_read_df[column])
                        max = np.max(data_read_df[column])
                        data_read_df[column] = (data_read_df[column] - min) / (
                            max - min
                        )
                    if step == "abs":
                        data_read_df[column] = np.abs(data_read_df[column])
                    if step == "scale":
                        logger.info(f"Column with s1-coh: divide by 300: {column}")
                        data_read_df[column] = data_read_df[column] / 300

        # Write warning if the data isn't scaled between 0 and 1
        for column in data_read_df.columns:
            max = data_read_df[column].max()
            min = data_read_df[column].min()
            if max > 1 or min < 0:
                warnings.warn(
                    f"{column=} in {curr_path} isn't fully normalized ({min=}, {max=})",
                    stacklevel=1,
                )

        # Join the data to the result...
        result_df = result_df.join(data_read_df, how="left")

    # No timeseries data was found, so stop
    if len(result_df.columns) == 0:
        raise ValueError("data collection resulted in 0 columns")

    # Remove rows with many null values from result
    max_number_null = int(max_fraction_null * len(result_df.columns))
    parcel_many_null_df = result_df[result_df.isnull().sum(axis=1) > max_number_null]
    if len(parcel_many_null_df.index) > 0:
        # Write the rows with empty data to a file
        parcel_many_null_path = Path(f"{output_path!s}_rows_many_null.sqlite")
        logger.warning(
            f"Write {len(parcel_many_null_df.index)} rows with > {max_number_null} of "
            f"{len(result_df.columns)} columns==null to {parcel_many_null_path}"
        )
        pdh.to_file(parcel_many_null_df, parcel_many_null_path)

        # Now remove them from result
        result_df = result_df[result_df.isnull().sum(axis=1) <= max_number_null]

    # Check if there are values not in the range -1 till +1
    gt1_df = result_df[result_df > 1].dropna()
    ltm1_df = result_df[result_df < -1].dropna()
    if gt1_df.size > 0:
        logger.warning(f"result_df containes values > 1: {gt1_df}")
    if ltm1_df.size > 0:
        logger.warning(f"result_df containes values < -1: {ltm1_df}")

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
