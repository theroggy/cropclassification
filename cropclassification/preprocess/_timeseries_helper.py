"""Calculates periodic timeseries for input parcels."""

import gc
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import geofileops as gfo
import numpy as np
import pandas as pd

# Import local stuff
import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# Get a logger...
logger = logging.getLogger(__name__)

IMAGETYPE_S1_GRD = "S1_GRD"
IMAGETYPE_S1_COHERENCE = "S1_COH"
IMAGETYPE_S2_L2A = "S2_L2A"


def prepare_input(
    input_parcel_path: Path,
    output_imagedata_parcel_input_path: Path,
    output_parcel_nogeo_path: Optional[Path] = None,
    force: bool = False,
) -> bool:
    """Prepare a file so it is ready for timeseries extraction of sentinel images.

    Args:
        input_parcel_path (Path): input file
        output_imagedata_parcel_input_path (Path): prepared output file
        output_parcel_nogeo_path (Path): output file with a copy of the non-geo data
        force: force creation, even if output file(s) exist already
    """
    # Check if parameters are OK and init some extra params
    if not input_parcel_path.exists():
        raise Exception(f"Input file doesn't exist: {input_parcel_path}")

    # Check if the input file has a projection specified
    if gfo.get_crs(input_parcel_path) is None:
        message = (
            "The parcel input file doesn't have a projection/crs specified, so STOP: "
            f"{input_parcel_path}"
        )
        logger.critical(message)
        raise Exception(message)

    # If force == False Check and the output file exists already, stop.
    if (
        force is False
        and output_imagedata_parcel_input_path.exists()
        and (output_parcel_nogeo_path is None or output_parcel_nogeo_path.exists())
    ):
        logger.warning(
            "prepare_input: force is False and output files exist, so stop: "
            f"{output_imagedata_parcel_input_path}, "
            f"{output_parcel_nogeo_path}"
        )
        return False

    logger.info(f"Process input file {input_parcel_path}")

    # Create temp dir to store temporary data for tracebility
    temp_output_dir = output_imagedata_parcel_input_path.parent / "temp"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # Read the parcel data and write nogeo version
    parceldata_gdf = gfo.read_file(input_parcel_path)
    logger.info(f"Parceldata read, shape: {parceldata_gdf.shape}")

    # Check if the id column is present and set as index
    if conf.columns["id"] in parceldata_gdf.columns:
        parceldata_gdf.set_index(conf.columns["id"], inplace=True)
    else:
        message = (
            f"STOP: Column {conf.columns['id']} not found in input parcel file: "
            f"{input_parcel_path}. Make sure the column is present or change the "
            "column name in global_constants.py"
        )
        logger.critical(message)
        raise Exception(message)

    if output_parcel_nogeo_path is not None and (
        force is True or not output_parcel_nogeo_path.exists()
    ):
        logger.info(f"Save non-geo data to {output_parcel_nogeo_path}")
        parceldata_nogeo_df = parceldata_gdf.drop(["geometry"], axis=1)
        pdh.to_file(parceldata_nogeo_df, output_parcel_nogeo_path)

    # Do the necessary conversions and write buffered file

    # If force == False Check and the output file exists already, stop.
    if force is False and output_imagedata_parcel_input_path.exists():
        logger.warning(
            "prepare_input: force is False and output files exist, so stop: "
            f"{output_imagedata_parcel_input_path}"
        )
        return False

    # Apply buffer
    parceldata_buf_gdf = parceldata_gdf.copy()
    # resolution = number of segments per circle
    buffer_size = -conf.timeseries.getfloat("buffer")
    logger.info(f"Apply buffer of {buffer_size} on parcel")
    parceldata_buf_gdf[conf.columns["geom"]] = parceldata_buf_gdf[
        conf.columns["geom"]
    ].buffer(buffer_size, resolution=5)

    # Export buffered geometries that result in empty geometries
    logger.info("Export parcels that are empty after buffer")
    parceldata_buf_empty_df = parceldata_buf_gdf.loc[
        parceldata_buf_gdf[conf.columns["geom"]].is_empty
    ].copy()
    if len(parceldata_buf_empty_df.index) > 0:
        parceldata_buf_empty_df.drop(conf.columns["geom"], axis=1, inplace=True)
        temp_empty_path = (
            temp_output_dir / f"{output_imagedata_parcel_input_path.stem}_empty.sqlite"
        )
        pdh.to_file(parceldata_buf_empty_df, temp_empty_path)

    # Export parcels that don't result in an empty geometry
    parceldata_buf_notempty_gdf = parceldata_buf_gdf.loc[
        ~parceldata_buf_gdf[conf.columns["geom"]].is_empty
    ]
    parceldata_buf_nopoly_gdf = parceldata_buf_notempty_gdf.loc[
        ~parceldata_buf_notempty_gdf[conf.columns["geom"]].geom_type.isin(
            ["Polygon", "MultiPolygon"]
        )
    ]
    if len(parceldata_buf_nopoly_gdf.index) > 0:
        logger.info("Export parcels that are no (multi)polygons after buffer")
        parceldata_buf_nopoly_df = parceldata_buf_nopoly_gdf.drop(
            conf.columns["geom"], axis=1
        )
        temp_nopoly_path = (
            temp_output_dir / f"{output_imagedata_parcel_input_path.stem}_nopoly.sqlite"
        )
        pdh.to_file(parceldata_buf_nopoly_df, temp_nopoly_path)

    # Export parcels that are (multi)polygons after buffering
    parceldata_buf_poly_gdf = parceldata_buf_notempty_gdf.loc[
        parceldata_buf_notempty_gdf[conf.columns["geom"]].geom_type.isin(
            ["Polygon", "MultiPolygon"]
        )
    ]
    for column in parceldata_buf_poly_gdf.columns:
        if column not in [conf.columns["id"], conf.columns["geom"]]:
            parceldata_buf_poly_gdf.drop(column, axis=1, inplace=True)
    logger.info(
        "Export parcels that are (multi)polygons after buffer to"
        f"{output_imagedata_parcel_input_path}"
    )
    parceldata_buf_poly_gdf.to_file(
        output_imagedata_parcel_input_path, engine="pyogrio"
    )
    logger.info(parceldata_buf_poly_gdf)

    return True


def calculate_periodic_timeseries(
    parcel_path: Path,
    timeseries_per_image_dir: Path,
    start_date: datetime,
    end_date: datetime,
    sensordata_to_get: list[str],
    timeseries_periodic_dir: Path,
    force: bool = False,
):
    """This function creates a file that is a weekly aggregation of timeseries files.

    TODO: add possibility to choose which values to extract (mean, min, max,...)?

    Args:
        parcel_path (Path): path to the parcel file.
        timeseries_per_image_dir (Path): directory with timeseries data per satellite
            image to be used for weekly aggregation.
        start_date (datetime): Start date. Needs to be aligned
            already on the periods wanted + data on this date is included.
        end_date (datetime): End date. Needs to be aligned already on
            the periods wanted + data on this date is excluded.
        sensordata_to_get ([]): list of sensor configurations to get data for.
        timeseries_periodic_dir (Path): directory the timeseries will be written to.
        force (bool, optional): True to overwrite existing files. Defaults to False.
    """
    logger.info("calculate_periodic_data")

    # Init
    # TODO: in config?
    input_ext = ".sqlite"
    output_ext = ".sqlite"

    year = start_date.year

    # Prepare output dir
    timeseries_periodic_dir.mkdir(parents=True, exist_ok=True)

    # Create Dataframe with all files with their info
    logger.debug("Create Dataframe with all files and their properties")
    file_info_list = []
    for filename in os.listdir(timeseries_per_image_dir):
        if filename.endswith(input_ext):
            # Get seperate filename parts
            file_info = get_fileinfo_timeseries(timeseries_per_image_dir / filename)
            file_info_list.append(file_info)

    all_inputfiles_df = pd.DataFrame(file_info_list)

    # Loop over the data we need to get
    id_column = conf.columns["id"]
    for sensordata_type in sensordata_to_get:
        logger.debug(
            "Get files we need based on start- & stopdates, sensordata_to_get,..."
        )
        orbits: list[Optional[str]] = [None]
        if sensordata_type == "S1AscDesc":
            # Filter files to the ones we need
            # satellitetype = "S1"
            imagetype = IMAGETYPE_S1_GRD
            bands = ["VV", "VH"]
            orbits = ["ASC", "DESC"]
            needed_inputfiles_df = all_inputfiles_df.loc[
                (all_inputfiles_df.date >= start_date)
                & (all_inputfiles_df.date < end_date)
                & (all_inputfiles_df.imagetype == imagetype)
                & (all_inputfiles_df.band.isin(bands))
                & (all_inputfiles_df.orbit.isin(orbits))
            ]
        elif sensordata_type == "S1Coh":
            # satellitetype = "S1"
            imagetype = IMAGETYPE_S1_COHERENCE
            bands = ["VV", "VH"]
            orbits = ["ASC", "DESC"]
            needed_inputfiles_df = all_inputfiles_df.loc[
                (all_inputfiles_df.date >= start_date)
                & (all_inputfiles_df.date < end_date)
                & (all_inputfiles_df.imagetype == imagetype)
                & (all_inputfiles_df.band.isin(bands))
            ]
        elif sensordata_type == "S2gt95":
            # satellitetype = "S2"
            imagetype = IMAGETYPE_S2_L2A
            bands = ["B02-10m", "B03-10m", "B04-10m", "B08-10m", "B11-20m", "B12-20m"]
            needed_inputfiles_df = all_inputfiles_df.loc[
                (all_inputfiles_df.date >= start_date)
                & (all_inputfiles_df.date < end_date)
                & (all_inputfiles_df.imagetype == imagetype)
                & (all_inputfiles_df.band.isin(bands))
            ]
        elif sensordata_type == "S2-landcover":
            # satellitetype = "S2"
            imagetype = IMAGETYPE_S2_L2A
            bands = ["landcover"]
            needed_inputfiles_df = all_inputfiles_df.loc[
                (all_inputfiles_df.date >= start_date)
                & (all_inputfiles_df.date < end_date)
                & (all_inputfiles_df.imagetype == imagetype)
                & (all_inputfiles_df.band.isin(bands))
            ]
        elif sensordata_type == "S2-ndvi":
            # satellitetype = "S2"
            imagetype = IMAGETYPE_S2_L2A
            bands = ["ndvi"]
            needed_inputfiles_df = all_inputfiles_df.loc[
                (all_inputfiles_df.date >= start_date)
                & (all_inputfiles_df.date < end_date)
                & (all_inputfiles_df.imagetype == imagetype)
                & (all_inputfiles_df.band.isin(bands))
            ]
        else:
            raise ValueError(f"Unsupported sensordata_type: {sensordata_type}")

        # There should also be one pixcount file
        pixcount_filename = f"{parcel_path.stem}_weekly_pixcount{output_ext}"
        pixcount_path = timeseries_periodic_dir / pixcount_filename

        # For each week
        start_week = int(datetime.strftime(start_date, "%W"))
        end_week = int(datetime.strftime(end_date, "%W"))
        for period_index in range(start_week, end_week):
            # Get the date of the first day of period period_index
            # (eg. monday for a week)
            period_date = datetime.strptime(
                str(year) + "_" + str(period_index) + "_1", "%Y_%W_%w"
            )

            # New file name
            period_date_str_long = period_date.strftime("%Y-%m-%d")
            period_data_filename = (
                f"{parcel_path.stem}_weekly_{period_date_str_long}_{sensordata_type}"
                f"{output_ext}"
            )
            period_data_path = timeseries_periodic_dir / period_data_filename

            # Check if output file exists already
            if period_data_path.exists() and pixcount_path.exists():
                if force is False:
                    logger.info(
                        f"SKIP: force is False and file exists: {period_data_path}"
                    )
                    continue
                else:
                    os.remove(period_data_path)

            # Loop over bands and orbits (all combinations of bands and orbits!)
            logger.info(f"Calculate file: {period_data_filename}")
            period_data_df = None
            gc.collect()  # Try to evade memory errors
            for band, orbit in [(band, orbit) for band in bands for orbit in orbits]:
                # Get list of files needed for this period, band
                period_files_df = needed_inputfiles_df.loc[
                    (needed_inputfiles_df.week == period_index)
                    & (needed_inputfiles_df.band == band)
                ]

                # If an orbit to be filtered was specified, filter
                if orbit is not None:
                    period_files_df = period_files_df.loc[
                        (period_files_df.orbit == orbit)
                    ]

                if len(period_files_df) == 0:
                    logger.warning("No input files found!")

                # Loop all period_files
                period_band_data_df = None
                statistic_columns_dict: dict[str, Any] = {
                    "count": [],
                    "max": [],
                    "mean": [],
                    "median": [],
                    "min": [],
                    "std": [],
                }
                for j, imagedata_path in enumerate(period_files_df.path.tolist()):
                    # If file has filesize == 0, skip
                    imagedata_path = Path(imagedata_path)
                    if imagedata_path.stat().st_size == 0:
                        continue

                    # Read the file (but only the columns we need)
                    columns = list(statistic_columns_dict)
                    columns.append(id_column)

                    image_data_df = pdh.read_file(imagedata_path, columns=columns)
                    image_data_df.set_index(id_column, inplace=True)
                    image_data_df.index.name = id_column

                    # Remove rows with nan values
                    nb_before_dropna = len(image_data_df.index)
                    image_data_df.dropna(inplace=True)
                    nb_after_dropna = len(image_data_df.index)
                    if nb_after_dropna != nb_before_dropna:
                        logger.warning(
                            f"Before dropna: {nb_before_dropna}, after: "
                            f"{nb_after_dropna} for file {imagedata_path}"
                        )
                    if nb_after_dropna == 0:
                        continue

                    # recalculate duplicate rows (the -5 buffer can cause break ups?)
                    image_data_recalculate_df = (
                        image_data_df.loc[image_data_df.index.duplicated()]
                        .groupby(id_column)
                        .agg(dict.fromkeys(statistic_columns_dict, "mean"))
                    )
                    image_data_df = image_data_df.loc[~image_data_df.index.duplicated()]
                    image_data_df = pd.concat(
                        [image_data_df, image_data_recalculate_df]
                    )

                    # Rename columns so column names stay unique
                    for statistic_column in list(statistic_columns_dict):
                        new_column_name = statistic_column + str(j + 1)
                        image_data_df.rename(
                            columns={statistic_column: new_column_name}, inplace=True
                        )
                        image_data_df[new_column_name] = image_data_df[
                            new_column_name
                        ].astype(float)
                        statistic_columns_dict[statistic_column].append(new_column_name)

                    # Create 1 dataframe for all weekfiles
                    #   - one row for each code_obj
                    #   - using concat (code_obj = index)
                    if period_band_data_df is None:
                        period_band_data_df = image_data_df
                    else:
                        period_band_data_df = pd.concat(
                            [period_band_data_df, image_data_df], axis=1, sort=False
                        )
                        # Apparently concat removes the index name in some situations
                        period_band_data_df.index.name = id_column

                # Calculate max, mean, min, ...
                if period_band_data_df is not None:
                    logger.debug("Calculate max, mean, min, ...")
                    period_date_str_short = period_date.strftime("%Y%m%d")
                    # Remark: prefix column names: sqlite doesn't like a numeric start
                    if orbit is None:
                        column_basename = (
                            f"TS_{period_date_str_short}_{imagetype}_{band}"
                        )
                    else:
                        column_basename = (
                            f"TS_{period_date_str_short}_{imagetype}_{orbit}_{band}"
                        )

                    # Number of pixels
                    # TODO: onderzoeken hoe aantal pixels best bijgehouden wordt:
                    # afwijkingen weglaten ? max nemen ? ...
                    period_band_data_df[f"{column_basename}_count"] = np.nanmax(
                        period_band_data_df[statistic_columns_dict["count"]], axis=1
                    )
                    # Maximum of all max columns
                    period_band_data_df[f"{column_basename}_max"] = np.nanmax(
                        period_band_data_df[statistic_columns_dict["max"]], axis=1
                    )
                    # Mean of all mean columns
                    period_band_data_df[f"{column_basename}_mean"] = np.nanmean(
                        period_band_data_df[statistic_columns_dict["mean"]], axis=1
                    )
                    # Mean of all median columns
                    period_band_data_df[f"{column_basename}_median"] = np.nanmean(
                        period_band_data_df[statistic_columns_dict["median"]], axis=1
                    )
                    # Minimum of all min columns
                    period_band_data_df[f"{column_basename}_min"] = np.nanmin(
                        period_band_data_df[statistic_columns_dict["min"]], axis=1
                    )
                    # Mean of all std columns
                    period_band_data_df[f"{column_basename}_std"] = np.nanmean(
                        period_band_data_df[statistic_columns_dict["std"]], axis=1
                    )
                    # Number of Files used
                    period_band_data_df[f"{column_basename}_used_files"] = (
                        period_band_data_df[statistic_columns_dict["max"]].count(axis=1)
                    )

                    # Only keep the columns we want to keep
                    columns_to_keep = [
                        f"{column_basename}_count",
                        f"{column_basename}_max",
                        f"{column_basename}_mean",
                        f"{column_basename}_median",
                        f"{column_basename}_min",
                        f"{column_basename}_std",
                        f"{column_basename}_used_files",
                    ]
                    period_band_data_df = period_band_data_df[columns_to_keep]

                    # Merge the data with the other bands/orbits for this period
                    if period_data_df is None:
                        period_data_df = period_band_data_df
                    else:
                        period_data_df = pd.concat(
                            [period_band_data_df, period_data_df], axis=1, sort=False
                        )
                        # Apparently concat removes the index name in some situations
                        period_data_df.index.name = id_column

            if period_data_df is not None:
                logger.info(f"Write new file: {period_data_filename}")
                pdh.to_file(period_data_df, period_data_path)

                # Create pixcount file if it doesn't exist yet...
                if not pixcount_path.exists():
                    pixcount_s1s2_column = conf.columns["pixcount_s1s2"]

                    # Get max count of all count columns available
                    columns_to_use = [
                        column
                        for column in period_data_df.columns
                        if column.endswith("_count")
                    ]
                    period_data_df[pixcount_s1s2_column] = np.nanmax(
                        period_data_df[columns_to_use], axis=1
                    )

                    pixcount_df = period_data_df[pixcount_s1s2_column]
                    pixcount_df.fillna(value=0, inplace=True)
                    pdh.to_file(pixcount_df, pixcount_path)


def get_fileinfo_timeseries(path: Path) -> dict:
    """This function gets info of a timeseries data file.

    Args:
        path (Path): The path to the file to get info about.

    Returns:
        dict: a dict containing info about the file
    """
    try:
        # Split name on parcelinfo versus imageinfo
        filename_splitted = path.stem.split("__")
        parcel_part = filename_splitted[0]
        imageinfo_part = filename_splitted[1]

        # Extract imageinfo
        imageinfo_values = imageinfo_part.split("_")
        orbit = None
        image_profile = None
        time_reducer = None
        if "-" in imageinfo_values[0]:
            # OpenEO mosaic filename format
            image_profile = imageinfo_values[0]
            imageprofile_parts = image_profile.split("-")
            satellite = imageprofile_parts[0]
            product = imageprofile_parts[1]
            if satellite == "s1":
                if product in ["asc", "desc"]:
                    imagetype = IMAGETYPE_S1_GRD
                    orbit = product
                else:
                    imagetype = IMAGETYPE_S1_COHERENCE
            elif satellite == "s2":
                imagetype = IMAGETYPE_S2_L2A
            else:
                raise ValueError(f"invalid imageprofile in {path}")

            start_date = datetime.fromisoformat(imageinfo_values[1])
            end_date = datetime.fromisoformat(imageinfo_values[2])
            time_reducer = imageinfo_values[4]
            band = imageinfo_values[-1]  # =last value

        else:
            # ONDA/ESA filename format
            # Satellite
            satellite = imageinfo_values[0].lower()
            # Get the date taken from the filename, depending on the satellite type
            # Remark: the datetime is in this format: '20180101T055812'
            if satellite.startswith("s1"):
                # Check if it is a GRDH image
                if imageinfo_values[2] == "GRDH":
                    imagetype = IMAGETYPE_S1_GRD
                    filedatetime = imageinfo_values[4]
                elif imageinfo_values[1].startswith("S1"):
                    imagetype = IMAGETYPE_S1_COHERENCE
                    filedatetime = imageinfo_values[2]
                else:
                    raise ValueError(f"Unsupported file: {path}")
                # Also get the orbit
                orbit = imageinfo_values[-2].lower()  # =2nd last value

            elif satellite.startswith("s2"):
                imagetype = IMAGETYPE_S2_L2A
                filedatetime = imageinfo_values[2]
            else:
                raise ValueError(f"Unsupported file: {path}")

            # Parse the data found
            filedate = filedatetime.split("T")[0]
            parseddate = datetime.strptime(filedate, "%Y%m%d")
            start_date = parseddate
            end_date = start_date

            # Get the band
            band = imageinfo_values[-1]  # =last value

        # Week
        fileweek = int(start_date.strftime("%W"))

        # The file paths of these files sometimes are longer than 256
        # characters, so use trick on windows to support this anyway
        path_safe = path.as_posix()
        if os.name == "nt" and len(path.as_posix()) > 240:
            if path_safe.startswith("//"):
                path_safe = f"//?/UNC/{path_safe}"
            else:
                path_safe = f"//?/{path_safe}"

        image_metadata = {
            "path": path_safe,
            "parcel_stem": parcel_part,
            "imagetype": imagetype,
            "filestem": path.stem,
            "start_date": start_date,
            "end_date": end_date,
            "week": fileweek,
            "band": band,
            "orbit": orbit,  # ASC/DESC
        }
        if time_reducer is not None:
            image_metadata["time_reducer"] = time_reducer
        if image_profile is not None:
            image_metadata["image_profile"] = image_profile

    except Exception as ex:
        message = f"Error extracting info from filename {path}"
        logger.exception(message)
        raise Exception(message) from ex

    return image_metadata


def get_fileinfo_timeseries_periods(path: Path) -> dict:
    """This function gets info of a period timeseries data file.

    Args:
        path (Path): The path to the file to get info about.

    Returns:
        dict: a dict containing info about the file
    """
    # If there is no double underscore in name: "old file type"
    if "__" not in path.stem:
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 4:
            raise ValueError(f"stem doesn't contain enough parts: {path.name}")
        period_name = stem_parts[-3]
        start_date = datetime.fromisoformat(stem_parts[-2])
        if period_name.lower() == "weekly":
            end_date = start_date + timedelta(days=6)
        else:
            raise ValueError(f"unsupported period_name in {path}")
        return {
            "path": path,
            "parcel_stem": "_".join(stem_parts[0:-3]),
            "period_name": period_name,
            "start_date": start_date,
            "end_date": end_date,
            "image_profile": stem_parts[-1],
        }

    return get_fileinfo_timeseries(path)
