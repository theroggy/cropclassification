"""Calculates periodic timeseries for input parcels."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import geofileops as gfo

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
        if temp_empty_path.exists():
            gfo.remove(temp_empty_path)
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
