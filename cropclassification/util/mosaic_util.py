from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pyproj
from cropclassification.util import raster_index_util
from . import openeo_util
from .openeo_util import ImageProfile

# Get a logger...
logger = logging.getLogger(__name__)


def calc_periodic_mosaic(
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    images_to_get: List[ImageProfile],
    output_base_dir: Path,
    period_name: Optional[str] = None,
    delete_existing_openeo_jobs: bool = False,
    raise_errors: bool = True,
    force: bool = False,
) -> List[Tuple[Path, ImageProfile]]:
    """
    Generate a periodic mosaic.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[pyproj.CRS]): the CRS of the roi.
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
        days_per_period (int): number of days per period.
        images_to_get (List[ImageProfile]): list of imageprofiles to create the mosaic
            with.
        output_base_dir (Path): base directory to save the images to. The images will be
            saved in a subdirectory based on the image profile name.
        period_name (Optional[str], optional): name of the period. If None, default
            names are used: if ``days_per_period=7``: "weekly", if
            ``days_per_period=14``: "biweekly", for other values of ``days_per_period``
            a ValueError is thrown. Defaults to None.
        delete_existing_openeo_jobs (bool, optional): True to delete existing openeo
            jobs. If False, they are just left running and the results are downloaded if
            they are ready like other jobs. Defaults to False.
        raise_errors (bool, optional): True to raise if an error occurs. If False,
            errors are only logged. Defaults to True.
        force (bool, optional): True to force recreation of existing output files.
            Defaults to False.

    Raises:
        ValueError: _description_
        Exception: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        List[Tuple[Path, ImageProfile]]: _description_
    """
    images_to_get_openeo = []
    images_to_get_local = []
    for image_to_get in images_to_get:
        if image_to_get.image_source == "openeo":
            images_to_get_openeo.append(image_to_get)
        elif image_to_get.image_source == "local":
            images_to_get_local.append(image_to_get)
        else:
            raise ValueError(
                f"unsupported image_source in image_to_get: {image_to_get.image_source}"
            )

    periods = prepare_periods(
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        period_name=period_name,
    )

    # Fetch the images of openeo
    periodic_mosaic_params = []
    for period, imageprofile in zip(periods, images_to_get_openeo):
        image_path, image_relative_path = prepare_image_path(
            imageprofile.name,
            start_date=period["start_date"],
            end_date=period["end_date_incl"],
            bands=imageprofile.bands,
            time_dimension_reducer=imageprofile.process_options[
                "time_dimension_reducer"
            ],
            output_base_dir=output_base_dir,
        )
        if output_exists(image_path, force):
            continue

        periodic_mosaic_params.append(
            {
                "roi_bounds": roi_bounds,
                "roi_crs": roi_crs,
                "start_date": period["start_date"],
                "end_date": period["end_date"],
                "image_relative_path": image_relative_path,
                "imageprofile": imageprofile,
            }
        )

    _ = openeo_util.get_images(
        periodic_mosaic_params,
        output_base_dir=output_base_dir,
        delete_existing_openeo_jobs=delete_existing_openeo_jobs,
        raise_errors=raise_errors,
        force=force,
    )

    # Fetch the images of openeo
    for period, imageprofile in zip(periods, images_to_get_local):
        # Prepare index output file path
        index_path, index_relative_path = prepare_image_path(
            imageprofile.name,
            start_date=period["start_date"],
            end_date=period["end_date_incl"],
            bands=imageprofile.bands,
            time_dimension_reducer=imageprofile.process_options[
                "time_dimension_reducer"
            ],
            output_base_dir=output_base_dir,
        )
        if output_exists(image_path, force):
            continue

        # Prepare image file path the index is to be based on
        base_image_path, base_image_relative_path = prepare_image_path(
            imageprofile.base_image_profile,
            start_date=period["start_date"],
            end_date=period["end_date_incl"],
            bands=imageprofile.bands,
            time_dimension_reducer=imageprofile.process_options[
                "time_dimension_reducer"
            ],
            output_base_dir=output_base_dir,
        )
        if not base_image_path.exists():
            raise RuntimeError(
                f"base image doesn't exist to calculate index {imageprofile.name}: "
                f"{base_image_path}"
            )
        raster_index_util.calc_index(
            base_image_path,
            index_path,
            index=imageprofile.name,
            save_as_byte=True,
            force=force,
        )


def prepare_periods(
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    period_name: Optional[str] = None,
):
    """
    Download a periodic mosaic.

    Args:
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
        days_per_period (int): number of days per period.
        period_name (Optional[str], optional): name of the period. If None, default
            names are used: if ``days_per_period=7``: "weekly", if
            ``days_per_period=14``: "biweekly", for other values of ``days_per_period``
            a ValueError is thrown. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        List[dict]: list of dicts with info about the images to get
    """
    if start_date == end_date:
        raise ValueError(f"start date and end date are the same: {start_date}")
    if end_date > datetime.now():
        logger.warning(f"end_date is in the future: {end_date}")

    # Prepare period_name:
    if period_name is None:
        if days_per_period == 7:
            period_name = "weekly"
        elif days_per_period == 14:
            period_name = "biweekly"
        else:
            raise ValueError("Unknown period name, please specify")

    period_start_date = start_date

    result = []
    while period_start_date <= (end_date - timedelta(days=days_per_period)):
        # Period in openeo is inclusive for startdate and excludes enddate
        period_end_date = period_start_date + timedelta(days=days_per_period)
        if period_end_date > datetime.now():
            logger.info(
                f"skip period ({period_start_date}, {period_end_date}): it is in the "
                "future!"
            )
            break

        period_end_date_incl = period_end_date
        if days_per_period > 1:
            period_end_date_incl = period_end_date - timedelta(days=1)

        result.append(
            {
                "start_date": period_start_date,
                "end_date": period_end_date,
                "end_date_incl": period_end_date_incl,
            }
        )

        period_start_date = period_end_date

    return result


def prepare_periodic_mosaic_params(
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    images_to_get: List[ImageProfile],
    output_base_dir: Path,
    period_name: Optional[str] = None,
    force: bool = False,
) -> List[Tuple[Path, ImageProfile]]:
    """
    Download a periodic mosaic.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[pyproj.CRS]): the CRS of the roi.
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
        days_per_period (int): number of days per period.
        images_to_get (List[ImageProfile]): list of imageprofiles to create the mosaic
            with.
        output_base_dir (Path): base directory to save the images to. The images will be
            saved in a subdirectory based on the image profile name.
        period_name (Optional[str], optional): name of the period. If None, default
            names are used: if ``days_per_period=7``: "weekly", if
            ``days_per_period=14``: "biweekly", for other values of ``days_per_period``
            a ValueError is thrown. Defaults to None.
        delete_existing_openeo_jobs (bool, optional): True to delete existing openeo
            jobs. If False, they are just left running and the results are downloaded if
            they are ready like other jobs. Defaults to False.
        raise_errors (bool, optional): True to raise if an error occurs. If False,
            errors are only logged. Defaults to True.
        force (bool, optional): True to force recreation of existing output files.
            Defaults to False.

    Raises:
        ValueError: _description_
        Exception: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        List[dict]: list of dicts with info about the images to get
    """
    if start_date == end_date:
        raise ValueError(f"start date and end date are the same: {start_date}")
    if end_date > datetime.now():
        logger.warning(f"end_date is in the future: {end_date}")
    roi_bounds = list(roi_bounds)

    # Validate time_dimension_reducer
    for imageprofile in images_to_get:
        reducer = imageprofile.process_options.get("time_dimension_reducer", None)
        if reducer is None:
            raise ValueError(
                "process_options must contain a time_dimension_reducer for "
                "calc_periodic_mosaic"
            )

    # Prepare period_name:
    if period_name is None:
        if days_per_period == 7:
            period_name = "weekly"
        elif days_per_period == 14:
            period_name = "biweekly"
        else:
            raise ValueError("Unknown period name, please specify")

    period_start_date = start_date

    result = []
    while period_start_date <= (end_date - timedelta(days=days_per_period)):
        # Period in openeo is inclusive for startdate and excludes enddate
        period_end_date = period_start_date + timedelta(days=days_per_period)
        if period_end_date > datetime.now():
            logger.info(
                f"skip period ({period_start_date}, {period_end_date}): it is in the "
                "future!"
            )
            break

        for imageprofile in images_to_get:
            # Prepare path. Remark: in the image path the end_date should be inclusive.
            end_date_incl = period_end_date
            if days_per_period > 1:
                end_date_incl = period_end_date - timedelta(days=1)
            image_path, image_relative_path = prepare_image_path(
                imageprofile.name,
                start_date=period_start_date,
                end_date=end_date_incl,
                bands=imageprofile.bands,
                time_dimension_reducer=imageprofile.process_options[
                    "time_dimension_reducer"
                ],
                output_base_dir=output_base_dir,
            )
            if output_exists(image_path, force):
                continue

            result.append(
                {
                    "roi_bounds": roi_bounds,
                    "roi_crs": roi_crs,
                    "start_date": period_start_date,
                    "end_date": period_end_date,
                    "image_relative_path": image_relative_path,
                    "imageprofile": imageprofile,
                }
            )

        period_start_date = period_end_date

    return result


def output_exists(path: Path, force: bool, log_prefix: Optional[str] = None) -> bool:
    """
    Check if the output file exists already. If force is True, the file is removed.

    Args:
        path (Path): Output file path to check.
        force (bool): If True, remove the output file if it exists.
        log_prefix (str, optional): Prefix to use when logging that the file already
            exists. Can be used to give more context in the logging. Defaults to None.

    Raises:
        ValueError: raised when the output directory does not exist.

    Returns:
        bool: True if the file exists.
    """
    if not path.parent.exists():
        raise ValueError(f"output directory does not exist: {path.parent}")
    if path.exists():
        if force:
            path.unlink()
            return False
        else:
            log_prefix = f"{log_prefix}: " if log_prefix is not None else ""
            logger.info(f"{log_prefix}force is False and {path.name} exists already")
            return True

    return False


def prepare_image_path(
    imageprofile: str,
    start_date: datetime,
    end_date: datetime,
    bands: List[str],
    time_dimension_reducer: str,
    output_base_dir: Path,
) -> Tuple[Path, str]:
    """
    Returns an image_path + saves a metadata file for the image as f"{image_path}.json".

    Args:
        imageprofile (str): _description_
        start_date (datetime): _description_
        end_date (datetime):
        bands (List[str]): _description_
        output_base_dir (Path): _description_

    Raises:
        ValueError: _description_

    Returns:
        Tuple[Path, str]: the full path + the relative path
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    weeks = list(
        range(int(start_date.strftime("%W")), int(start_date.strftime("%W")) + 1)
    )
    # Concat bands, but remove all chars used as separators
    bands_str = "-".join([band.replace("_", "").replace("-", "") for band in bands])
    name = (
        f"{imageprofile}_{start_date_str}_{end_date_str}_{bands_str}_"
        f"{time_dimension_reducer}.tif"
    )

    # If the image metadata file doesn't exist, create it
    image_dir = output_base_dir / imageprofile
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / name
    imagemeta_path = image_dir / f"{name}.json"
    if not imagemeta_path.exists():
        imageprofile_parts = imageprofile.split("-")
        satellite = imageprofile_parts[0].lower()
        metadata = {
            "imageprofile": imageprofile,
            "satellite": satellite,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "weeks": weeks,
            "bands": bands,
            "time_dimension_reducer": time_dimension_reducer,
            "path": image_path.as_posix(),
        }
        if imageprofile.lower() == "s1-grd-sigma0-asc":
            metadata["orbit"] = "asc"
        elif imageprofile.lower() == "s1-grd-sigma0-desc":
            metadata["orbit"] = "desc"

        # Write to file
        with open(imagemeta_path, "w") as outfile:
            outfile.write(json.dumps(metadata, indent=4))

    return (image_path, f"{imageprofile}/{name}")
