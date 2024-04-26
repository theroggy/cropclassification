from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyproj

from cropclassification.util import raster_util
from . import raster_index_util
from . import openeo_util

# Get a logger...
logger = logging.getLogger(__name__)


@dataclass
class ImageProfile:
    """
    Profile of the image to be processed via openeo.
    """

    name: str
    satellite: str
    image_source: str
    collection: Optional[str] = None
    bands: Optional[List[str]] = None
    base_image_profile: Optional[str] = None
    index_type: Optional[str] = None
    max_cloud_cover: Optional[float] = None
    process_options: Optional[dict] = None
    job_options: Optional[dict] = None

    def __init__(
        self,
        name: str,
        satellite: str,
        image_source: str,
        collection: Optional[str] = None,
        bands: Optional[List[str]] = None,
        base_image_profile: Optional[str] = None,
        index_type: Optional[str] = None,
        max_cloud_cover: Optional[float] = None,
        process_options: Optional[dict] = None,
        job_options: Optional[dict] = None,
    ):
        self.name = name
        self.satellite = satellite
        self.image_source = image_source
        self.collection = collection
        self.bands = bands
        self.base_image_profile = base_image_profile
        self.index_type = index_type
        self.max_cloud_cover = max_cloud_cover
        self.process_options = process_options
        self.job_options = job_options

        # Some data validations
        if image_source == "local":
            if collection is not None:
                raise ValueError(f"collection must be None if {image_source=}, {self}")
            elif bands is not None:
                raise ValueError(f"bands must be None if {image_source=}, {self}")
            elif index_type is None:
                raise ValueError(f"index_type can't be None if {image_source=}, {self}")
            elif base_image_profile is None:
                raise ValueError(
                    f"base_image_profile can't be None if {image_source=}, {self}"
                )
        elif image_source == "openeo":
            if collection is None:
                raise ValueError(f"collection can't be None if {image_source=}, {self}")
            elif bands is None:
                raise ValueError(f"bands can't be None if {image_source=}, {self}")
            elif base_image_profile is not None:
                raise ValueError(
                    f"base_image_profile must be None if {image_source=}, {self}"
                )
        else:
            raise ValueError(f"{image_source=} is not supported, {self}")


def calc_periodic_mosaic(
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    days_per_period: int,
    time_dimension_reducer: str,
    imageprofiles_to_get: List[str],
    imageprofiles: Dict[str, ImageProfile],
    output_base_dir: Path,
    period_name: Optional[str] = None,
    delete_existing_openeo_jobs: bool = False,
    raise_errors: bool = True,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a periodic mosaic.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[pyproj.CRS]): the CRS of the roi.
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
        days_per_period (int): number of days per period.
        time_dimension_reducer (str): reducer to use to aggregate pixels in time
            dimension: one of: "mean", "min", "max",...
        imageprofiles_to_get (List[str]): list of image proles a periodic mosaic should
            be generated for.
        imageprofiles (Dict[str, ImageProfile]): dict with for all configured image
            profiles the image profile name as key, the ImageProfile as value. Should
            contain at least all profiles needed to process the image profiles listed in
            imageprofiles_to_get.
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

    Returns:
        List[Dict[str, Any]]: list with information about all mosaic image calculated.
    """
    # Prepare the periods to calculate mosaic images for.
    periods = _prepare_periods(
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        period_name=period_name,
    )

    periodic_mosaic_params = _prepare_periodic_mosaic_params(
        roi_bounds=roi_bounds,
        roi_crs=roi_crs,
        periods=periods,
        time_dimension_reducer=time_dimension_reducer,
        imageprofiles_to_get=imageprofiles_to_get,
        imageprofiles=imageprofiles,
        output_base_dir=output_base_dir,
    )

    # Split images to get by image_source.
    images_from_openeo = []
    images_local = []
    for image_params in periodic_mosaic_params:
        if image_params["image_source"] == "openeo":
            images_from_openeo.append(image_params)
        elif image_params["image_source"] == "local":
            images_local.append(image_params)
        else:
            raise ValueError(f"unsupported image_source in {image_params=}")

    # Make sure band information is embedded in the image
    for image in images_from_openeo:
        if image["path"].exists():
            raster_util.set_band_descriptions(
                image["path"], band_descriptions=image["bands"], overwrite=False
            )

    # First get all mosaic images from openeo
    _ = openeo_util.get_images(
        images_from_openeo,
        delete_existing_openeo_jobs=delete_existing_openeo_jobs,
        raise_errors=raise_errors,
        force=force,
    )

    # Process the mosaic images to be generated locally.
    for image_local in images_local:
        # Prepare index output file path
        raster_index_util.calc_index(
            image_local["base_image_path"],
            image_local["path"],
            index=image_local["index_type"],
            save_as_byte=True,
            force=force,
        )

    return periodic_mosaic_params


def _prepare_periods(
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


def _prepare_periodic_mosaic_params(
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    periods: dict,
    time_dimension_reducer: str,
    imageprofiles_to_get: List[str],
    imageprofiles: Dict[str, ImageProfile],
    output_base_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Prepare the parameters needed to generate a list of periodic mosaics.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[pyproj.CRS]): the CRS of the roi.
        periods (dict): ???.
        time_dimension_reducer (str): reducer to use to aggregate pixels in time
            dimension: one of: "mean", "min", "max",...
        imageprofiles_to_get (List[str]): list of image proles a periodic mosaic should
            be generated for.
        imageprofiles (Dict[str, ImageProfile]): dict with for all configured image
            profiles the image profile name as key, the ImageProfile as value. Should
            contain at least all profiles needed to process the image profiles listed in
            imageprofiles_to_get.
        output_base_dir (Path): base directory to save the images to. The images will be
            saved in a subdirectory based on the image profile name.

    Returns:
        List[Dict[str, Any]]: list of dicts with all neededparameters to generate the
            mosaic images.
    """
    # Prepare full list of image mosaics we want to calculate.
    # Use a dict indexed on path to avoid having duplicate mosaic_image_params.
    periodic_mosaic_params: Dict[str, Dict[str, Any]] = {}
    for period, imageprofile in product(periods, imageprofiles_to_get):
        main_image_params = _prepare_mosaic_image_params(
            roi_bounds=roi_bounds,
            roi_crs=roi_crs,
            imageprofile=imageprofiles[imageprofile],
            period=period,
            time_dimension_reducer=time_dimension_reducer,
            output_base_dir=output_base_dir,
        )

        # If the image isn't calculated locally, just add main image and continue.
        if imageprofiles[imageprofile].image_source != "local":
            periodic_mosaic_params[main_image_params["path"]] = main_image_params
            continue

        # Image calculated locally... so we need a base image.
        base_image_profile = imageprofiles[imageprofile].base_image_profile
        if base_image_profile is None:
            raise ValueError(
                "generating an image locally needs a base image "
                f"{imageprofiles[imageprofile]}"
            )
        base_image_params = _prepare_mosaic_image_params(
            roi_bounds=roi_bounds,
            roi_crs=roi_crs,
            imageprofile=imageprofiles[base_image_profile],
            period=period,
            time_dimension_reducer=time_dimension_reducer,
            output_base_dir=output_base_dir,
        )
        periodic_mosaic_params[base_image_params["path"]] = base_image_params

        # Add base image path to main image + add to list
        main_image_params["base_image_path"] = base_image_params["path"]
        periodic_mosaic_params[main_image_params["path"]] = main_image_params

    return list(periodic_mosaic_params.values())


def _prepare_mosaic_image_params(
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    imageprofile: ImageProfile,
    period: dict,
    time_dimension_reducer: str,
    output_base_dir: Path,
) -> Dict[str, Any]:
    """
    Prepares the relevant parameters + saves them as json file.

    Args:
        roi_bounds (Tuple[float, float, float, float]): _description_
        roi_crs (Optional[pyproj.CRS]): _description_
        imageprofile (ImageProfile): _description_
        period (dict): _description_
        time_dimension_reducer (str): _description_
        output_base_dir (Path): _description_

    Returns:
        Dict[str, Any]: _description_
    """
    # Prepare image path
    image_path = _prepare_mosaic_image_path(
        imageprofile.name,
        start_date=period["start_date"],
        end_date=period["end_date_incl"],
        bands=imageprofile.bands,
        time_dimension_reducer=time_dimension_reducer,
        output_base_dir=output_base_dir,
    )

    # Prepare image metadata to write it to json + to return it
    weeks = list(
        range(
            int(period["start_date"].strftime("%W")),
            int(period["end_date_incl"].strftime("%W")) + 1,
        )
    )
    imageprofile_parts = imageprofile.name.split("-")
    satellite = imageprofile_parts[0].lower()
    imagemeta = {
        "imageprofile": imageprofile.name,
        "collection": imageprofile.collection,
        "index_type": imageprofile.index_type,
        "max_cloud_cover": imageprofile.max_cloud_cover,
        "image_source": imageprofile.image_source,
        "satellite": satellite,
        "roi_bounds": roi_bounds,
        "roi_crs": roi_crs,
        "start_date": period["start_date"],
        "end_date_incl": period["end_date_incl"],
        "end_date": period["end_date"],
        "weeks": weeks,
        "bands": imageprofile.bands,
        "time_dimension_reducer": time_dimension_reducer,
        "path": image_path,
    }
    if imageprofile.name.lower() == "s1-grd-sigma0-asc":
        imagemeta["orbit"] = "asc"
    elif imageprofile.name.lower() == "s1-grd-sigma0-desc":
        imagemeta["orbit"] = "desc"

    imagemeta_path = Path(f"{image_path}.json")
    if not imagemeta_path.exists():
        _imagemeta_to_file(imagemeta_path, imagemeta)

    return imagemeta


def _prepare_mosaic_image_path(
    imageprofile: str,
    start_date: datetime,
    end_date: datetime,
    bands: List[str],
    time_dimension_reducer: str,
    output_base_dir: Path,
) -> Path:
    """
    Returns an image_path.

    Args:
        imageprofile (str): name of the image profile.
        start_date (datetime): start date of the mosaic.
        end_date (datetime): end date of the mosaic.
        bands (List[str]): list of bands that will be in the file.
        output_base_dir (Path): base directory to put the file in.

    Raises:
        ValueError: _description_

    Returns:
        Path: the path
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Concat bands, but remove all chars used as separators
    if bands is not None:
        bands_str = "-".join([band.replace("_", "").replace("-", "") for band in bands])
    else:
        bands_str = imageprofile.split("-")[1]
    name = (
        f"{imageprofile}_{start_date_str}_{end_date_str}_{bands_str}_"
        f"{time_dimension_reducer}.tif"
    )

    image_dir = output_base_dir / imageprofile
    image_path = image_dir / name

    return image_path


def _imagemeta_to_file(path: Path, imagemeta: Dict[str, Any]):
    # Write to file
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as outfile:
        # Prepare some properties that don't serialize automatically
        imagemeta_prepared = deepcopy(imagemeta)
        imagemeta_prepared["start_date"] = imagemeta["start_date"].strftime("%Y-%m-%d")
        imagemeta_prepared["end_date_incl"] = imagemeta["end_date_incl"].strftime(
            "%Y-%m-%d"
        )
        imagemeta_prepared["end_date"] = imagemeta["end_date"].strftime("%Y-%m-%d")
        imagemeta_prepared["path"] = imagemeta["path"].as_posix()

        outfile.write(json.dumps(imagemeta_prepared, indent=4))
