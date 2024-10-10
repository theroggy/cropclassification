"""Utility functions to generate a periodic mosaic."""

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pyproj
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import date_util, openeo_util, raster_index_util, raster_util

# Get a logger...
logger = logging.getLogger(__name__)


@dataclass
class ImageProfile:
    """Profile of the image to be calculated."""

    name: str
    satellite: str
    image_source: str
    bands: Optional[list[str]] = None
    collection: Optional[str] = None
    time_reducer: Optional[str] = None
    period_name: Optional[str] = None
    period_days: Optional[int] = None
    base_imageprofile: Optional[str] = None
    index_type: Optional[str] = None
    pixel_type: Optional[str] = None
    max_cloud_cover: Optional[float] = None
    process_options: Optional[dict] = None
    job_options: Optional[dict] = None

    def __init__(
        self,
        name: str,
        satellite: str,
        image_source: str,
        bands: list[str],
        collection: Optional[str] = None,
        time_reducer: Optional[str] = None,
        period_name: Optional[str] = None,
        period_days: Optional[int] = None,
        base_image_profile: Optional[str] = None,
        index_type: Optional[str] = None,
        pixel_type: Optional[str] = None,
        max_cloud_cover: Optional[float] = None,
        process_options: Optional[dict] = None,
        job_options: Optional[dict] = None,
    ):
        """Profile of the image to be calculated.

        The `period_name` and `period_days` are used to determine the periods of the
        images to be generated. For some pre-defined values of `period_name`, the
        periods will be aligned to certain week days and/or weeks of the year:
          - "weekly": image periods will start on a monday and end on a sunday.
          - "biweekly": image periods will start on even mondays of the year.

        Args:
            name (str): name of the image profile. To be choosen by the user, but
                typically a combination of the satellite, image sensor and the period
                name.
            satellite (str): satellite the image is from.
            image_source (str): source where the images should be generated. One of
                "local" and "openeo".
            bands (List[str]): bands to include in the image.
            collection (Optional[str], optional): image collection to get the input
                images from. Defaults to None.
            time_reducer (Optional[str], optional): _description_. Defaults to None.
            period_name (Optional[str], optional): name of the period. If None, default
                names are used: if ``days_per_period=7``: "weekly", if
                ``days_per_period=14``: "biweekly", for other values of
                ``days_per_period`` a ValueError is thrown. Defaults to None.
            period_days (int, optional): number of days per period. If None, it is
                derived of the `period_name` if possible. Defaults to None.
            index_type (Optional[str], optional): only supported if
                `image_source="local"`. The index to calculate based on another image.
                One of "ndvi", "bsi", "dprvi". Defaults to None.
            base_image_profile (Optional[str], optional): only supported if
                `image_source="local"`. The image profile of the image an index image
                should be based on. Defaults to None.
            pixel_type (Optional[str], optional): pixel type to use when saving. E.g.
                BYTE, FLOAT16, FLOAT32. Defaults to None.
            max_cloud_cover (Optional[float], optional): the maximum cloud cover an
                input image can have to be used. Defaults to None.
            process_options (Optional[dict], optional): extra process options.
                Defaults to None.
            job_options (Optional[dict], optional): extra job options.
                Defaults to None.

        Raises:
            ValueError: when invalid parameters are passed in.
        """
        self.name = name
        self.satellite = satellite
        self.image_source = image_source
        self.collection = collection
        self.bands = bands
        self.time_reducer = time_reducer
        self.period_name = period_name
        self.period_days = period_days
        self.base_imageprofile = base_image_profile
        self.index_type = index_type
        self.pixel_type = pixel_type
        self.max_cloud_cover = max_cloud_cover
        self.process_options = process_options
        self.job_options = job_options

        # Some data validations
        if image_source == "local":
            errors = []
            if collection is not None:
                errors.append(f"collection must be None if {image_source=}")
            if period_name is not None:
                errors.append(f"period_name must be None if {image_source=}")
            if period_days is not None:
                errors.append(f"period_days must be None if {image_source=}")
            if index_type is None:
                errors.append(f"index_type can't be None if {image_source=}")
            if base_image_profile is None:
                errors.append(f"base_image_profile can't be None if {image_source=}")
            if pixel_type is None:
                errors.append(f"pixel_type can't be None if {image_source=}")

            if len(errors) > 0:
                raise ValueError(f"Invalid input in init of {self}: {errors}")
        elif image_source == "openeo":
            errors = []
            if collection is None:
                errors.append(f"collection can't be None if {image_source=}")
            if base_image_profile is not None:
                errors.append(f"base_image_profile must be None if {image_source=}")
            if pixel_type is not None:
                errors.append(f"pixel_type must be None if {image_source=}")

            # The period name and days need some extra validation/preprocessing
            try:
                self.period_name, self.period_days = _prepare_period_params(
                    period_name, period_days
                )
            except Exception as ex:
                errors.append(str(ex))

            if len(errors) > 0:
                raise ValueError(f"Invalid input in init of {self}: {errors}")
        else:
            raise ValueError(f"{image_source=} is not supported, {self}")


def calc_periodic_mosaic(
    roi_bounds: tuple[float, float, float, float],
    roi_crs: Optional[Any],
    start_date: datetime,
    end_date: datetime,
    imageprofiles_to_get: list[str],
    imageprofiles: dict[str, ImageProfile],
    output_base_dir: Path,
    delete_existing_openeo_jobs: bool = False,
    on_missing_image: str = "calculate_raise",
    force: bool = False,
) -> list[dict[str, Any]]:
    """Generate a periodic mosaic.

    Depending on the period_name specified in the ImageProfiles, the `start_date` and
    `end_date` will  be adjusted:
      - "weekly": image periods will start on a monday and end on a sunday. If the
        `start_date` is no monday, the first monday following it will be used. If the
        `end_date` (exclusive) is no monday, the first monday before it will be used.
      - "biweekly": image periods will start on even mondays of the year. If the
        `start_date` is no monday, the first even monday following it will be used.
        If the `end_date` (exclusive) is no monday, the first even monday before it
        will be used.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[Any]): the CRS of the roi.
        start_date (datetime): start date, included. Depending on the period used in the
            imageprofiles, the start_date might be adjusted to e.g. the next monday for
            "weekly",...
        end_date (datetime): end date, excluded. Depending on the period used in the
            imageprofiles, the end_date might be adjusted to e.g. the previous monday
            for "weekly",...
        imageprofiles_to_get (List[str]): list of image proles a periodic mosaic should
            be generated for.
        imageprofiles (Dict[str, ImageProfile]): dict with for all configured image
            profiles the image profile name as key, the ImageProfile as value. Should
            contain at least all profiles needed to process the image profiles listed in
            imageprofiles_to_get.
        output_base_dir (Path): base directory to save the images to. The images will be
            saved in a subdirectory based on the image profile name.
        delete_existing_openeo_jobs (bool, optional): True to delete existing openeo
            jobs. If False, they are just left running and the results are downloaded if
            they are ready like other jobs. Defaults to False.
        on_missing_image (str, optional): what to do when an image is missing. Defaults
            to "calculate_raise". Options are:
              - ignore: ignore that the image, don't try to download it
              - calculate_raise: calculate the image and raise an error if it fails
              - calculate_ignore: calculate the image and ignore the error if it fails

        force (bool, optional): True to force recreation of existing output files.
            Defaults to False.

    Returns:
        List[Dict[str, Any]]: list with information about all mosaic image calculated.
    """
    on_missing_image_values = ["ignore", "calculate_raise", "calculate_ignore"]
    if on_missing_image is None or on_missing_image not in on_missing_image_values:
        raise ValueError(
            f"invalid value for {on_missing_image=}: expected one of "
            "{on_missing_image_values}"
        )

    # Prepare the params that can be used to calculate mosaic images.
    periodic_mosaic_params = _prepare_periodic_mosaic_params(
        roi_bounds=roi_bounds,
        roi_crs=roi_crs,
        start_date=start_date,
        end_date=end_date,
        imageprofiles_to_get=imageprofiles_to_get,
        imageprofiles=imageprofiles,
        output_base_dir=output_base_dir,
    )

    if on_missing_image != "ignore":
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
            raise_errors="raise" in on_missing_image,
            force=force,
        )

        # Process the mosaic images to be generated locally.
        with logging_redirect_tqdm():
            for image_local in tqdm(images_local):
                # Prepare index output file path
                raster_index_util.calc_index(
                    image_local["base_image_path"],
                    image_local["path"],
                    index=image_local["index_type"],
                    pixel_type=image_local["pixel_type"],
                    process_options=image_local["process_options"],
                    force=force,
                )

    return periodic_mosaic_params


def _prepare_periods(
    start_date: datetime,
    end_date: datetime,
    period_name: Optional[str],
    period_days: Optional[int],
) -> list[dict[str, Any]]:
    """Prepare the periods to download a periodic mosaic.

    Args:
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
        period_name (str): the name of the periods to use.
        period_days (Optional[int]): number of days per period.

    Raises:
        ValueError: invalid input parameter values were passed in.

    Returns:
        list[dict[str, Any]]: list of dicts with info about the images to get
    """
    period_name, period_days = _prepare_period_params(period_name, period_days)

    # Adjust start_date and end_date based on the period specified
    if period_name == "weekly":
        start_date = date_util.get_monday(start_date, before=False)
        end_date = date_util.get_monday(end_date, before=True)
    elif period_name == "biweekly":
        # Also make sure the start date is compatible with using biweekly mondays
        # starting from the first monday of the year.
        start_date = date_util.get_monday_biweekly(start_date, before=False)
        end_date = date_util.get_monday_biweekly(end_date, before=True)

    # Check if start date and end date are (still) valid
    if start_date == end_date:
        raise ValueError(f"start_date == end_date: this is not supported: {start_date}")
    if end_date > datetime.now():
        logger.warning(f"end_date is in the future: {end_date}")

    period_start_date = start_date

    result = []
    while period_start_date <= (end_date - timedelta(days=period_days)):
        # Period in openeo is inclusive for startdate and excludes enddate
        period_end_date = period_start_date + timedelta(days=period_days)
        if period_end_date > datetime.now():
            logger.info(
                f"skip period ({period_start_date}, {period_end_date}): it is in the "
                "future!"
            )
            break

        period_end_date_incl = period_end_date
        if period_days > 1:
            period_end_date_incl = period_end_date - timedelta(days=1)

        result.append(
            {
                "start_date": period_start_date,
                "end_date": period_end_date,
                "end_date_incl": period_end_date_incl,
                "period_name": period_name,
            }
        )

        period_start_date = period_end_date

    return result


def _prepare_period_params(
    period_name: Optional[str], period_days: Optional[int]
) -> tuple[str, int]:
    """Interprete period_name and period_days and return cleaned version.

    Args:
        period_name (str): _description_
        period_days (Optional[int]): _description_

    Raises:
        ValueError: if invalid input is passed in.

    Returns:
        Tuple[str, int]: cleaned up period_name and period_days.
    """
    if period_name is None:
        if period_days is None:
            raise ValueError("both period_name and period_days are None")
        elif period_days == 7:
            period_name = "weekly"
        elif period_days == 14:
            period_name = "biweekly"
        else:
            raise ValueError("period_name is None and period_days is not 7 or 14")
    elif period_name == "weekly":
        period_days = 7
    elif period_name == "biweekly":
        period_days = 14
    else:
        if period_days is None:
            raise ValueError(
                "If period_name is not one of the basic names, period_days must be "
                "specified."
            )

    return (period_name, period_days)


def _prepare_periodic_mosaic_params(
    roi_bounds: tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    imageprofiles_to_get: list[str],
    imageprofiles: dict[str, ImageProfile],
    output_base_dir: Path,
) -> list[dict[str, Any]]:
    """Prepare the parameters needed to generate a list of periodic mosaics.

    Args:
        roi_bounds (Tuple[float, float, float, float]): bounds (xmin, ymin, xmax, ymax)
            of the region of interest to download the mosaic for.
        roi_crs (Optional[pyproj.CRS]): the CRS of the roi.
        start_date (datetime): start date, included.
        end_date (datetime): end date, excluded.
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
    # Check input
    if start_date >= end_date:
        raise ValueError(f"{start_date=} should be before {end_date=}")

    # Prepare full list of image mosaics we want to calculate.
    # Use a dict indexed on path to avoid having duplicate mosaic_image_params.
    periodic_mosaic_params: dict[str, dict[str, Any]] = {}
    for imageprofile in imageprofiles_to_get:
        # If the image is calculated locally, we need a base image profile.
        base_imageprofile = None
        if imageprofiles[imageprofile].image_source == "local":
            base_imageprofile_name = imageprofiles[imageprofile].base_imageprofile
            if base_imageprofile_name is not None:
                base_imageprofile = imageprofiles[base_imageprofile_name]

        # For local images, determine period info from base image profile
        if base_imageprofile is None:
            period_name = imageprofiles[imageprofile].period_name
            period_days = imageprofiles[imageprofile].period_days
        else:
            period_name = base_imageprofile.period_name
            period_days = base_imageprofile.period_days

        # First determine list of periods needed
        assert period_name is not None
        periods = _prepare_periods(
            start_date=start_date,
            end_date=end_date,
            period_name=period_name,
            period_days=period_days,
        )

        # Now determine all parameters for each period
        for period in periods:
            main_image_params = _prepare_mosaic_image_params(
                roi_bounds=roi_bounds,
                roi_crs=roi_crs,
                imageprofile=imageprofiles[imageprofile],
                period=period,
                output_base_dir=output_base_dir,
                base_imageprofile=base_imageprofile,
            )

            # If the image isn't calculated locally, just add main image and continue.
            if imageprofiles[imageprofile].image_source != "local":
                periodic_mosaic_params[main_image_params["path"]] = main_image_params
                continue

            # Image calculated locally... so we need a base image.
            base_image_profile = imageprofiles[imageprofile].base_imageprofile
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
                output_base_dir=output_base_dir,
            )
            periodic_mosaic_params[base_image_params["path"]] = base_image_params

            # Add base image path to main image + add to list
            main_image_params["base_image_path"] = base_image_params["path"]
            periodic_mosaic_params[main_image_params["path"]] = main_image_params

    return list(periodic_mosaic_params.values())


def _prepare_mosaic_image_params(
    roi_bounds: tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    imageprofile: ImageProfile,
    period: dict[str, Any],
    output_base_dir: Path,
    base_imageprofile: Optional[ImageProfile] = None,
) -> dict[str, Any]:
    """Prepares the relevant parameters + saves them as json file.

    Args:
        roi_bounds (Tuple[float, float, float, float]): _description_
        roi_crs (Optional[pyproj.CRS]): _description_
        imageprofile (ImageProfile): _description_
        period (Dict[str, Any]): period information: a dict with the keys start_date,
            end_date, end_date_incl and period_name.
        output_base_dir (Path): _description_
        base_imageprofile (ImageProfile, optional): Defaults to None.

    Returns:
        Dict[str, Any]: _description_
    """
    # Prepare image path
    assert imageprofile.bands is not None
    time_reducer = imageprofile.time_reducer
    if time_reducer is None and base_imageprofile is not None:
        time_reducer = base_imageprofile.time_reducer
    assert time_reducer is not None

    image_path = _prepare_mosaic_image_path(
        imageprofile.name,
        start_date=period["start_date"],
        end_date=period["end_date_incl"],
        bands=imageprofile.bands,
        time_reducer=time_reducer,
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
        "base_imageprofile": imageprofile.base_imageprofile,
        "pixel_type": imageprofile.pixel_type,
        "satellite": satellite,
        "roi_bounds": roi_bounds,
        "roi_crs": roi_crs,
        "start_date": period["start_date"],
        "end_date_incl": period["end_date_incl"],
        "end_date": period["end_date"],
        "period_name": period["period_name"],
        "weeks": weeks,
        "bands": imageprofile.bands,
        "time_reducer": time_reducer,
        "path": image_path,
        "job_options": imageprofile.job_options,
        "process_options": imageprofile.process_options,
    }
    if imageprofile.name.lower() == "s1-grd-sigma0-asc":
        imagemeta["orbit"] = "asc"
    elif imageprofile.name.lower() == "s1-grd-sigma0-desc":
        imagemeta["orbit"] = "desc"

    imagemeta_path = Path(f"{image_path}.json")
    if not imagemeta_path.exists() or imagemeta_path.stat().st_size == 0:
        _imagemeta_to_file(imagemeta_path, imagemeta)

    return imagemeta


def _prepare_mosaic_image_path(
    imageprofile: str,
    start_date: datetime,
    end_date: datetime,
    bands: list[str],
    time_reducer: str,
    output_base_dir: Path,
) -> Path:
    """Returns an image_path.

    Args:
        imageprofile (str): name of the image profile.
        start_date (datetime): start date of the mosaic.
        end_date (datetime): end date of the mosaic.
        bands (List[str]): list of bands that will be in the file.
        time_reducer (str): reducer to use to aggregate pixels in time dimension: one
            of: "mean", "min", "max",...
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
        f"{time_reducer}.tif"
    )

    image_dir = output_base_dir / imageprofile
    image_path = image_dir / name

    return image_path


def _imagemeta_to_file(path: Path, imagemeta: dict[str, Any]):
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
        imagemeta_prepared["roi_crs"] = str(imagemeta["roi_crs"])

        outfile.write(json.dumps(imagemeta_prepared, indent=4))
