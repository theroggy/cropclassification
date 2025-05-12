import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import geofileops as gfo
import pyproj
import shapely

from cropclassification.helpers import config_helper as conf
from cropclassification.util import mosaic_util, zonal_stats_bulk
from cropclassification.util.mosaic_util import ImageProfile

# Get a logger...
logger = logging.getLogger(__name__)


def calculate_periodic_timeseries(
    input_parcel_path: Path,
    roi_bounds: tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    imageprofiles_to_get: list[str],
    imageprofiles: dict[str, ImageProfile],
    images_periodic_dir: Path,
    timeseries_periodic_dir: Path,
    engine: str,
    stats: list[str],
    nb_parallel: int,
    on_missing_image: str,
):
    """Calculate timeseries data for the input parcels.

    Args:
        input_parcel_path (Path): path to the input parcel file.
        roi_bounds (tuple[float, float, float, float]): bounds of the roi.
        roi_crs (Optional[pyproj.CRS]): crs of the roi bounds.
        start_date (datetime): start date of the timeseries.
        end_date (datetime): end date of the timeseries (non inclusive).
        imageprofiles_to_get (list[str]): an array with images you want to be
            calculated.
        imageprofiles (dict[str, ImageProfile]): imageprofiles to use for the
            calculation.
        images_periodic_dir (Path): directory where the images are stored.
        timeseries_periodic_dir (Path): directory where the timeseries data will be
            saved.
        engine (str): the engine to use for the calculation. Options are
            "exactextract", "rasterstats" and "pyqgis".
        stats (list[str]): statistics to calculate. Available statistics and
            special options are dependent on the `engine` specified:

                - "rasterstats": `rasterstats documentation <https://pythonhosted.org/rasterstats/manual.html#statistics>`_
                - "pyqgis": "count", "sum", "mean", "median", "std", "min", "max",
                        "range", "minority", "majority" and "variance".
                - "exactextract": `exactextract documentation <https://isciences.github.io/exactextract/operations.html>`_

        nb_parallel (int): number of parallel processes to use.
        on_missing_image (str): what to do when an image is missing. Options are:

            - ignore: ignore that the image, don't try to download it
            - calculate_raise: calculate the image and raise an error if it fails
            - calculate_ignore: calculate the image and ignore the error if it fails
    """
    info = gfo.get_layerinfo(input_parcel_path)
    if info.crs is not None and not info.crs.equals(roi_crs):
        raise ValueError(f"parcel crs ({info.crs}) <> roi crs ({roi_crs})")
    if not shapely.box(*info.total_bounds).within(shapely.box(*roi_bounds)):
        raise ValueError(
            f"parcel bounds ({info.total_bounds}) not within roi_bounds ({roi_bounds})"
        )

    periodic_images_result = mosaic_util.calc_periodic_mosaic(
        roi_bounds=roi_bounds,
        roi_crs=roi_crs,
        start_date=start_date,
        end_date=end_date,
        output_base_dir=images_periodic_dir,
        imageprofiles_to_get=imageprofiles_to_get,
        imageprofiles=imageprofiles,
        on_missing_image=on_missing_image,
        force=False,
    )

    # Now calculate the timeseries
    images_bands = []
    for image_info in periodic_images_result:
        if not image_info["path"].exists():
            if on_missing_image in ("ignore", "calculate_ignore"):
                logger.info(f"Image {image_info['path']} is missing: ignore")
                continue
            else:
                raise RuntimeError(f"Image {image_info['path']} is missing")
        images_bands.append((image_info["path"], image_info["bands"]))

    temp_dir = conf.paths.getpath("temp_dir")
    if temp_dir == "None":
        temp_dir = Path(tempfile.gettempdir())

    logger.info(f"Calculating timeseries for {len(images_bands)} images")
    zonal_stats_bulk.zonal_stats(
        vector_path=input_parcel_path,
        id_column=conf.columns["id"],
        rasters_bands=images_bands,
        output_dir=timeseries_periodic_dir,
        stats=stats,
        engine=engine,
        nb_parallel=nb_parallel,
    )
