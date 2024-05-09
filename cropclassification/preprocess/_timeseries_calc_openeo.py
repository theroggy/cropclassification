from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple

import geofileops as gfo
import pyproj
import shapely

from cropclassification.helpers import config_helper as conf
from cropclassification.util import mosaic_util
from cropclassification.util.mosaic_util import ImageProfile
from cropclassification.util import zonal_stats_bulk

# Get a logger...
logger = logging.getLogger(__name__)


def calculate_periodic_timeseries(
    input_parcel_path: Path,
    roi_bounds: Tuple[float, float, float, float],
    roi_crs: Optional[pyproj.CRS],
    start_date: datetime,
    end_date: datetime,
    period_name: str,
    imageprofiles_to_get: List[str],
    imageprofiles: Dict[str, ImageProfile],
    dest_image_data_dir: Path,
    dest_data_dir: Path,
    nb_parallel: int,
):
    """
    Calculate timeseries data for the input parcels.

    args
        imageprofiles_to_get: an array with data you want to be calculated.
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
        period_name=period_name,
        output_base_dir=dest_image_data_dir,
        imageprofiles_to_get=imageprofiles_to_get,
        imageprofiles=imageprofiles,
        force=False,
    )

    # Now calculate the timeseries
    images_bands = [
        (image_info["path"], image_info["bands"])
        for image_info in periodic_images_result
    ]
    temp_dir = conf.dirs.getpath("temp_dir")
    if temp_dir == "None":
        temp_dir = Path(tempfile.gettempdir())
    zonal_stats_bulk.zonal_stats(
        vector_path=input_parcel_path,
        id_column=conf.columns["id"],
        rasters_bands=images_bands,
        output_dir=dest_data_dir,
        stats=["count", "mean", "median", "std", "min", "max"],  # type: ignore[arg-type]
        engine="pyqgis",
        nb_parallel=nb_parallel,
    )
