from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import List

import geofileops as gfo

from cropclassification.helpers import config_helper as conf
from cropclassification.util import mosaic_util
from cropclassification.util import zonal_stats_bulk

# Get a logger...
logger = logging.getLogger(__name__)


def calculate_periodic_timeseries(
    input_parcel_path: Path,
    start_date: datetime,
    end_date: datetime,
    imageprofiles_to_get: List[str],
    dest_image_data_dir: Path,
    dest_data_dir: Path,
    nb_parallel: int,
):
    """
    Calculate timeseries data for the input parcels.

    args
        imageprofiles_to_get: an array with data you want to be calculated.
    """
    # As we want a weekly calculation, get nearest monday for start and stop day
    days_per_period = 7
    roi_info = gfo.get_layerinfo(input_parcel_path)

    periodic_images_result = mosaic_util.calc_periodic_mosaic(
        roi_bounds=roi_info.total_bounds,
        roi_crs=roi_info.crs,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        output_base_dir=dest_image_data_dir,
        imageprofiles_to_get=imageprofiles_to_get,
        imageprofiles=conf.image_profiles,
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

    """
    for image_path in periodic_images:
        output_path = (
            dest_data_dir / f"{input_parcel_path.stem}__{image_path.stem}.gpkg"
        )
        geoops_util.zonal_stats(
            input_vector_path=input_parcel_path,
            input_raster_path=image_path,
            output_path=output_path,
        )
    """
