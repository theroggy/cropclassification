from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import List

from cropclassification.helpers import config_helper as conf
from cropclassification.util import openeo_util
from cropclassification.util.openeo_util import ImageProfile
from cropclassification.util import zonal_stats_bulk

# First define/init some general variables/constants
# -------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

# The real work
# -------------------------------------------------------------


def calculate_periodic_timeseries(
    input_parcel_path: Path,
    start_date: datetime,
    end_date: datetime,
    sensordata_to_get: List[ImageProfile],
    dest_image_data_dir: Path,
    dest_data_dir: Path,
):
    """
    Calculate timeseries data for the input parcels.

    args
        data_to_get: an array with data you want to be calculated: check out the
            constants starting with DATA_TO_GET... for the options.
    """
    # As we want a weekly calculation, get nearest monday for start and stop day
    days_per_period = 7
    periodic_images_result = openeo_util.calc_periodic_mosaic(
        roi_path=input_parcel_path,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        output_dir=dest_image_data_dir,
        images_to_get=sensordata_to_get,
        force=False,
    )

    # Now calculate the timeseries
    images_bands = [
        (image_path, imageprofile.bands)
        for image_path, imageprofile in periodic_images_result
    ]
    temp_dir = conf.dirs.getpath("temp_dir")
    if temp_dir == "None":
        temp_dir = Path(tempfile.gettempdir())
    zonal_stats_bulk.zonal_stats(
        vector_path=input_parcel_path,
        id_column=conf.columns["id"],
        rasters_bands=images_bands,
        output_dir=dest_data_dir,
        stats=["count", "mean", "median", "std", "min", "max"],
        engine="pyqgis",
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
