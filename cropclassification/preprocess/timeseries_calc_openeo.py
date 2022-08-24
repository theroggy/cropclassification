from datetime import datetime
import logging
from pathlib import Path
from typing import List

from cropclassification.util import openeo_util

# -------------------------------------------------------------
# First define/init some general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def calc_timeseries_data(
    input_parcel_path: Path,
    input_country_code: str,
    start_date: datetime,
    end_date: datetime,
    sensordata_to_get: List[str],
    base_filename: str,
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
    openeo_util.calc_periodic_images(
        roi_path=input_parcel_path,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        filename_suffix=filename_suffix,
        output_dir=dest_data_dir,
        sensordata_to_get=sensordata_to_get,
        force=False,
    )
