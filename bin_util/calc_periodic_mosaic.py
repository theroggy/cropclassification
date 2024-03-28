from datetime import datetime
import logging
from pathlib import Path

import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess._timeseries_helper as ts_helper
from cropclassification.util import mosaic_util


def main():
    logging.basicConfig(level=logging.INFO)

    # Init some variables
    days_per_period = 7
    start_date = datetime(2024, 3, 4)
    end_date = datetime(2024, 3, 11)
    dest_image_data_dir = Path("c:/temp/test")
    image_profiles_path = (
        Path(__file__).resolve().parent.parent / "tests/data/image_profiles.ini"
    )
    image_profiles = conf._get_image_profiles(image_profiles_path)
    sensordata_to_get = [image_profiles[profile] for profile in ["s2-agri", "s2-ndvi"]]

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_helper.get_monday(start_date)
    end_date = ts_helper.get_monday(end_date)

    _ = mosaic_util.calc_periodic_mosaic(
        # roi_bounds=[20_000, 150_000, 260_000, 245_000],
        roi_bounds=[161_000, 188_000, 162_000, 189_000],
        roi_crs=31370,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        output_base_dir=dest_image_data_dir,
        images_to_get=sensordata_to_get,
        force=False,
    )


if __name__ == "__main__":
    main()
