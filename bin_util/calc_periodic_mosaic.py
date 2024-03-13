from datetime import datetime
import logging
from pathlib import Path

import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess._timeseries_helper as ts_helper
from cropclassification.util import openeo_util


def main():
    logging.basicConfig(level=logging.INFO)

    # Init some variables
    days_per_period = 7
    start_date = datetime(2022, 2, 25)
    end_date = datetime(2022, 6, 5)
    dest_image_data_dir = Path("//dg3.be/alp/Datagis/satellite_periodic/BEFL")
    sensordata_to_get = [
        conf._get_image_profiles(
            Path("x:/monitoring/markers/dev/_config/image_profiles.ini")
        )["s2-agri"]
    ]

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_helper.get_monday(start_date)
    end_date = ts_helper.get_monday(end_date)

    _ = openeo_util.calc_periodic_mosaic(
        roi_bounds=[20_000, 150_000, 260_000, 245_000],
        roi_crs=31370,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        output_dir=dest_image_data_dir,
        images_to_get=sensordata_to_get,
        force=False,
    )


if __name__ == "__main__":
    main()
