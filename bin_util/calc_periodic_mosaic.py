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
    dest_image_data_dir = Path("c:/temp/periodic_mosaic/roi_test")
    image_profiles_path = (
        Path(__file__).resolve().parent.parent
        / "sample_marker_basedir/_config/image_profiles.ini"
    )
    imageprofiles = conf._get_image_profiles(image_profiles_path)

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_helper.get_monday(start_date)
    end_date = ts_helper.get_monday(end_date)

    _ = mosaic_util.calc_periodic_mosaic(
        # roi_bounds=[20_000, 150_000, 260_000, 245_000],
        roi_bounds=[160_000, 188_000, 160_500, 188_500],
        roi_crs=31370,
        start_date=start_date,
        end_date=end_date,
        days_per_period=days_per_period,
        time_dimension_reducer="mean",
        output_base_dir=dest_image_data_dir,
        imageprofiles_to_get=["s2-agri", "s2-ndvi"],
        imageprofiles=imageprofiles,
        force=False,
    )


if __name__ == "__main__":
    main()
