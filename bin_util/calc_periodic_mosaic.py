import logging
from datetime import datetime
from pathlib import Path

import cropclassification.helpers.config_helper as conf
from cropclassification.util import mosaic_util


def main():
    logging.basicConfig(level=logging.INFO)

    # Init some variables
    roi_crs = 31370
    # BEFL
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2023, 12, 31)
    roi_bounds = [20_000, 150_000, 260_000, 245_000]
    images_periodic_dir = Path("//dg3.be/alp/Datagis/satellite_periodic/BEFL")

    # roi_test
    """
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2023, 10, 13)
    roi_bounds = (161_400, 188_000, 161_900, 188_500)
    images_periodic_dir = Path("c:/temp/periodic_mosaic/roi_test")
    """

    imageprofiles_to_get = ["s1-rvi-asc-weekly", "s1-rvi-desc-weekly"]
    # imageprofiles_to_get = ["s1-dprvi-asc-weekly", "s1-dprvi-desc-weekly"]
    # imageprofiles_to_get = ["s2-agri-weekly"]

    image_profiles_path = (
        Path(__file__).resolve().parent.parent
        / "sample_marker_basedir/_config/image_profiles.ini"
    )
    imageprofiles = conf._get_image_profiles(image_profiles_path)

    _ = mosaic_util.calc_periodic_mosaic(
        roi_bounds=roi_bounds,
        roi_crs=roi_crs,
        start_date=start_date,
        end_date=end_date,
        output_base_dir=images_periodic_dir,
        imageprofiles_to_get=imageprofiles_to_get,
        imageprofiles=imageprofiles,
        force=False,
    )


if __name__ == "__main__":
    main()
