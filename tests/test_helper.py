from datetime import datetime
from pathlib import Path

from cropclassification.util.mosaic_util import ImageProfile


testdata_dir = Path(__file__).resolve().parent / "data"


class SampleData:
    marker_basedir = Path(__file__).resolve().parent.parent / "sample_marker_basedir"
    tasks_dir = marker_basedir / "_tasks"
    config_dir = marker_basedir / "_config"
    inputdata_dir = marker_basedir / "_inputdata"
    refe_dir = marker_basedir / "_refe"
    image_dir = marker_basedir / "_images_periodic"
    input_dir = marker_basedir / "_inputdata"
    roi_name = "roi_test"
    image_roi_dir = image_dir / roi_name

    image_s1_asc_path = (
        image_roi_dir
        / "s1-grd-sigma0-asc-weekly"
        / "s1-grd-sigma0-asc-weekly_2024-03-04_2024-03-10_VV-VH_last.tif"
    )
    image_s1_desc_path = (
        image_roi_dir
        / "s1-grd-sigma0-desc-weekly"
        / "s1-grd-sigma0-desc-weekly_2024-03-04_2024-03-10_VV-VH_last.tif"
    )
    image_s2_path = (
        image_roi_dir
        / "s2-agri-weekly"
        / "s2-agri-weekly_2024-03-04_2024-03-10_B02-B03-B04-B08-B11-B12_mean.tif"
    )

    start_date = datetime(2024, 3, 4)
    end_date = datetime(2024, 3, 11)
    end_date_incl = datetime(2024, 3, 10)
    roi_bounds = (161_400.0, 188_000.0, 161_900.0, 188_500.0)
    roi_crs = 31370


IMAGEPROFILES: dict[str, ImageProfile] = {
    "s2-agri": ImageProfile(
        name="s2-agri",
        satellite="s2",
        image_source="openeo",
        collection="TERRASCOPE_S2_TOC_V2",
        bands=["B02", "B03", "B04", "B08", "B11", "B12"],
        period_name="weekly",
        max_cloud_cover=80,
    ),
    "s2-ndvi": ImageProfile(
        name="s2-ndvi",
        satellite="s2",
        image_source="local",
        index_type="ndvi",
        bands=["ndvi"],
        base_image_profile="s2-agri",
    ),
}
