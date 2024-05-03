from pathlib import Path
from typing import Dict

from cropclassification.util.mosaic_util import ImageProfile


testdata_dir = Path(__file__).resolve().parent / "data"


class SampleDirs:
    marker_basedir = Path(__file__).resolve().parent.parent / "sample_marker_basedir"
    tasks_dir = marker_basedir / "_tasks"
    config_dir = marker_basedir / "_config"
    image_dir = marker_basedir / "periodic_mosaic"
    task_path = tasks_dir / "task_test_calc_periodic_mosaic.ini"


IMAGEPROFILES: Dict[str, ImageProfile] = {
    "s2-agri": ImageProfile(
        name="s2-agri",
        satellite="s2",
        image_source="openeo",
        collection="TERRASCOPE_S2_TOC_V2",
        bands=["B02", "B03", "B04", "B08", "B11", "B12"],
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
