from datetime import datetime
import shutil

from cropclassification.util import openeo_util
from tests import test_helper


def test_get_images_s1_edge_first(tmp_path):
    """
    Test result of using the "first" reducer on the edge of two images.

    The s1 mosaic is created at a roi + time + orbit so we are at the border of two
    images. The question is what firsts does with the nodata pixels at the border of
    the images.

    Remark: to avoid really using openeo, the test doesn't really do anything by default
    as force=False! The file "s1-on-border-first.tif" is the result though of running
    it once upon a time.
    """
    test_filename = "s1-on-border-first.tif"
    output_path = tmp_path / test_filename
    shutil.copy(test_helper.testdata_dir / test_filename, output_path)
    images_to_get = [
        {
            "path": output_path,
            "roi_bounds": (557_000, 5_655_000, 558_000, 5_656_000),
            "roi_crs": 32631,
            "collection": "S1_GRD_SIGMA0_DESCENDING",
            "start_date": datetime(2023, 7, 17),
            "end_date": datetime(2023, 7, 24),
            "bands": ["VV"],
            "time_reducer": "first",
            "max_cloud_cover": None,
            "process_options": {},
            "job_options": {},
        }
    ]

    openeo_util.get_images(images_to_get=images_to_get, force=False)
    assert output_path.exists()
