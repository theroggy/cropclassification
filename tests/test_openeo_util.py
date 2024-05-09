from datetime import datetime
import shutil

import pytest

from cropclassification.util import openeo_util
from tests import test_helper


@pytest.mark.parametrize(
    "input_path, collection",
    [
        (test_helper.SampleData.image_s1_asc_path, "S1_GRD_SIGMA0_ASCENDING"),
        (test_helper.SampleData.image_s1_desc_path, "S1_GRD_SIGMA0_DESCENDING"),
    ],
)
def test_get_images_s1(tmp_path, input_path, collection):
    """
    Test creating an s1 image.

    Remark: the default way to run is with the result pre-copied and force=False, to
    avoid really calling the openeo API.
    """
    output_path = tmp_path / input_path.name
    shutil.copy(input_path, output_path)
    images_to_get = [
        {
            "path": output_path,
            "roi_bounds": test_helper.SampleData.roi_bounds,
            "roi_crs": test_helper.SampleData.roi_crs,
            "collection": collection,
            "start_date": test_helper.SampleData.start_date,
            "end_date": test_helper.SampleData.end_date,
            "bands": ["VV", "VH"],
            "time_reducer": "last",
            "max_cloud_cover": None,
            "process_options": {},
            "job_options": {},
        }
    ]

    openeo_util.get_images(images_to_get=images_to_get, force=False)
    assert output_path.exists()


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


def test_get_images_s2(tmp_path):
    """
    Test creating an s2 image.

    Remark: the default way to run is with the result pre-copied and force=False, to
    avoid really calling the openeo API.
    """
    test_helper.SampleData.image_s2_path
    output_path = tmp_path / test_helper.SampleData.image_s2_path.name
    shutil.copy(test_helper.SampleData.image_s2_path, output_path)
    images_to_get = [
        {
            "path": output_path,
            "roi_bounds": test_helper.SampleData.roi_bounds,
            "roi_crs": test_helper.SampleData.roi_crs,
            "collection": "TERRASCOPE_S2_TOC_V2",
            "start_date": test_helper.SampleData.start_date,
            "end_date": test_helper.SampleData.end_date,
            "bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
            "time_reducer": "mean",
            "max_cloud_cover": 80,
            "process_options": {},
            "job_options": {},
        }
    ]

    openeo_util.get_images(images_to_get=images_to_get, force=False)
    assert output_path.exists()
