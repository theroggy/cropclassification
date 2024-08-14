import os
import shutil
from datetime import datetime

import pytest
import rasterio

from cropclassification.util import openeo_util
from tests import test_helper
from tests.test_helper import SampleData


@pytest.mark.parametrize(
    "input_path, collection",
    [
        (SampleData.image_s1_asc_path, "S1_GRD_SIGMA0_ASCENDING"),
        (SampleData.image_s1_desc_path, "S1_GRD_SIGMA0_DESCENDING"),
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
            "roi_bounds": SampleData.roi_bounds,
            "roi_crs": SampleData.roi_crs,
            "collection": collection,
            "start_date": SampleData.start_date,
            "end_date": SampleData.end_date,
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


@pytest.mark.parametrize("time_reducer", ["mean", "best"])
def test_get_images_s2(tmp_path, time_reducer):
    """
    Test creating an s2 image.

    Remark: the default way to run is with the result pre-copied and force=False, to
    avoid really calling the openeo API.
    """
    # Prepare test data
    if time_reducer == "best":
        output_path = tmp_path / SampleData.image_s2_best_path.name
        shutil.copy(SampleData.image_s2_best_path, output_path)
    elif time_reducer == "mean":
        output_path = tmp_path / SampleData.image_s2_mean_path.name
        shutil.copy(SampleData.image_s2_mean_path, output_path)
    else:
        raise ValueError(f"Unknown time_reducer {time_reducer}")

    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    images_to_get = [
        {
            "path": output_path,
            "roi_bounds": SampleData.roi_bounds,
            "roi_crs": SampleData.roi_crs,
            "collection": "TERRASCOPE_S2_TOC_V2",
            "satellite": "s2",
            "start_date": SampleData.start_date,
            "end_date": SampleData.end_date,
            "bands": bands,
            "time_reducer": time_reducer,
            "max_cloud_cover": 80,
            "process_options": {},
            "job_options": {},
        }
    ]
    openeo_util.get_images(images_to_get=images_to_get, force=False)

    # Check result
    assert output_path.exists()
    with rasterio.open(output_path, "r") as file:
        assert file.count == len(bands)
        assert file.profile["dtype"] == "uint16"
        assert file.profile["nodata"] is not None
        # All bands should have a description
        assert all(file.descriptions)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ,
    reason="Don't run on CI: not possible to connect to openeo.",
)
@pytest.mark.parametrize(
    "expected_error, satellite, time_reducer",
    [("only Sentinel 2 can be used with time_reducer = best", "s1", "best")],
)
def test_get_images_invalid_params(
    tmp_path, expected_error: str, satellite: str, time_reducer: str
):
    """
    Test creating an s2 image.

    Remark: the default way to run is with the result pre-copied and force=False, to
    avoid really calling the openeo API.
    """
    # Prepare test data
    output_path = tmp_path / SampleData.image_s2_best_path.name
    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    images_to_get = [
        {
            "path": output_path,
            "roi_bounds": SampleData.roi_bounds,
            "roi_crs": SampleData.roi_crs,
            "collection": "TERRASCOPE_S2_TOC_V2",
            "satellite": satellite,
            "start_date": SampleData.start_date,
            "end_date": SampleData.end_date,
            "bands": bands,
            "time_reducer": time_reducer,
            "max_cloud_cover": 80,
            "process_options": {},
            "job_options": {},
        }
    ]

    # Test
    with pytest.raises(ValueError, match=expected_error):
        openeo_util.get_images(images_to_get=images_to_get)
