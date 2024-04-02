import re
import shutil

import pytest
import rioxarray

from cropclassification.util import raster_util
from tests import test_helper


@pytest.mark.parametrize(
    "band_descriptions",
    [
        ["B02", "B03", "B04", "B08", "B11", "B12"],
        {6: "B12", 1: "B02", 2: "B03", 3: "B04", 4: "B08", 5: "B11"},
        {1: "B02", 2: "B03", 3: "B04", 4: "B08"},
    ],
)
def test_add_descriptions(tmp_path, band_descriptions):
    # Prepare and validate test file
    path = (
        test_helper.SampleDirs.image_dir
        / "roi_test/s2-agri"
        / "s2-agri_2024-03-04_2024-03-10_B02-B03-B04-B08-B11-B12_mean.tif"
    )
    test_path = tmp_path / path.name
    shutil.copy(path, test_path)
    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.add_band_descriptions(test_path, empty_band_descriptions)

    # Test
    raster_util.add_band_descriptions(test_path, band_descriptions=band_descriptions)

    # Validate results
    result_descriptions = {}
    with rioxarray.open_rasterio(test_path, cache=False) as image_file:
        image = image_file.to_dataset("band")
        assert "long_name" in image.attrs
        for idx, description in enumerate(image.attrs["long_name"]):
            result_descriptions[idx + 1] = description

    if isinstance(band_descriptions, dict):
        result_descriptions = {
            key: value
            for key, value in result_descriptions.items()
            if value is not None
        }
        assert band_descriptions == result_descriptions
    else:
        expected_descriptions = {
            idx + 1: value for idx, value in enumerate(band_descriptions)
        }
        assert expected_descriptions == result_descriptions


def test_add_descriptions_remove(tmp_path):
    # Prepare and validate test file
    path = (
        test_helper.SampleDirs.image_dir
        / "roi_test/s2-agri"
        / "s2-agri_2024-03-04_2024-03-10_B02-B03-B04-B08-B11-B12_mean.tif"
    )
    test_path = tmp_path / path.name
    shutil.copy(path, test_path)

    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.add_band_descriptions(test_path, empty_band_descriptions)

    # Validate that the input test file does not have descriptions.
    with rioxarray.open_rasterio(test_path, cache=False) as image_file:
        image = image_file.to_dataset("band")
        assert "long_name" not in image.attrs


@pytest.mark.parametrize(
    "band_descriptions, expected_error",
    [
        (["B02"], "number of bands (6) != number of band_descriptions (1)"),
        ([], "number of bands (6) != number of band_descriptions (0)"),
    ],
)
def test_add_descriptions_invalid(tmp_path, band_descriptions, expected_error):
    # Prepare and validate test file
    path = (
        test_helper.SampleDirs.image_dir
        / "roi_test/s2-agri"
        / "s2-agri_2024-03-04_2024-03-10_B02-B03-B04-B08-B11-B12_mean.tif"
    )
    test_path = tmp_path / path.name
    shutil.copy(path, test_path)
    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.add_band_descriptions(test_path, empty_band_descriptions)

    # Test
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        raster_util.add_band_descriptions(test_path, band_descriptions)
