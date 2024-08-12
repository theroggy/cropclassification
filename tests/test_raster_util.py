import re
import shutil

import pytest
import rasterio
import rioxarray

from cropclassification.util import raster_util
from tests.test_helper import SampleData


@pytest.mark.parametrize("resampling", ["average", "bilinear"])
def test_add_overviews(tmp_path, resampling):
    # Prepare test file
    test_path = tmp_path / SampleData.image_s2_mean_path.name
    shutil.copy(SampleData.image_s2_mean_path, test_path)
    with rasterio.open(test_path) as file:
        for i in file.indexes:
            assert len(file.overviews(i)) == 0
        assert file.tags(ns="rio_overview").get("resampling") is None

    # Test
    raster_util.add_overviews(test_path, min_pixels=20, resampling=resampling)

    # Validate results
    with rasterio.open(test_path) as file:
        for i in file.indexes:
            assert len(file.overviews(i)) == 1
        assert file.tags(ns="rio_overview").get("resampling") == resampling


@pytest.mark.parametrize(
    "band_descriptions",
    [
        ["B02", "B03", "B04", "B08", "B11", "B12"],
        {6: "B12", 1: "B02", 2: "B03", 3: "B04", 4: "B08", 5: "B11"},
        {1: "B02", 2: "B03", 3: "B04", 4: "B08"},
    ],
)
def test_set_band_descriptions(tmp_path, band_descriptions):
    # Prepare test file
    test_path = tmp_path / SampleData.image_s2_mean_path.name
    shutil.copy(SampleData.image_s2_mean_path, test_path)
    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.set_band_descriptions(test_path, empty_band_descriptions)

    # Test
    raster_util.set_band_descriptions(test_path, band_descriptions=band_descriptions)

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


@pytest.mark.parametrize(
    "band_descriptions, expected_error",
    [
        (["B02"], "number of bands (6) != number of band_descriptions (1)"),
        ([], "number of bands (6) != number of band_descriptions (0)"),
        ("B02", "number of bands (6) != number of band_descriptions (1)"),
    ],
)
def test_set_band_descriptions_invalid(tmp_path, band_descriptions, expected_error):
    # Prepare and validate test file
    test_path = tmp_path / SampleData.image_s2_mean_path.name
    shutil.copy(SampleData.image_s2_mean_path, test_path)
    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.set_band_descriptions(test_path, empty_band_descriptions)

    # Test
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        raster_util.set_band_descriptions(test_path, band_descriptions)


def test_set_band_descriptions_overwrite_False(tmp_path):
    # Prepare and validate test file
    test_path = tmp_path / SampleData.image_s2_mean_path.name
    shutil.copy(SampleData.image_s2_mean_path, test_path)

    with rasterio.open(test_path, "r") as file:
        descriptions_orig = list(file.descriptions)

    # There are already descriptions, so overwrite should not happen.
    descriptions_new = [f"{desc}_new" for desc in descriptions_orig]
    raster_util.set_band_descriptions(test_path, descriptions_new, overwrite=False)
    with rasterio.open(test_path, "r") as file:
        descriptions_read = list(file.descriptions)
    assert descriptions_read == descriptions_orig

    # Set one description to None (with `overwrite=True`).
    descriptions_one_None = list(descriptions_orig)
    descriptions_one_None[3] = None
    raster_util.set_band_descriptions(test_path, descriptions_one_None, overwrite=True)
    with rasterio.open(test_path, "r") as file:
        descriptions_read = list(file.descriptions)
    assert descriptions_read == descriptions_one_None

    # As there is a None description, overwrite must happen even with `overwrite=False`.
    raster_util.set_band_descriptions(test_path, descriptions_new, overwrite=False)
    with rasterio.open(test_path, "r") as file:
        descriptions_read = list(file.descriptions)
    assert descriptions_read == descriptions_new


def test_set_band_descriptions_remove(tmp_path):
    # Prepare and validate test file
    test_path = tmp_path / SampleData.image_s2_mean_path.name
    shutil.copy(SampleData.image_s2_mean_path, test_path)

    # Remove the band descriptions
    empty_band_descriptions = [None, None, None, None, None, None]
    raster_util.set_band_descriptions(test_path, empty_band_descriptions)

    # Validate that the input test file does not have descriptions.
    with rioxarray.open_rasterio(test_path, cache=False) as image_file:
        image = image_file.to_dataset("band")
        assert "long_name" not in image.attrs


def test_set_no_data(tmp_path):
    # Prepare and validate test file
    test_path = tmp_path / SampleData.image_s2_mean_path.name
    shutil.copy(SampleData.image_s2_mean_path, test_path)

    # Set nodata value
    raster_util.set_no_data(test_path)

    # Validate result
    with rasterio.open(test_path) as file:
        assert file.nodata == 32767
