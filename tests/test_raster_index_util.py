import pytest

from cropclassification.util import raster_index_util
from tests import test_helper


@pytest.mark.parametrize("index", ["dprvi"])
@pytest.mark.parametrize("save_as_byte", [True, False])
def test_calc_index_s1(tmp_path, index, save_as_byte):
    # Prepare test data
    input_path = (
        test_helper.SampleDirs.image_dir
        / "roi_test/s1-grd-sigma0-asc"
        / "s1-grd-sigma0-asc_2024-03-04_2024-03-10_VV-VH_mean.tif"
    )
    output_path = tmp_path / f"{input_path.stem}_{index}_{save_as_byte}.tif"

    # Prepare parameters
    assert not output_path.exists()
    raster_index_util.calc_index(
        input_path=input_path, output_path=output_path, index=index
    )
    assert output_path.exists()


@pytest.mark.parametrize(
    "index, save_as_byte", [("ndvi", True), ("ndvi", False), ("bsi", False)]
)
def test_calc_index_s2(tmp_path, index, save_as_byte):
    # Prepare test data
    input_path = (
        test_helper.SampleDirs.image_dir
        / "roi_test/s2-agri"
        / "s2-agri_2024-03-04_2024-03-10_B02-B03-B04-B08-B11-B12_mean.tif"
    )
    output_path = tmp_path / f"{input_path.stem}_{index}_{save_as_byte}.tif"

    # Prepare parameters
    assert not output_path.exists()
    raster_index_util.calc_index(
        input_path=input_path,
        output_path=output_path,
        index=index,
        save_as_byte=save_as_byte,
    )
    assert output_path.exists()
