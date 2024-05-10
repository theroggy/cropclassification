import shutil

import geofileops as gfo
import pytest

from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util import zonal_stats_bulk
from tests.test_helper import SampleData


@pytest.mark.parametrize("engine", ["pyqgis", "rasterstats"])
def test_zonal_stats_bulk(tmp_path, engine):
    # Prepare test data
    sample_dir = SampleData.marker_basedir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    image_dir = test_dir / "periodic_mosaic/roi_test"
    input_image_paths = list((image_dir / "s1-grd-sigma0-asc").glob("*.tif"))
    input_image_paths.extend((image_dir / "s1-grd-sigma0-desc").glob("*.tif"))
    images_bands = [(path, ["VV", "VH"]) for path in input_image_paths]
    vector_path = test_dir / SampleData.input_dir.name / "Prc_BEFL_2023_2023-07-24.gpkg"
    vector_info = gfo.get_layerinfo(vector_path)

    zonal_stats_bulk.zonal_stats(
        vector_path=vector_path,
        id_column="UID",
        rasters_bands=images_bands,
        output_dir=tmp_path,
        stats=["mean"],
        engine=engine,
    )

    result_paths = list(tmp_path.glob("*.sqlite"))
    assert len(result_paths) == 4
    for result_path in result_paths:
        result_df = pdh.read_file(result_path)
        # The result should have the same number of rows as the input vector file.
        assert len(result_df) == vector_info.featurecount
        # The calculates stats should not be nan for any row.
        assert not any(result_df["mean"].isna())
