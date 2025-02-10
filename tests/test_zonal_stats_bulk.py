import shutil

import geofileops as gfo
import pytest

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util import zonal_stats_bulk
from cropclassification.util.zonal_stats_bulk._zonal_stats_bulk_pyqgis import HAS_QGIS
from tests.test_helper import SampleData


@pytest.mark.parametrize(
    "engine, stats",
    [
        ("pyqgis", ["mean", "count"]),
        ("rasterstats", ["mean", "count"]),
        (
            "exactextract",
            [
                "mean(min_coverage_frac=0.5,coverage_weight=none)",
                "count(min_coverage_frac=0.5,coverage_weight=none)",
            ],
        ),
    ],
)
def test_zonal_stats_bulk(tmp_path, engine, stats):
    if engine == "pyqgis" and not HAS_QGIS:
        pytest.skip("QGIS is not available on this system.")

    # Prepare test data
    sample_dir = SampleData.marker_basedir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    # Read the configuration
    config_paths = [
        SampleData.config_dir / "cropgroup.ini",
        SampleData.tasks_dir / "local_overrule.ini",
    ]
    conf.read_config(
        config_paths=config_paths,
        default_basedir=SampleData.marker_basedir,
    )

    # Make sure the s2-agri input file was copied
    test_image_roi_dir = test_dir / SampleData.image_dir.name / SampleData.roi_name
    test_s1_asc_dir = test_image_roi_dir / SampleData.image_s1_asc_path.parent.name
    test_s1_desc_dir = test_image_roi_dir / SampleData.image_s1_desc_path.parent.name
    test_image_paths = list(test_s1_asc_dir.glob("*.tif"))
    test_image_paths.extend(test_s1_desc_dir.glob("*.tif"))
    images_bands = [(path, ["VV", "VH"]) for path in test_image_paths]
    vector_path = test_dir / SampleData.input_dir.name / "Prc_BEFL_2023_2023-07-24.gpkg"
    vector_info = gfo.get_layerinfo(vector_path)

    zonal_stats_bulk.zonal_stats(
        vector_path=vector_path,
        id_column="UID",
        rasters_bands=images_bands,
        output_dir=tmp_path,
        stats=stats,
        engine=engine,
    )

    result_paths = list(tmp_path.glob("*.sqlite"))
    assert len(result_paths) == 8
    for result_path in result_paths:
        result_df = pdh.read_file(result_path)
        # The result should have the same number of rows as the input vector file.
        assert len(result_df) == vector_info.featurecount
        # The calculates stats should not be nan for any row.
        assert not any(result_df["mean"].isna())
