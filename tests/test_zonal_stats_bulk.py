import shutil

import geofileops as gfo
import pytest

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util import zonal_stats_bulk
from cropclassification.util.zonal_stats_bulk import _zonal_stats_bulk_exactextract
from cropclassification.util.zonal_stats_bulk._zonal_stats_bulk_pyqgis import HAS_QGIS
from tests.test_helper import SampleData


@pytest.mark.parametrize(
    "engine, stats",
    [
        ("pyqgis", ["mean", "count", "std"]),
        ("rasterstats", ["mean", "count", "std"]),
        (
            "exactextract",
            [
                "mean(min_coverage_frac=0.8,coverage_weight=none)",
                "count(min_coverage_frac=0.8,coverage_weight=none)",
                "stdev(min_coverage_frac=0.8,coverage_weight=none)",
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "bands, exp_results_path",
    [
        (["vvdvh"], 2),
        (["VV", "VH"], 8),
        (["B02", "B03", "B04", "B08", "B11", "B12"], 18),
    ],
)
def test_zonal_stats_bulk(tmp_path, engine, stats, bands, exp_results_path):
    if engine == "pyqgis" and not HAS_QGIS:
        pytest.skip("QGIS is not available on this system.")

    # Prepare test data
    sample_dir = SampleData.markers_dir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    # Read the configuration
    config_paths = [
        SampleData.config_dir / "cropgroup.ini",
        SampleData.tasks_dir / "local_overrule.ini",
    ]
    conf.read_config(
        config_paths=config_paths,
        default_basedir=sample_dir,
    )

    # Make sure the s2-agri input file was copied
    test_image_roi_dir = test_dir / SampleData.image_dir.name / SampleData.roi_name
    if bands == ["VV", "VH"]:
        image1_dir = test_image_roi_dir / SampleData.image_s1_asc_path.parent.name
        image2_dir = test_image_roi_dir / SampleData.image_s1_desc_path.parent.name
    elif bands == ["vvdvh"]:
        image1_dir = (
            test_image_roi_dir
            / SampleData.image_s1_grd_vvdvh_asc_weekly_path.parent.name
        )
        image2_dir = (
            test_image_roi_dir
            / SampleData.image_s1_grd_vvdvh_desc_weekly_path.parent.name
        )
    elif bands == ["B02", "B03", "B04", "B08", "B11", "B12"]:
        image1_dir = test_image_roi_dir / SampleData.image_s2_mean_path.parent.name
        image2_dir = None
    test_image_paths = list(image1_dir.glob("*.tif"))
    if image2_dir:
        test_image_paths.extend(image2_dir.glob("*.tif"))
    images_bands = [(path, bands) for path in test_image_paths]
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
    assert len(result_paths) == exp_results_path
    for result_path in result_paths:
        result_df = pdh.read_file(result_path)
        # The result should have the same number of rows as the input vector file.
        assert len(result_df) == vector_info.featurecount
        # The calculates stats should not be nan for any row.
        assert not any(result_df["mean"].isna())


def test_zonal_stats_bulk_invalid(tmp_path):
    # Prepare test data
    sample_dir = SampleData.markers_dir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    vector_path = test_dir / SampleData.input_dir.name / "Prc_BEFL_2023_2023-07-24.gpkg"

    with pytest.raises(ValueError, match="Error calculating zonal stats"):
        _zonal_stats_bulk_exactextract.zonal_stats_band(
            vector_path=vector_path,
            raster_path=test_dir / SampleData.image_s1_asc_path,
            tmp_dir=tmp_path,
            stats=["means"],
            include_cols=["index", "UID", "x_ref"],
        )
