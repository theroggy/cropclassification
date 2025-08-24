import shutil

import geofileops as gfo
import geopandas as gpd
import pytest
import shapely

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


def test_exactextract_invalid_stats(tmp_path):
    # Prepare test data
    sample_dir = SampleData.markers_dir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    vector_path = test_dir / SampleData.input_dir.name / "Prc_BEFL_2023_2023-07-24.gpkg"

    with pytest.raises(ValueError, match="Unsupported stat: means"):
        _zonal_stats_bulk_exactextract.zonal_stats_band(
            vector_path=vector_path,
            raster_path=test_dir / SampleData.image_s1_asc_path,
            tmp_dir=tmp_path,
            stats=["means"],
            include_cols=["index", "UID", "x_ref"],
        )


@pytest.mark.parametrize(
    "min_coverage_frac, exp_count",
    [
        (0.0, [49, 42, 42, 42]),
        (0.2, [25, 30, 30, 30]),
        (0.5, [25, 30, 30, 20]),
        (0.8, [25, 20, 30, 20]),
        (1.0, [25, 20, 20, 20]),
    ],
)
def test_exactextract_min_coverage_frac(tmp_path, min_coverage_frac, exp_count):
    # Prepare test data
    sample_dir = SampleData.markers_dir
    test_dir = tmp_path / sample_dir.name
    shutil.copytree(sample_dir, test_dir)

    geoms = [
        (
            "POLYGON ((161405.210003 188506.174464, 161456.432668 188505.089414,"
            " 161455.476062 188453.597478, 161403.717327 188454.559035, 161405.210003"
            " 188506.174464))"
        ),
        (
            "POLYGON ((161495.16002 188499.432483, 161546.38515 188498.480826,"
            " 161545.277792 188446.05759, 161493.919267 188447.011718, 161495.16002"
            " 188499.432483))"
        ),
        (
            "POLYGON ((161433.705572 188423.843391, 161484.794828 188422.760828,"
            " 161483.986074 188364.860805, 161432.368184 188366.22007, 161433.705572"
            " 188423.843391))"
        ),
        (
            "POLYGON ((161513.51231 188417.022976, 161565.006707 188416.199771,"
            " 161563.840308 188367.780984, 161512.222428 188369.140255, 161513.51231"
            " 188417.022976))"
        ),
    ]

    box_geoms = shapely.from_wkt(geoms)

    gdf_geoms = gpd.GeoDataFrame(geometry=box_geoms, crs="EPSG:31370")

    # write the geofile to a file
    vector_path = tmp_path / "vector.gpkg"
    gfo.to_file(gdf=gdf_geoms, path=vector_path)
    expression = """
        CASE
            WHEN fid = 1
                THEN -0.2
            WHEN fid = 2
                THEN 0.5
            WHEN fid = 3
                THEN 0.8
            WHEN fid = 4
                THEN 0.2
            ELSE NULL
        END
    """
    gfo.add_column(path=vector_path, name="UID", type="TEXT", expression=expression)

    test_image_roi_dir = test_dir / SampleData.image_dir.name / SampleData.roi_name
    image1_dir = test_image_roi_dir / SampleData.image_s1_asc_path.parent.name
    image2_dir = test_image_roi_dir / SampleData.image_s1_desc_path.parent.name
    test_image_paths = list(image1_dir.glob("*.tif"))
    if image2_dir:
        test_image_paths.extend(image2_dir.glob("*.tif"))
    bands = ["VV", "VH"]
    images_bands = [(path, bands) for path in test_image_paths]
    output_path = tmp_path / f"min_cov_frac_{min_coverage_frac}"
    stats = [f"count(min_coverage_frac={min_coverage_frac},coverage_weight=none)"]

    zonal_stats_bulk.zonal_stats(
        vector_path=vector_path,
        id_column="UID",
        rasters_bands=images_bands,
        output_dir=output_path,
        stats=stats,
        engine="exactextract",
    )

    result_paths = list(output_path.glob("*.sqlite"))
    result_df = pdh.read_file(result_paths[0])
    for index in range(4):
        count = result_df[result_df["index"] == index]["count"].values[0]
        assert count == exp_count[index]
