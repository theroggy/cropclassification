import shutil
from datetime import datetime

import geofileops as gfo
import pytest
from pandas.api.types import is_numeric_dtype

from cropclassification import taskrunner
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.util.zonal_stats_bulk._zonal_stats_bulk_pyqgis import HAS_QGIS
from tests import test_helper


@pytest.mark.parametrize(
    "markertype, cover_periodic_dir_suffix, exp_nb_parcels",
    [
        ("COVER", "", 10),
        ("COVER_TBG_BMG_VOORJAAR", "_TBG_BMG", 3),
        ("COVER_TBG_BMG_NAJAAR", "_TBG_BMG", 3),
        ("COVER_EEB_VOORJAAR", "_EEB", 1),
        ("COVER_EEF_VOORJAAR", "_EEF", 0),
        ("COVER_BMG_MEG_MEV_NAJAAR", "_BMG_MEG_MEV", 2),
    ],
)
def test_cover(tmp_path, markertype, cover_periodic_dir_suffix, exp_nb_parcels):
    """Test running the cover task with all COVER marker types."""
    if not HAS_QGIS:
        pytest.skip("QGIS is needed for timeseries calculation, but is not available.")

    markers_dir = tmp_path / test_helper.SampleData.markers_dir.name
    shutil.copytree(test_helper.SampleData.markers_dir, markers_dir)

    # Create configparser and read task file!
    tasks_dir = markers_dir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = "task_test_calc_cover.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    taskrunner.run_tasks(
        tasksdir=tasks_dir, config_overrules=[f"marker.markertype={markertype}"]
    )

    cover_periodic_dir = markers_dir / "_cover_periodic"
    parcels_dir = f"Prc_BEFL_2023_2023-07-24{cover_periodic_dir_suffix}"
    cover_periodic_parcels_dir = cover_periodic_dir / parcels_dir
    assert cover_periodic_parcels_dir.exists()

    # Check the result
    output_stems = ["cover_2024-03-04_2024-03-11", "cover_2024-03-11_2024-03-18"]
    for output_stem in output_stems:
        output_path = cover_periodic_parcels_dir / f"{output_stem}.sqlite"
        assert output_path.exists()

        df = pdh.read_file(output_path)
        assert len(df) == exp_nb_parcels


@pytest.mark.parametrize(
    "balancing_strategy, cross_pred_models",
    [
        ("BALANCING_STRATEGY_MEDIUM", 0),
        ("BALANCING_STRATEGY_MEDIUM2", 0),
        ("BALANCING_STRATEGY_PROPORTIONAL_GROUPS", 2),
        ("BALANCING_STRATEGY_UPPER_LIMIT", 0),
        ("BALANCING_STRATEGY_EQUAL", 0),
    ],
)
def test_cropclass(tmp_path, balancing_strategy, cross_pred_models):
    """Test running the calc_cropclass task for marker CROPGROUP.

    The different balancing strategies are tested as well.
    """
    if not HAS_QGIS:
        pytest.skip("QGIS is needed for timeseries calculation, but is not available.")

    markers_dir = tmp_path / test_helper.SampleData.markers_dir.name
    shutil.copytree(test_helper.SampleData.markers_dir, markers_dir)

    # Create configparser and read task file!
    tasks_dir = markers_dir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = "task_test_calc_cropclass.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    taskrunner.run_tasks(
        tasksdir=tasks_dir,
        config_overrules=[
            f"marker.balancing_strategy={balancing_strategy}",
            f"classifier.cross_pred_models={cross_pred_models}",
            "preprocess.min_parcels_in_class=1",
        ],
    )

    today_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = markers_dir / f"2024_CROPGROUP/Run_{today_str}_001"
    assert run_dir.exists()
    base_stem = "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_all"
    assert (run_dir / f"{base_stem}.gpkg").exists()
    assert (run_dir / f"{base_stem}.sqlite_accuracy_report.html").exists()
    details_gpgk_path = (
        run_dir
        / f"{base_stem}.sqlite_accuracy_report.txt_groundtruth_pred_quality_details.gpkg"  # noqa: E501
    )
    assert details_gpgk_path.exists()

    df_predict = gfo.read_file(path=details_gpgk_path)
    assert is_numeric_dtype(df_predict["LAYER_ID"].dtype)
    assert is_numeric_dtype(df_predict["PRC_ID"].dtype)
    assert is_numeric_dtype(df_predict["pixcount"].dtype)
    assert is_numeric_dtype(df_predict["pred1_prob"].dtype)
    assert is_numeric_dtype(df_predict["pred2_prob"].dtype)
    assert is_numeric_dtype(df_predict["pred3_prob"].dtype)

    if cross_pred_models <= 1:
        assert (run_dir / "marker_01_randomforest.hdf5").exists()
    else:
        for model_id in range(cross_pred_models):
            assert (run_dir / f"cross_pred_model_{model_id}").exists()


def test_task_calc_periodic_mosaic(tmp_path):
    markers_dir = tmp_path / test_helper.SampleData.markers_dir.name
    shutil.copytree(test_helper.SampleData.markers_dir, markers_dir)
    # Create configparser and read task file!
    tasks_dir = markers_dir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = "task_test_calc_periodic_mosaic.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    taskrunner.run_tasks(tasksdir=tasks_dir)

    # Check if a log file was written
    log_dir = markers_dir / "log"
    assert log_dir.exists()
    assert len(list(log_dir.glob("*.log"))) == 1
