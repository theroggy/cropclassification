import shutil
from datetime import datetime

import geofileops as gfo
import pytest
from pandas.api.types import is_numeric_dtype

from cropclassification import cropclassification
from cropclassification.util.zonal_stats_bulk._zonal_stats_bulk_pyqgis import HAS_QGIS
from tests import test_helper


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
def test_task_calc_marker(tmp_path, balancing_strategy, cross_pred_models):
    if not HAS_QGIS:
        pytest.skip("QGIS is not available on this system.")

    marker_basedir = tmp_path / test_helper.SampleData.marker_basedir.name
    shutil.copytree(test_helper.SampleData.marker_basedir, marker_basedir)

    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = "task_test_calc_marker.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    cropclassification.cropclassification(
        tasksdir=tasks_dir,
        config_overrules=[
            f"marker.balancing_strategy={balancing_strategy}",
            f"classifier.cross_pred_models={cross_pred_models}",
            "preprocess.min_parcels_in_class=1",
        ],
    )

    today_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = marker_basedir / f"2024_CROPGROUP/Run_{today_str}_001"
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
        assert (run_dir / "model.hdf5").exists()
    else:
        for model_id in range(cross_pred_models):
            assert (run_dir / f"cross_pred_model_{model_id}").exists()


def test_task_calc_periodic_mosaic(tmp_path):
    marker_basedir = tmp_path / test_helper.SampleData.marker_basedir.name
    shutil.copytree(test_helper.SampleData.marker_basedir, marker_basedir)
    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = "task_test_calc_periodic_mosaic.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    cropclassification.cropclassification(tasksdir=tasks_dir)

    # Check if a log file was written
    log_dir = marker_basedir / "log"
    assert log_dir.exists()
    assert len(list(log_dir.glob("*.log"))) == 1
