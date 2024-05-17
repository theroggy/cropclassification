import shutil
from datetime import datetime

import geofileops as gfo
import pytest
from pandas.api.types import is_numeric_dtype

from cropclassification import cropclassification
from tests import test_helper


@pytest.mark.parametrize(
    "task, balancing_strategy",
    [
        ("calc_periodic_mosaic", None),
        ("calc_marker", "BALANCING_STRATEGY_MEDIUM"),
        ("calc_marker", "BALANCING_STRATEGY_MEDIUM2"),
        ("calc_marker", "BALANCING_STRATEGY_PROPORTIONAL_GROUPS"),
        ("calc_marker", "BALANCING_STRATEGY_UPPER_LIMIT"),
        ("calc_marker", "BALANCING_STRATEGY_EQUAL"),
    ],
)
def test_end2end_task(tmp_path, task, balancing_strategy):
    marker_basedir = tmp_path / test_helper.SampleData.marker_basedir.name
    shutil.copytree(test_helper.SampleData.marker_basedir, marker_basedir)

    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = f"task_test_{task}.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    cropclassification.cropclassification(
        tasksdir=tasks_dir,
        config_overrules=[
            f"marker.balancing_strategy={balancing_strategy}",
        ],
    )

    if task == "calc_marker":
        run_dir = (
            marker_basedir
            / "2024_CROPGROUP"
            / f"Run_{datetime.now().strftime('%Y-%m-%d')}_001"
        )
        assert run_dir.exists()
        assert (run_dir / "CROPGROUP_01_mlp.hdf5").exists()
        assert (
            run_dir / "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_all.gpkg"
        ).exists()
        assert (
            run_dir
            / "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_all.sqlite_accuracy_report.html"  # noqa: E501
        ).exists()
        assert (
            run_dir
            / "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_test.sqlite_accuracy_report.txt_groundtruth_pred_quality_details.gpkg"  # noqa: E501
        )
        df_predict = gfo.read_file(
            path=run_dir
            / "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_test.sqlite_accuracy_report.txt_groundtruth_pred_quality_details.gpkg"  # noqa: E501
        )
        assert is_numeric_dtype(df_predict["LAYER_ID"].dtype)
        assert is_numeric_dtype(df_predict["PRC_ID"].dtype)
        assert is_numeric_dtype(df_predict["pixcount"].dtype)
        assert is_numeric_dtype(df_predict["pred1_prob"].dtype)
        assert is_numeric_dtype(df_predict["pred2_prob"].dtype)
        assert is_numeric_dtype(df_predict["pred3_prob"].dtype)
        assert (
            run_dir
            / "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_all.sqlite_accuracy_report.txt_groundtruth_pred_quality_details.gpkg"  # noqa: E501
        )
        df_predict = gfo.read_file(
            path=run_dir
            / "Prc_BEFL_2023_2023-07-24_bufm5_weekly_predict_all.sqlite_accuracy_report.txt_groundtruth_pred_quality_details.gpkg"  # noqa: E501
        )
        assert is_numeric_dtype(df_predict["LAYER_ID"].dtype)
        assert is_numeric_dtype(df_predict["PRC_ID"].dtype)
        assert is_numeric_dtype(df_predict["pixcount"].dtype)
        assert is_numeric_dtype(df_predict["pred1_prob"].dtype)
        assert is_numeric_dtype(df_predict["pred2_prob"].dtype)
        assert is_numeric_dtype(df_predict["pred3_prob"].dtype)

    if task == "calc_periodic_mosaic":
        # Check if a log file was written
        log_dir = marker_basedir / "log"
        assert log_dir.exists()
        assert len(list(log_dir.glob("*.log"))) == 1
