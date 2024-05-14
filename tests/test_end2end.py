from datetime import datetime
import shutil

import pytest
from cropclassification import cropclassification
from tests import test_helper


@pytest.mark.parametrize(
    "task",
    [
        "calc_marker",
        "calc_periodic_mosaic",
        "calc_timeseries",
    ],
)
def test_end2end_task(tmp_path, task):
    marker_basedir = tmp_path / test_helper.SampleData.marker_basedir.name
    shutil.copytree(test_helper.SampleData.marker_basedir, marker_basedir)

    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    ignore_dir = tasks_dir / "ignore"
    task_ini = f"task_test_{task}.ini"

    shutil.copy(src=ignore_dir / task_ini, dst=tasks_dir / task_ini)

    cropclassification.cropclassification(tasksdir=tasks_dir)

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
    if task == "calc_periodic_mosaic":
        # Check if a log file was written
        log_dir = marker_basedir / "log"
        assert log_dir.exists()
        assert len(list(log_dir.glob("*.log"))) == 1
    if task == "calc_timeseries":
        # Check if a log file was written
        log_dir = marker_basedir / "log"
        assert log_dir.exists()
