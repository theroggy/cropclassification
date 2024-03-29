import shutil

from cropclassification import cropclassification
from tests import test_helper


def test_task_calc_periodic_mosaic(tmp_path):
    marker_basedir = tmp_path / test_helper.SampleDirs.marker_basedir.name
    shutil.copytree(test_helper.SampleDirs.marker_basedir, marker_basedir)

    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    cropclassification.cropclassification(tasksdir=tasks_dir)

    # Check if a log file was written
    log_dir = marker_basedir / "log"
    assert log_dir.exists()
    assert len(list(log_dir.glob("*.log"))) == 1
