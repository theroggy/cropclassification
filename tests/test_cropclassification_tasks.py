import shutil

from cropclassification import cropclassification
from tests.test_helper import SampleData


def test_task_calc_periodic_mosaic(tmp_path):
    marker_basedir = tmp_path / SampleData.marker_basedir.name
    shutil.copytree(SampleData.marker_basedir, marker_basedir)

    # Create configparser and read task file!
    tasks_dir = marker_basedir / "_tasks"
    cropclassification.cropclassification(tasksdir=tasks_dir)

    # Check if a log file was written
    log_dir = marker_basedir / "log"
    assert log_dir.exists()
    assert len(list(log_dir.glob("*.log"))) == 1
