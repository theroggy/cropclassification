import shutil
from tests import test_helper

from cropclassification import cropclassification


def test_task_calc_periodic_mosaic(tmp_path):
    projects_dir = tmp_path / test_helper.sampleprojects_dir.name
    shutil.copytree(test_helper.sampleprojects_dir, projects_dir)

    # Create configparser and read task file!
    tasks_dir = projects_dir / "_tasks"
    cropclassification.cropclassification(tasksdir=tasks_dir)

    # Check if a log file was written
    log_dir = projects_dir / "log"
    assert log_dir.exists()
    assert len(list(log_dir.glob("*.log"))) == 1
