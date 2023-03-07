from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cropclassification import taskrunner


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def test_taskrunner():
    taskrunner.run_tasks(get_testdata_dir() / "_tasks")
