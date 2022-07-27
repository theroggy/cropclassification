from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cropclassification


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def test_():
    cropclassification.cropclassification(get_testdata_dir() / "_tasks")
