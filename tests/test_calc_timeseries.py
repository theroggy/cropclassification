from pathlib import Path


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def test_test():
    assert True
