"""
Tests for functionalities in io_util.
"""

import pytest

from cropclassification.util import io_util


def test_create_file_atomic(tmp_path):
    path = tmp_path / "testje_atomic.txt"
    file_created = io_util.create_file_atomic(path)
    assert file_created
    file_created = io_util.create_file_atomic(path)
    assert not file_created


def test_output_exists(tmp_path):
    path = tmp_path / "test.txt"
    assert not io_util.output_exists(path, remove_if_exists=False)
    assert not io_util.output_exists(path, remove_if_exists=True)
    path.touch()
    assert io_util.output_exists(path, remove_if_exists=False)
    assert not io_util.output_exists(path, remove_if_exists=True)

    tmp_dir = tmp_path / "subdir" / "test.txt"
    with pytest.raises(ValueError, match="output directory does not exist"):
        io_util.output_exists(tmp_dir, remove_if_exists=False)
