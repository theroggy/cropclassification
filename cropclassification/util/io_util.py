import os
from pathlib import Path
import tempfile
from typing import Optional


def create_file_atomic(path: Path):
    """
    Create a lock file in an atomic way, so it is threadsafe.

    Returns True if the file was created by this thread, False if the file existed
    already.
    """
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL)
        return True
    except FileExistsError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def create_tempdir(base_dirname: str, parent_dir: Optional[Path] = None) -> Path:
    """
    Creates a new tempdir in the default temp location.

    Remark: the temp dir won't be cleaned up automatically!

    Examples:

        - base_dirname="foo" -> /tmp/foo_000001
        - base_dirname="foo/bar" -> /tmp/foo/bar_000001

    Args:
        base_dirname (str): The name the tempdir will start with. The name will be
            suffixed with a number to make the directory name unique. If a "/" is part
            of the base_dirname a subdirectory will be created: e.g. "foo/bar".
        parent_dir (Path, optional): The dir to create the tempdir in. If None, the
            system temp dir is used. Defaults to None.

    Raises:
        Exception: if it wasn't possible to create the temp dir because there
            wasn't found a unique directory name.

    Returns:
        Path: the path to the temp dir created.
    """

    if parent_dir is None:
        parent_dir = Path(tempfile.gettempdir())

    for i in range(1, 999999):
        try:
            tempdir = parent_dir / f"{base_dirname}_{i:06d}"
            tempdir.mkdir(parents=True)
            return tempdir
        except FileExistsError:
            continue

    raise Exception(
        f"Wasn't able to create a temporary dir with basedir: "
        f"{parent_dir / base_dirname}"
    )
