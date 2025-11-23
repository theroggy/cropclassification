"""I/O utility functions."""

import logging
import os
import tempfile
from pathlib import Path

# Get a logger...
logger = logging.getLogger(__name__)


def create_file_atomic(path: Path) -> bool:
    """Create a lock file in an atomic way, so it is threadsafe.

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


def create_tempdir(base_dirname: str, parent_dir: Path | None = None) -> Path:
    """Creates a new tempdir in the default temp location.

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


def output_exists(
    path: Path, remove_if_exists: bool, log_prefix: str | None = None
) -> bool:
    """Check if the output file exists.

    If ``remove_if_exists`` is True, the file is removed and False is returned.

    Args:
        path (Path): Output file path to check.
        remove_if_exists (bool): If True, remove the output file if it exists.
        log_prefix (str, optional): Prefix to use when logging that the file already
            exists. Can be used to give more context in the logging. Defaults to None.

    Raises:
        ValueError: raised when the output directory does not exist.

    Returns:
        bool: True if the file exists.
    """
    if path.exists():
        if remove_if_exists:
            path.unlink()
            return False
        else:
            log_prefix = f"{log_prefix}: " if log_prefix is not None else ""
            logger.debug(f"{log_prefix}force is False and {path.name} exists already")
            return True

    elif not path.parent.exists():
        raise ValueError(f"output directory does not exist: {path.parent}")

    return False
