import os
from pathlib import Path


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
