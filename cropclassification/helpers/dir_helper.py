"""Helper regarding directory operations."""

import os
import re
from datetime import datetime
from pathlib import Path


def create_run_dir(class_base_dir: Path, reuse_last_run_dir: bool) -> Path:
    """Create a new run dir, or get the last run dir.

    Args:
        class_base_dir: the base dir to use to create the run dir in
        reuse_last_run_dir: True to find the latest existing run dir and return that,
            False to create a new run dir
    """
    # Create class_base_dir if it doesn't exist
    class_base_dir.mkdir(parents=True, exist_ok=True)

    # Look for all existing run dirs
    base_filename = f"Run_{datetime.now().strftime('%Y-%m-%d')}"
    pattern = re.compile(base_filename + "_[0-9]{3}")
    dir_list = [
        x.path
        for x in os.scandir(class_base_dir)
        if x.is_dir() and re.search(pattern, x.path)
    ]

    # No dir yet with the pattern -> return new one
    if dir_list is None or not any(dir_list):
        run_dir = class_base_dir / f"{base_filename}_{1:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    # Get last run dir found... if we want to reuse it, return it
    last_dir = sorted(dir_list, reverse=True)[0]
    if reuse_last_run_dir:
        return Path(last_dir)

    # We don't want to reuse it, so create next one
    _, last_dirname = os.path.split(last_dir)
    last_iteration = int(last_dirname.replace(base_filename, "tmp").split("_")[1])
    run_dir = class_base_dir / f"{base_filename}_{last_iteration + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
