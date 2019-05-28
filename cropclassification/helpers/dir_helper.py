"""
Helper regarding directory operations.
"""

import os
import re

def create_run_dir(class_base_dir: str,
                   reuse_last_run_dir: bool) -> str:
    """
    Create a new run dir, or get the last run dir.

    Args
        class_base_dir: the base dir to use to create the run dir in
        reuse_last_run_dir: True to find the latest existing run dir and return that, False to create a new run dir
    """

    # Create class_base_dir if it doesn't exist
    if not os.path.exists(class_base_dir):
        os.makedirs(class_base_dir)

    # Look for all existing run dirs
    pattern = re.compile('Run_[0-9]{3}')
    dir_list = [x.path for x in os.scandir(class_base_dir) if x.is_dir() and re.search(pattern, x.path)]
    #print(f"Dirs found: {dir_list}")
    if (dir_list is None or not any(dir_list)):
        # first run
        return os.path.join(class_base_dir, f"Run_{1:03d}")

    # get last run and increment if needed
    last_dir = sorted(dir_list, reverse=True)[0]
    #print(f"Last dir: {last_dir}")

    if reuse_last_run_dir:
        return last_dir

    last_dir_iteration = re.search(pattern, last_dir)
    last_iteration = int(last_dir_iteration.group().split('_')[1])
    return os.path.join(class_base_dir, f"Run_{last_iteration + 1:03d}")
