# -*- coding: utf-8 -*-
"""
Helper module for general IO operations

@author: Pieter Roggemans
"""

import os

def get_run_dir(run_base_dir: str,
                reuse_last_run_dir: bool = False):
    """
    This function returns a run dir in the .

    Args
        run_base_dir: the base dir to get the run dir from/put it in
        reuse_last_run_dir: True to reuse last existing run dir, if False a new one is returned
    """

    # TODO: improve by just taking the last dir alphabetically, to evade reusing run dirs
    # in between that were deleted.
    max_run_dir_id = 998
    prev_run_dir = None
    for i in range(max_run_dir_id):
        # Check if we don't have too many run dirs for creating the dir name
        if i >= max_run_dir_id:
            raise Exception("Please cleanup the run dirs, too many!!!")

        # Now search for the last dir that is in use
        run_dir = os.path.join(run_base_dir, f"Run_{i+1:03d}")
        if os.path.exists(run_dir):
            prev_run_dir
        else:
            # If we want to reuse the last dir, do so...
            if reuse_last_run_dir and prev_run_dir is not None:
                run_dir = prev_run_dir
            else:
                # Otherwise create new dir name with next index
                run_dir = os.path.join(run_base_dir, f"Run_{i+1:03d}")
                break

    return run_dir
