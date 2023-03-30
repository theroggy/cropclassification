# -*- coding: utf-8 -*-
"""
Module to init logging.
"""

import datetime
import logging
import os
from pathlib import Path


def main_log_init(log_dir: Path, log_basefilename: str, log_level: str = "INFO"):
    # Make sure the log dir exists
    if not log_dir.exists():
        os.makedirs(log_dir, exist_ok=True)

    # Get root logger
    logger = logging.getLogger("")

    # Set the general maximum log level...
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.flush()
        handler.close()

    # Remove all handlers and add the ones I want again, so a new log file is created for each run
    # Remark: the function removehandler doesn't seem to work?
    logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # ch.setFormatter(logging.Formatter('%(levelname)s|%(name)s|%(message)s'))
    # ch.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    #                                  datefmt='%H:%M:%S,uuu'))
    ch.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    log_filepath = (
        log_dir / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{log_basefilename}.log"
    )
    fh = logging.FileHandler(filename=str(log_filepath))
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(fh)

    return logger
