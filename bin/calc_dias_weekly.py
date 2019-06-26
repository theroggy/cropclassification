# -*- coding: utf-8 -*-
"""
Calaculate the timeseries data per week based on per image data.
"""

from datetime import datetime
import glob
import os
import shutil
import sys
[sys.path.append(i) for i in ['.', '..']]

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
import cropclassification.preprocess.timeseries_calc_dias_onda as calc

def main():
   
    config_filepaths = ["../config/general.ini",
                        "../config/local_overrule.ini"]
    year = 2018
    test = True

    # Read the configuration files
    conf.read_config(config_filepaths, year=year)

    # Get the general output dir
    timeseries_per_image_dir = conf.dirs['timeseries_per_image_dir']
    timeseries_periodic_dir = conf.dirs['timeseries_periodic_dir']

    if test is True:
        timeseries_periodic_dir += "_test"

    # Init logging
    base_log_dir = conf.dirs['log_dir']
    log_dir = f"{base_log_dir}{os.sep}calc_dias_weekly{datetime.now():%Y-%m-%d_%H-%M-%S}"
    global logger
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Calculate!
    band = 'VV'
    orbit = 'ASC'
    calc.calculate_weekly_data(
            input_filepath=timeseries_per_image_dir,
            input_band=band, 
            input_orbit=orbit, 
            output_filepath=timeseries_periodic_dir)
