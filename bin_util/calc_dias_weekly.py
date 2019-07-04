# -*- coding: utf-8 -*-
"""
Calaculate the timeseries data per week based on per image data.
"""

from datetime import datetime
import os
import shutil
import sys
[sys.path.append(i) for i in ['.', '..']]

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
import cropclassification.preprocess.timeseries_util as ts_util

def main():
   
    config_filepaths = ["config/general.ini",
                        "config/local_overrule.ini"]
    year = 2018

    # Read the configuration files
    conf.read_config(config_filepaths, year=year)

    # Init logging
    base_log_dir = conf.dirs['log_dir']
    log_dir = f"{base_log_dir}{os.sep}calc_dias_weekly{datetime.now():%Y-%m-%d_%H-%M-%S}"
    global logger
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Get the config needed
    timeseries_per_image_dir = conf.dirs['timeseries_per_image_dir']
    timeseries_periodic_dir = conf.dirs['timeseries_periodic_dir']

    # Input features file depends on the year
    if year == 2017:
        input_features_filename = "Prc_BEFL_2017_2019-06-14_bufm5.shp"
    elif year == 2018:
        input_features_filename = "Prc_BEFL_2018_2019-06-14_bufm5.shp"
    elif year == 2019:
        input_features_filename = "Prc_BEFL_2019_2019-06-25_bufm5.shp"
    else:
        raise Exception(f"Not a valid year: {year}")

    # Calculate!
    input_parcel_filepath = os.path.join(conf.dirs['input_dir'], input_features_filename)
    ts_util.calculate_periodic_data(
            input_parcel_filepath=input_parcel_filepath,
            input_base_dir=timeseries_per_image_dir,
            start_date_str=f"{year}-03-15",
            end_date_str=f"{year}-08-15",
            #sensordata_to_get=conf.marker.getlist('sensordata_to_use'),
            sensordata_to_get=['SENSORDATA_S1_COHERENCE'],
            dest_data_dir=timeseries_periodic_dir,
            force=False)

if __name__ == '__main__':
    main()