# -*- coding: utf-8 -*-
"""
Script to create timeseries data per parcel of
  - S1: the mean VV and VH backscatter data
  - S2: the 4 bands for periods when there is good coverage of cloudfree images of the area of
        interest
"""

import logging
import os

# Import local stuff
import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess.timeseries_util as ts_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def calc_timeseries_data(
        input_parcel_filepath: str,
        input_country_code: str,
        start_date_str: str,
        end_date_str: str,
        sensordata_to_get: [],
        base_filename: str,
        dest_data_dir: str):
    """ 
    Calculate timeseries data.
    
    Args:
        input_parcel_filepath (str): [description]
        input_country_code (str): [description]
        start_date_str (str): [description]
        end_date_str (str): [description]
        sensordata_to_get ([type]): [description]
        base_filename (str): [description]
        dest_data_dir (str): [description]
    
    Raises:
        Exception: [description]
    """
    # Check and init some variables...
    if sensordata_to_get is None:
        raise Exception("sensordata_to_get cannot be None")
    if not os.path.exists(dest_data_dir):
        os.mkdir(dest_data_dir)

    # TODO: start calculation of per image data on DIAS
    timeseries_per_image_dir = conf.dirs['timeseries_per_image_dir']

    # Now all image data is available per image, calculate periodic data
    ts_util.calculate_periodic_data(
            input_parcel_filepath=input_parcel_filepath,
            input_base_dir=timeseries_per_image_dir,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            sensordata_to_get=sensordata_to_get,  
            output_dir=dest_data_dir)
