# -*- coding: utf-8 -*-
"""
Script to create timeseries data per parcel of
  - S1: the mean VV and VH backscatter data
  - S2: the 4 bands for periods when there is good coverage of cloudfree images of the area of
        interest
"""

from datetime import datetime
from datetime import timedelta
import glob
import logging
import os
import pathlib
import time
from typing import List

import numpy as np
import pandas as pd

# Import local stuff
import cropclassification.preprocess.timeseries as ts
import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def calc_timeseries_data(input_parcel_filepath: str,
                         input_country_code: str,
                         start_date_str: str,
                         end_date_str: str,
                         sensordata_to_get: List[str],
                         base_filename: str,
                         dest_data_dir: str):
    """ Calculate timeseries data for the input parcels

    args
    ------------
        data_to_get: an array with data you want to be calculated: check out the constants starting
                     with DATA_TO_GET... for the options.
    """
    # Check some variables...
    if sensordata_to_get is None:
        raise Exception("sensordata_to_get cannot be None")

    timeseries_calc_type = conf.timeseries['timeseries_calc_type']
    if timeseries_calc_type == 'gee':
        import cropclassification.preprocess.timeseries_calc_gee as tc_gee

        return tc_gee.calc_timeseries_data(
                input_parcel_filepath=input_parcel_filepath,
                input_country_code=input_country_code,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                sensordata_to_get=sensordata_to_get,
                base_filename=base_filename,
                dest_data_dir=dest_data_dir)
    else:
        message = f"Unsupported timeseries calculation type: {timeseries_calc_type}"
        logger.error(message)
        raise Exception(message)

'''
def calculate_sentinel_timeseries(input_parcel_filepath: str,
                                  input_country_code: str,
                                  start_date_str: str,
                                  end_date_str: str,
                                  sensordata_to_get: List[str],
                                  base_filename: str,
                                  dest_data_dir: str):

    # Init some variables
    dest_data_dir_todownload = os.path.join(dest_data_dir, 'TODOWNLOAD')
    if not os.path.exists(dest_data_dir_todownload):
        os.mkdir(dest_data_dir_todownload)

    # TODO: add check if the ID column exists in the parcel file, otherwise he processes everything without ID column in output :-(!!!

    # First adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
    logger.info('Adapt start_date and end_date so they are mondays')
    def get_monday(date_str):
        """ Get the first monday before the date provided. """
        parseddate = datetime.strptime(date_str, '%Y-%m-%d')
        year_week = parseddate.strftime('%Y_%W')
        year_week_monday = datetime.strptime(year_week + '_1', '%Y_%W_%w')
        return year_week_monday

    start_date = get_monday(start_date_str)
    end_date = get_monday(end_date_str)       # Remark: de end date is exclusive in gee filtering, so must be a monday as well...
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Keep the tasklist in a variable so it is loaded only once (if necessary)

#    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
#    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    logging.info(f'Create sentinel timeseries from {start_date} till {end_date} for parcel in file {input_parcel_filepath}')

    # Add some columns with feature info, to be able to eg. filter only polygons...
#    def add_feature_info(feature):
#        return feature.set('area', feature.area(), 'perimeter', feature.perimeter(), 'type', feature.geometry().type())
#    bevl2017 = bevl2017.map(add_feature_info)

    """
    # Buffer
    # Now define a function to buffer the parcels inward
    def bufferFeature(ft):
      ft = ft.simplify(1).buffer(-20).simplify(1)
#     return ft.set({ncoords: ft.geometry().coordinates().length()})
      return ft
    input_parcels = input_parcels.map(bufferFeature)
    """

    # Export non-polygons..
#    bevl2017_nopoly = bevl2017.filterMetadata('type', 'not_equals', 'Polygon')
#    ee.batch.Export.table.toDrive(collection = bevl2017_nopoly, folder = 'Monitoring', description = 'BEVL2017_no_polygon', fileFormat = 'KMZ')


    def reduce_and_export(imagedata, reducer, export_descr: str):
        """ Reduces the imagedata over the features and export to drive. """

        # First check if the file exists already locally...
        # Format relevant local filename
        export_filename = export_descr + '.csv'
        dest_fullpath = os.path.join(dest_data_dir, export_filename)
        dest_fullpath_todownload = os.path.join(dest_data_dir_todownload, export_filename)

        # If the data is already available locally... go to next period
        if os.path.isfile(dest_fullpath):
            logger.info(f"For task {export_descr}, file already available locally: SKIP")
            return

        # If the data is already "ordered" in a previous run and is still busy processing, don't
        # start processing again
        if (os.path.isfile(dest_fullpath_todownload)
                and(check_if_task_exists(export_description, ['RUNNING', 'READY', 'COMPLETED']))):
            logger.info(f"For task {export_descr}, file still busy processing or is ready on gee: SKIP")
            return

    export_description = f"{base_filename}_pixcount"

    nb_periods = periods.length().getInfo()
    logger.info(f"Loop through all <{nb_periods}> periods")
    for i in range(0, nb_periods):

        # Calculate the start and end dates of this period...
        period_start_str = (start_date + timedelta(days=i*7)).strftime('%Y-%m-%d')
        period_end_str = (start_date + timedelta(days=(i+1)*7)).strftime('%Y-%m-%d')
        logger.debug(f"Process period: {period_start_str} till {period_end_str}")

        # Get mean s1 image of the s1 images that are available in this period
        SENSORDATA_S1 = conf.general['SENSORDATA_S1']
        if SENSORDATA_S1 in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
                return

        # Get mean s1 image of the s1 images that are available in this period
        SENSORDATA_S1DB = conf.general['SENSORDATA_S1DB']
        if SENSORDATA_S1DB in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1DB}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
                return

        # Get mean s1 asc and desc image of the s1 images that are available in this period
        SENSORDATA_S1_ASCDESC = conf.general['SENSORDATA_S1_ASCDESC']
        if SENSORDATA_S1_ASCDESC in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1_ASCDESC}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")

        # Get mean s1 in DB, asc and desc image of the s1 images that are available in this period
        SENSORDATA_S1DB_ASCDESC = conf.general['SENSORDATA_S1DB_ASCDESC']
        if SENSORDATA_S1DB_ASCDESC in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1DB_ASCDESC}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")

        # Get mean s2 image of the s2 images that have (almost)cloud free images available in this
        # period
        SENSORDATA_S2gt95 = conf.general['SENSORDATA_S2gt95']
        if SENSORDATA_S2gt95 in sensordata_to_get:
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S2gt95}"

            # If the data is already available locally... skip
            # Remark: this logic is puth here additionaly to evade having to calculate the 95% rule
            #         even if data is available.
            dest_fullpath = os.path.join(dest_data_dir, f"{sensordata_descr}.csv")
            if os.path.isfile(dest_fullpath):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
'''

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
