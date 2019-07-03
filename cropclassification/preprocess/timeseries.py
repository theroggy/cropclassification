# -*- coding: utf-8 -*-
"""
This module contains general functions that apply to timeseries data...
"""

import glob
import logging
import os
from typing import List

import pandas as pd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh
import cropclassification.preprocess.timeseries_util as ts_util

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
    """
    Calculate timeseries data for the input parcels

    Args:
        input_parcel_filepath (str): [description]
        input_country_code (str): [description]
        start_date_str (str): [description]
        end_date_str (str): [description]
        sensordata_to_get (List[str]): an array with data you want to be calculated: 
                check out the constants starting with DATA_TO_GET... for the options.
        base_filename (str): [description]
        dest_data_dir (str): [description]
    """
    # Check some variables...
    if sensordata_to_get is None:
        raise Exception("sensordata_to_get cannot be None")

    # As we want a weekly calculation, get nearest monday for start and stop day
    start_date = ts_util.get_monday(start_date_str) # output: vb 2018_2_1 - maandag van week 2 van 2018
    end_date = ts_util.get_monday(end_date_str) 
    start_date_monday = start_date.strftime('%Y-%m-%d') # terug omzetten naar Y/M/D
    end_date_monday = end_date.strftime('%Y-%m-%d')

    timeseries_calc_type = conf.timeseries['timeseries_calc_type']
    if timeseries_calc_type == 'gee':
        import cropclassification.preprocess.timeseries_calc_gee as ts_calc_gee

        return ts_calc_gee.calc_timeseries_data(
                input_parcel_filepath=input_parcel_filepath,
                input_country_code=input_country_code,
                start_date_str=start_date_monday,
                end_date_str=end_date_monday,
                sensordata_to_get=sensordata_to_get,
                base_filename=base_filename,
                dest_data_dir=dest_data_dir)
    elif timeseries_calc_type == 'onda':
        import cropclassification.preprocess.timeseries_calc_dias_onda as ts_calc_dias_onda

        return ts_calc_dias_onda.calc_timeseries_data(
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

def collect_and_prepare_timeseries_data(
        input_parcel_filepath: str,
        timeseries_dir: str,
        base_filename: str,
        output_filepath: str,
        start_date_str: str,
        end_date_str: str,
        sensordata_to_use: List[str],
        parceldata_aggregations_to_use: List[str],
        force: bool = False):
    """
    Collect all timeseries data to use for the classification and prepare it by applying
    scaling,... as needed.
    """

    # Some constants to choose which type of data to use in the marker.
    # Remark: the string needs to be the same as the end of the name of the columns in the csv files!
    PARCELDATA_AGGRAGATION_MEAN = conf.general['PARCELDATA_AGGRAGATION_MEAN']      # Mean value of the pixels values in a parcel.
    PARCELDATA_AGGRAGATION_STDDEV = conf.general['PARCELDATA_AGGRAGATION_STDDEV']  # std dev of the values of the pixels in a parcel

    # Constants for types of sensor data
    SENSORDATA_S1 = conf.general['SENSORDATA_S1']                     # Sentinel 1 data
    SENSORDATA_S1DB = conf.general['SENSORDATA_S1DB']                 # Sentinel 1 data, in dB
    SENSORDATA_S1_ASCDESC = conf.general['SENSORDATA_S1_ASCDESC']     # Sentinel 1 data, divided in Ascending and Descending passes
    SENSORDATA_S1DB_ASCDESC = conf.general['SENSORDATA_S1DB_ASCDESC'] # Sentinel 1 data, in dB, divided in Ascending and Descending passes
    SENSORDATA_S2 = conf.general['SENSORDATA_S2']                     # Sentinel 2 data
    SENSORDATA_S2gt95 = conf.general['SENSORDATA_S2gt95']             # Sentinel 2 data (B2,B3,B4,B8) IF available for 95% or area
    SENSORDATA_S1_COHERENCE = conf.general['SENSORDATA_S1_COHERENCE']

    # If force == False Check and the output file exists already, stop.
    if(force is False and os.path.exists(output_filepath) is True):
        logger.warning(f"Output file already exists and force == False, so stop: {output_filepath}")
        return

    # Init the result with the id's of the parcels we want to treat
    result_df = pdh.read_file(input_parcel_filepath, columns=[conf.columns['id']])
    if result_df.index.name != conf.columns['id']: 
        result_df.set_index(conf.columns['id'], inplace=True)
    nb_input_parcels = len(result_df.index)
    logger.info(f"Parceldata aggregations that need to be used: {parceldata_aggregations_to_use}")
    logger.setLevel(logging.DEBUG)

    # Loop over all input timeseries data to find the data we really need
    columndata_ext = conf.general['columndata_ext']
    filepath_start = os.path.join(timeseries_dir, f"{base_filename}_{start_date_str}{columndata_ext}")
    filepath_end = os.path.join(timeseries_dir, f"{base_filename}_{end_date_str}{columndata_ext}")
    logger.debug(f'filepath_start_date: {filepath_start}')
    logger.debug(f'filepath_end_date: {filepath_end}')

    ts_data_files = glob.glob(os.path.join(timeseries_dir, f"{base_filename}_*{columndata_ext}"))
    for curr_filepath in sorted(ts_data_files):

        # Only process data that is of the right sensor types
        curr_filepath_noext = os.path.splitext(curr_filepath)[0]
        sensor_type = curr_filepath_noext.split('_')[-1]
        if sensor_type not in sensordata_to_use:
            logger.debug(f"SKIP: file is not in sensor types asked ({sensordata_to_use}): {curr_filepath}")
            continue
        # The only data we want to process is the data in the range of dates
        if((curr_filepath < filepath_start) or (curr_filepath >= filepath_end)):
            logger.debug(f"SKIP: File is not in date range asked: {curr_filepath}")
            continue
        # An empty file signifies that there wasn't any valable data for that period/sensor/...
        if os.path.getsize(curr_filepath) == 0:
            logger.info(f"SKIP: file is empty: {curr_filepath}")
            continue

        # Read data, and check if there is enough data in it
        data_read_df = pdh.read_file(curr_filepath)
        nb_data_read = len(data_read_df.index)
        fraction_data_available = nb_data_read/nb_input_parcels
        min_fraction_parcels_with_data = conf.timeseries.getfloat('min_fraction_parcels_with_data')
        if fraction_data_available < min_fraction_parcels_with_data:
            logger.info(f"SKIP: only data for {fraction_data_available} of parcels, should be > {min_fraction_parcels_with_data}: {curr_filepath}")
            continue

        # Start processing the file
        logger.info(f'Process file: {curr_filepath}') 
        if data_read_df.index.name != conf.columns['id']: 
            data_read_df.set_index(conf.columns['id'], inplace=True)

        # Loop over columns to check if there are columns that need to be dropped.       
        for column in data_read_df.columns:

            # If it is the id column, continue
            if column == conf.columns['id']:
                continue

            # Check if the column is "asked"
            column_ok = False
            for parceldata_aggregation in parceldata_aggregations_to_use:
                if column.endswith('_' + parceldata_aggregation):
                    column_ok = True
            if column_ok is False:
                # Drop column if it doesn't end with something in parcel_data_aggregations_to_use
                logger.debug(f"Drop column as it's column aggregation isn't to be used: {column}")
                data_read_df.drop(column, axis=1, inplace=True)
                continue

            # Check if the column contains data for enough parcels
            fraction_input_data = 1-(data_read_df[column].isnull().sum()/nb_input_parcels)
            if fraction_input_data < min_fraction_parcels_with_data:
                # If the number of nan values for the column > x %, drop column
                logger.warn(f"Drop column as it contains only {fraction_input_data} real data compared to input (= not nan) which is < {min_fraction_parcels_with_data}!: {column}")
                data_read_df.drop(column, axis=1, inplace=True)
            """
            fraction_real_data = 1-(data_read_df[column].isnull().sum()/len(data_read_df[column]))
            if fraction_real_data < min_fraction_parcels_with_data:
                # If the number of nan values for the column > x %, drop column
                logger.warn(f"Drop column as it contains only {fraction_real_data} real data (= not nan) which is < {min_fraction_parcels_with_data}!: {column}")
                data_read_df.drop(column, axis=1, inplace=True)
            """

        # If S2, rescale data
        if sensor_type.startswith(SENSORDATA_S2):
            for column in data_read_df.columns:
                logger.info(f"Column contains S2 data, so scale it by dividing by 10.000: {column}")
                data_read_df[column] = data_read_df[column]/10000

        # If S1 coherence, rescale data
        if sensor_type == SENSORDATA_S1_COHERENCE:
            for column in data_read_df.columns:
                logger.info(f"Column contains S1 Coherence data, so scale it by dividing by 300: {column}")
                data_read_df[column] = data_read_df[column]/300

        # Join the data to the result...
        result_df = result_df.join(data_read_df, how='left')

    # Remove rows with many null values from result
    max_number_null = int(0.6 * len(result_df.columns))
    parcel_many_null_df = result_df[result_df.isnull().sum(axis=1) > max_number_null]
    if len(parcel_many_null_df.index) > 0:
        # Write the rows with empty data to a file
        parcel_many_null_filepath = f'{output_filepath}_rows_many_null.sqlite'
        logger.warn(f"Write {len(parcel_many_null_df.index)} rows with > {max_number_null} of {len(result_df.columns)} columns==null to {parcel_many_null_filepath}")
        pdh.to_file(parcel_many_null_df, parcel_many_null_filepath)

        # Now remove them from result
        result_df = result_df[result_df.isnull().sum(axis=1) <= max_number_null]

    # For rows with some null values, set them to 0
    #result_df.fillna(0, inplace=True)
    # TODO: it is more logical to interpolate only between the different types of data:
    # S1_VV, S1_VH, ASC?, DESC?, S2?
    #result_df.interpolate(inplace=True)
    result_df.fillna(0, inplace=True)
    
    # Write output file...
    logger.info(f"Write output to file, start: {output_filepath}")
    pdh.to_file(result_df, output_filepath)
    logger.info(f"Write output to file, ready (with shape: {result_df.shape})")
