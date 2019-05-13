# -*- coding: utf-8 -*-
"""
This module contains general functions that apply to timeseries data...

@author: Pieter Roggemans
"""

import logging
import os
import glob
from typing import List
import pandas as pd
import cropclassification.helpers.config_helper as conf

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Some constantsto choose which type of data to use in the marker.
# Remark: the string needs to be the same as the end of the name of the columns in the csv files!
PARCELDATA_AGGRAGATION_MEAN = 'mean'      # Mean value of the pixels values in a parcel.
PARCELDATA_AGGRAGATION_STDDEV = 'stdDev'  # std dev of the values of the pixels in a parcel

# Constants for types of sensor data
SENSORDATA_S1 = 'S1'                    # Sentinel 1 data
SENSORDATA_S1DB = 'S1dB'                # Sentinel 1 data, in dB
SENSORDATA_S1_ASCDESC = 'S1AscDesc'     # Sentinel 1 data, divided in Ascending and Descending passes
SENSORDATA_S1DB_ASCDESC = 'S1dBAscDesc' # Sentinel 1 data, in dB, divided in Ascending and Descending passes
SENSORDATA_S2 = 'S2'                    # Sentinel 2 data
SENSORDATA_S2gt95 = 'S2gt95'            # Sentinel 2 data (B2,B3,B4,B8) IF available for 95% or area

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def collect_and_prepare_timeseries_data(imagedata_dir: str,
                                        base_filename: str,
                                        output_csv: str,
                                        start_date_str: str,
                                        end_date_str: str,
                                        sensordata_to_use: List[str],
                                        parceldata_aggregations_to_use: List[str],
                                        min_fraction_data_in_column: float = 0.0,
                                        force: bool = False):
    """
    Collect all timeseries data to use for the classification and prepare it by applying
    scaling,... as needed.
    """
    # TODO: If we use S2 data, it is necessary to fill missing values in whatever way, otherwise
    #       there won't be a classification at all for that parcel!!!

    # If force == False Check and the output file exists already, stop.
    if(force is False and os.path.exists(output_csv) is True):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_csv}")
        return

    logger.info(f"Parceldata aggregations that need to be used: {parceldata_aggregations_to_use}")

    logger.setLevel(logging.DEBUG)
    logger.info("Create the input file for the classification")

    # Loop over all input data... to find the data we really need...
    filepath_start = os.path.join(imagedata_dir, f"{base_filename}_{start_date_str}.csv")
    filepath_end = os.path.join(imagedata_dir, f"{base_filename}_{end_date_str}.csv")
    logger.debug(f'filepath_start_date: {filepath_start}')
    logger.debug(f'filepath_end_date: {filepath_end}')

    csv_files = glob.glob(os.path.join(imagedata_dir, f"{base_filename}_*.csv"))
    result_df = None
    for curr_csv in sorted(csv_files):

        # Only process data that is of the right sensor types
        curr_csv_noext = os.path.splitext(curr_csv)[0]
        sensor_type = curr_csv_noext.split('_')[-1]
        if sensor_type not in sensordata_to_use:
            logger.debug(f"File is not in senser types asked ({sensordata_to_use}), skip it: {curr_csv}")
            continue

        # The only data we want to process is the data in the range of dates
        if((curr_csv < filepath_start) or (curr_csv >= filepath_end)):
            logger.debug(f"File is not in date range asked, skip it: {curr_csv}")
            continue

        logger.info(f'Process file: {curr_csv}')

        # An empty file signifies that there wasn't any valable data for that period/sensor/...
        if os.path.getsize(curr_csv) == 0:
            logger.info(f"File is empty, so SKIP: {curr_csv}")
            continue

        # Read data, and loop over columns to check if there are columns that need to be dropped.
        df_in = pd.read_csv(curr_csv, low_memory=False)
        for column in df_in.columns:

            # If it is the id column, continue
            if column == conf.csv['id_column']:
                continue

            # Check if the column is "asked"
            column_ok = False
            for parceldata_aggregation in parceldata_aggregations_to_use:
                if column.endswith('_' + parceldata_aggregation):
                    column_ok = True

            if column_ok is False:
                # Drop column if it doesn't end with something in parcel_data_aggregations_to_use
                logger.debug(f"Drop column as it's column aggregation isn't to be used: {column}")
                df_in.drop(column, axis=1, inplace=True)
                continue

            fraction_real_data = 1-(df_in[column].isnull().sum()/len(df_in[column]))
            if fraction_real_data < min_fraction_data_in_column:
                # If the number of nan values for the column > x %, drop column
                logger.warn(f"Drop column as it contains only {fraction_real_data} real data (= not nan) which is < {min_fraction_data_in_column}!: {column}")
                df_in.drop(column, axis=1, inplace=True)

        # Loop over columns to check if there are columns that need to be rescaled.
        for column in df_in.columns:
            if column.startswith(SENSORDATA_S2):
                logger.info(f"Column contains S2 data, so scale it by dividing by 10.000: {column}")
                df_in[column] = df_in[column]/10000

        # Set the index to the id_columnname, and join the data to the result...
        df_in.set_index(conf.csv['id_column'], inplace=True)
        if result_df is None:
            result_df = df_in
        else:
            nb_rows_before_join = len(result_df)
            result_df = result_df.join(df_in, how='inner')
            nb_rows_after_join = len(result_df)
            if nb_rows_after_join != nb_rows_before_join:
                logger.warning(f"Number of rows in result decreased in join from {nb_rows_before_join} to {nb_rows_after_join}")

    # Write all rows that have empty data to a file
    df_parcel_with_empty_data = result_df[result_df.isnull().any(1)]
    if len(df_parcel_with_empty_data) > 0:
        # Write the rows with empty data to a csv file
        parcel_with_empty_data_csv = f'{output_csv}_rowsWithEmptyData.csv'
        logger.warn(f"There were {len(df_parcel_with_empty_data)} rows with at least one columns = nan, write them to {parcel_with_empty_data_csv}")
        df_parcel_with_empty_data.to_csv(parcel_with_empty_data_csv)

        # replace empty data with 0
        logger.info("Replace NAN values with 0, so the rows are usable afterwards!")
        result_df.fillna(0, inplace=True)                   # Replace NAN values with 0
        #result.dropna(inplace=True)                        # Delete rows with empty values

    # Write output file...
    logger.debug(f"Start writing output to csv: {output_csv}")
    result.to_csv(output_csv)
    logger.info(f"Write output to file, start: {output_csv}")
    logger.info(f"Write output to file, ready (with shape: {result_df.shape})")
