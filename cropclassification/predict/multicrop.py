# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:18:13 2018

@author: Pieter Roggemans
"""

import logging

import pandas as pd

import cropclassification.global_settings as gs

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def detect_multicrop(input_parcel_csv: str,
                     input_parcel_timeseries_data_csv: str):

    '''
    logger.info(f"Read input file: {input_parcel_csv}")
    df_input_parcel = pd.read_csv(input_parcel_csv, low_memory=False)
    logger.debug('Read train file ready')
    '''

    # If the classification data isn't passed as dataframe, read it from the csv
    logger.info(f"Read classification data file: {input_parcel_timeseries_data_csv}")
    df_timeseries_data = pd.read_csv(input_parcel_timeseries_data_csv, low_memory=False)
    df_timeseries_data.set_index(gs.id_column, inplace=True)
    logger.debug('Read classification data file ready')

    # Add column with the max of all columns (= all stdDev's)
    df_timeseries_data['max_stddev'] = df_timeseries_data.max(axis=1)

    '''
    # Prepare the data to send to prediction logic...
    logger.info("Join train sample with the classification data")
    df_input_parcel_for_detect = (df_input_parcel#[[gs.id_column, gs.class_column]]
                                   .join(df_timeseries_data
                                         , how='inner', on=gs.id_column))

    # Only keep the parcels with relevant crops/production types
    productiontype_column = 'm#GESP_PM'
    if productiontype_column in df_input_parcel_for_detect.columns:
        # Serres, tijdelijke overkappingen en loodsen
        df_input_parcel_for_detect.loc[~df_input_parcel_for_detect[productiontype_column].isin(['SER', 'SGM'])]
        df_input_parcel_for_detect.loc[~df_input_parcel_for_detect[productiontype_column].isin(['PLA', 'PLO', 'NPO'])]
        df_input_parcel_for_detect.loc[df_input_parcel_for_detect[productiontype_column] != 'LOO']     # Een loods is hetzelfde als een stal...
        df_input_parcel_for_detect.loc[df_input_parcel_for_detect[productiontype_column] != 'CON']    # Containers, niet op volle grond...

    crop_columnname = 'm#GWSCOD_H'
    df_input_parcel_for_detect.loc[~df_input_parcel_for_detect[crop_columnname].isin(['1', '2', '3'])]

    # Keep the parcels with the 1000 largest stdDev
    df_largest = df_input_parcel_for_detect.nlargest(1000, columns='max_stddev', keep='first')
    '''

    #df_result = df_timeseries_data['max_stddev'].to_frame()
    df_result = df_timeseries_data
    logger.info(df_result)

    # Write to file
    output_filepath = input_parcel_timeseries_data_csv + '_largestStdDev.csv'
    logger.info(f"Write output file: {output_filepath}")
    df_result.to_csv(output_filepath)
