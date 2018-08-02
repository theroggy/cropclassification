# -*- coding: utf-8 -*-
"""
Module with helper functions to preprocess the data to use for the classification.

@author: Pieter Roggemans
"""

import logging
import os
import glob
import pandas as pd
import classification_preprocessing_BEFL as befl
import global_settings as gs
from typing import List

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Some constants to choose the balancing strategy to use to crete the training set
BALANCING_STRATEGY_NONE = 'BALANCE_NONE'
BALANCING_STRATEGY_MEDIUM = 'BALANCE_MEDIUM'
BALANCING_STRATEGY_EQUAL = 'BALANCE_EQUAL'

# Some constantsto choose which type of data to use in the marker.
# Remark: the string needs to be the same as the end of the name of the columns in the csv files!
PARCELDATA_AGGRAGATION_MEAN       = 'mean'      # Mean value of the pixels values in a parcel.
PARCELDATA_AGGRAGATION_STDDEV     = 'stdDev'    # Standard Deviation of the values of the pixels in a parcel

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_input(input_parcel_filepath: str
                 ,input_filetype: str
                 ,output_parcel_filepath: str
                 ,output_classes_type: str
                 ,force: bool = False):
    """ Prepare a raw input file by eg. adding the classification classes to use for the classification,... """

    # If force == False Check and the output file exists already, stop.
    if force == False and os.path.exists(output_parcel_filepath):
        logger.warning(f"prepare_input: output file already exists and force == False, so stop: {output_parcel_filepath}")
        return

    if input_filetype == 'BEFL':
        befl.prepare_input(input_parcel_filepath = input_parcel_filepath
                          ,output_parcel_filepath = output_parcel_filepath
                          ,output_classes_type = output_classes_type)
    else:
        message = f"Unknown value for parameter input_filetype: {input_filetype}"
        logger.critical(message)
        raise Exception(message)

def collect_and_prepare_timeseries_data(imagedata_dir: str
                                       ,base_filename: str
                                       ,start_date_str: str
                                       ,end_date_str: str
                                       ,parceldata_aggregations_to_use: List[str]
                                       ,output_csv: str
                                       ,force: bool = False):
    """ Collect all timeseries data to use for the classification and prepare it by applying scaling,... as needed. """
    # TODO: If we use S2 data, it is necessary to fill missing values in whatever way, otherwise there won't be a classification
    #       at all for that parcel!!!

    # If force == False Check and the output file exists already, stop.
    if force == False and os.path.exists(output_csv):
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
    result = None
    for curr_csv in sorted(csv_files):
        # The only data we want to process is the data in the range of dates or ending with prcinfo.csv
        if((curr_csv < filepath_start) or (curr_csv >= filepath_end)):
            logger.debug(f'File is not needed for this classification, skip it: {curr_csv}')
            continue

        logger.info(f'Process file: {curr_csv}')

        # An empty file signifies that there wasn't any valable data for that period/sensor/...
        if os.path.getsize(curr_csv) == 0:
            logger.info(f"File is empty, so SKIP: {curr_csv}")
            continue

        # Read data, and loop over columns to check if there are columns that need to be dropped.
        df_in = pd.read_csv(curr_csv, low_memory=False)
        for column in df_in.columns:
            if column == gs.id_column:
                continue
            elif ((df_in[column].isnull().sum()/len(df_in[column])) > 0.1):
                # If the number of nan values for the column > x %, drop column
                logger.warn(f"Drop column as it contains > 10% nan values!: {column}")
                df_in.drop(column, axis=1, inplace=True)
            else:
                column_ok = False
                for parceldata_aggregation in parceldata_aggregations_to_use:
                    if column.endswith('_' + parceldata_aggregation):
                        column_ok = True

                # If the column aggregation of the column wasn't found in the parcel_data_aggregations_to_use, drop it.
                if column_ok == False:
                    logger.info(f"Drop column as it's column aggregation isn't to be used: {column}")
                    df_in.drop(column, axis=1, inplace=True)

        # Loop over columns to check if there are columns that need to be rescaled.
        for column in df_in.columns:
            if column.startswith('S2'):
                logger.info(f"Column contains S2 data, so scale it by dividing by 10.000: {column}")
                df_in[column] = df_in[column]/10000

        # Set the index to the id_columnname, and join the data to the result...
        df_in.set_index(gs.id_column, inplace=True)
        if result is None:
            result = df_in
        else:
            result = result.join(df_in, how='inner')

    # Write all rows that (still) have empty data to a file
    df_parcel_with_empty_data = result[result.isnull().any(1)]
    if len(df_parcel_with_empty_data) > 0:
        # Write the rows with empty data to a csv file
        parcel_with_empty_data_csv = f'{output_csv}_rowsWithEmptyData.csv'
        logger.warn('There were {len(df_parcel_with_empty_data)} rows with at least one columns = nan, write them to {parcel_with_empty_data_csv}')
        df_parcel_with_empty_data.to_csv(parcel_with_empty_data_csv)

        # Remove rows that have empty data
        logger.info('Remove the rows with empty data fields from the result')
        result.dropna(inplace=True)                 # Delete rows with empty values

    # Write output file...
    logger.debug(f"Start writing output to csv: {output_csv}")
    result.to_csv(output_csv)
    logger.info(f"Output (with shape: {result.shape}) written to : {output_csv}")

def create_train_test_sample(input_parcel_classes_csv: str
                            ,input_parcel_pixcount_csv: str
                            ,output_parcel_classes_train_csv: str
                            ,output_parcel_classes_test_csv: str
                            ,balancing_strategy: str
                            ,force: bool = False):
    """ Create a seperate train and test sample from the general input file. """

    # If force == False Check and the output files exist already, stop.
    if force == False and os.path.exists(output_parcel_classes_train_csv) and os.path.exists(output_parcel_classes_test_csv):
        logger.warning(f"create_train_test_sample: output files already exist and force == False, so stop: {output_parcel_classes_train_csv}, {output_parcel_classes_test_csv}")
        return

    # Load input data...
    logger.info(f'Start create_train_test_sample with balancing_strategy {balancing_strategy}')
    logger.info(f'Read input file {input_parcel_classes_csv}')
    df_in = pd.read_csv(input_parcel_classes_csv)
    logger.debug(f'Read input file ready, shape: {df_in.shape}')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_in.groupby(gs.class_column, as_index=False).size()
        logger.info(f'Number of elements per classname in input dataset:\n{count_per_class}')

    # Load pixcount data and join it
    logger.info(f'Read pixcount file {input_parcel_pixcount_csv}')
    df_pixcount = pd.read_csv(input_parcel_pixcount_csv)
    logger.debug(f'Read pixcount file ready, shape: {df_pixcount.shape}')
    df_pixcount.set_index(gs.id_column, inplace=True)
    df_in.set_index(gs.id_column, inplace=True)
    df_in = df_in.join(df_pixcount[gs.pixcount_s1s2_column], how='left')

    # The test dataset should be as representative as possible for the entire dataset, so create this first as a 20% sample of each class
    # without any additional checks...
    # Remark: group_keys=False evades that apply creates an extra index-level of the groups above the data
    #         and evades having to do an extra .reset_index(level=CLASS_COLUMN_NAME, drop=True) to get rid of the group level
    df_test = df_in.groupby(gs.class_column, group_keys=False).apply(pd.DataFrame.sample, frac=0.20)
    logger.debug(f"df_test after sampling 20% of data per class, shape: {df_test.shape}")

    # The candidate parcel for training are all non-test parcel
    df_train_base = df_in[~df_in.index.isin(df_test.index)]
    logger.debug(f"df_train_base after isin\n{df_train_base}")

    # Remove parcel with too few pixels from the train sample
    df_train_base = df_train_base[df_train_base[gs.pixcount_s1s2_column] >= 20]
    logger.debug(f'Number of parcel in df_train_base after filter on pixcount >= 20: {len(df_train_base)}')

    # The 'UNKNOWN' and 'IGNORE_' classes arent' meant for training... so remove them!
    logger.info("Remove the 'UNKNOWN' class from training sample")
    df_train_base = df_train_base[df_train_base[gs.class_column] != 'UNKNOWN']

    # The 'IGNORE_' classes aren't meant for training either...
    logger.info("Remove the classes that start with 'IGNORE_' from training sample")
    df_train_base = df_train_base[~df_train_base[gs.class_column].str.startswith('IGNORE_', na=True)]

    # Print the train base result before applying any balancing
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_train_base.groupby(gs.class_column, as_index=False).size()
        logger.info(f'Number of elements per classname for train dataset, before balancing:\n{count_per_class}')

    # Depending on the balancing_strategy, use different way to get a training sample
    if balancing_strategy == BALANCING_STRATEGY_NONE:
        # Just use 25% of all non-test data as train data -> 25% of 80% of data -> 20% of all data will be training date
        # Remark: - this is very unbalanced, eg. classes with 10.000 times the input size than other classes
        #         - this results in a relatively high accuracy in overall numbers, but the small classes are not detected at all
        df_train = df_train_base.groupby(gs.class_column, group_keys=False).apply(pd.DataFrame.sample, frac=0.15)

    elif balancing_strategy == BALANCING_STRATEGY_MEDIUM:
        # Balance the train data, but still use some larger samples for the classes that have a lot of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall, and also the smaller classes give some results
        #         with upper limit of 4000 results significantly less good.

        # For the larger classes, favor them a bit by leaving the samples larger but cap at upper_limit
        upper_limit = 10000
        lower_limit = 1000
        logger.info(f"Cap over {upper_limit}, keep the full number of training sample till {lower_limit}, samples smaller than that are oversampled")
        df_train = df_train_base.groupby(gs.class_column).filter(lambda x: len(x) >= upper_limit).groupby(gs.class_column, group_keys=False).apply(pd.DataFrame.sample, upper_limit)
        # Middle classes use the number as they are
        df_train = df_train.append(df_train_base.groupby(gs.class_column).filter(lambda x: len(x) < upper_limit).groupby(gs.class_column).filter(lambda x: len(x) >= lower_limit))
        # For smaller classes, oversample...
        df_train = df_train.append(df_train_base.groupby(gs.class_column).filter(lambda x: len(x) < lower_limit).groupby(gs.class_column, group_keys=False).apply(pd.DataFrame.sample, lower_limit, replace=True))

    elif balancing_strategy == BALANCING_STRATEGY_EQUAL:
        # In theory the most logical way to balance: make sure all classes have the same amount of training data by undersampling the
        # largest classes and oversampling the small classes.
        df_train = df_train_base.groupby(gs.class_column, group_keys=False).apply(pd.DataFrame.sample, 2000, replace=True)

    else:
        logger.fatal(f"Unknown balancing strategy, STOP!: {balancing_strategy}")

    # Log the resulting numbers per class in the train sample
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_train.groupby(gs.class_column, as_index=False).size()
        logger.info(f'Number of elements per classname in train dataset:\n{count_per_class}')

    # Log the resulting numbers per class in the test sample
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_test.groupby(gs.class_column, as_index=False).size()
        logger.info(f'Number of elements per classname in test dataset:\n{count_per_class}')

    # Write to output files
    logger.info('Write the output files')
    df_train.to_csv(output_parcel_classes_train_csv)    # The ID column is the index...
    df_test.to_csv(output_parcel_classes_test_csv)      # The ID column is the index...

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')