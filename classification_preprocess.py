# -*- coding: utf-8 -*-
"""
Module with helper functions to preprocess the data to use for the classification.

@author: Pieter Roggemans
"""

import logging
import os
import pandas as pd
import classification_preprocess_BEFL as befl
import global_settings as gs

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Some constants to choose the balancing strategy to use to crete the training set
BALANCING_STRATEGY_NONE = 'BALANCE_NONE'
BALANCING_STRATEGY_MEDIUM = 'BALANCE_MEDIUM'
BALANCING_STRATEGY_EQUAL = 'BALANCE_EQUAL'

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_input(input_parcel_filepath: str,
                  input_filetype: str,
                  input_parcel_pixcount_csv: str,
                  input_classtype_to_prepare: str,
                  output_parcel_filepath: str,
                  force: bool = False):
    """
    Prepare a raw input file by eg. adding the classification classes to use for the
    classification,...
    """

    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_parcel_filepath) is True:
        logger.warning(f"prepare_input: output file already exists and force == False, so stop: {output_parcel_filepath}")
        return

    if input_filetype == 'BEFL':
        df_parceldata = befl.prepare_input(input_parcel_filepath=input_parcel_filepath,
                                           input_classtype_to_prepare=input_classtype_to_prepare)
    else:
        message = f"Unknown value for parameter input_filetype: {input_filetype}"
        logger.critical(message)
        raise Exception(message)

    # Load pixcount data and join it
    logger.info(f'Read pixcount file {input_parcel_pixcount_csv}')
    df_pixcount = pd.read_csv(input_parcel_pixcount_csv)
    logger.debug(f'Read pixcount file ready, shape: {df_pixcount.shape}')
    df_pixcount.set_index(gs.id_column, inplace=True)

    df_parceldata.set_index(gs.id_column, inplace=True)
    df_parceldata = df_parceldata.join(df_pixcount[gs.pixcount_s1s2_column], how='left')

    # Export result to file
    output_ext = os.path.splitext(output_parcel_filepath)[1]
    for column in df_parceldata.columns:
        # if the output asked is a csv... we don't need the geometry...
        if column == 'geometry' and output_ext == '.csv':
            df_parceldata.drop(column, axis=1, inplace=True)

    logger.info(f'Write output to {output_parcel_filepath}')
    if output_ext == '.csv':         # If extension is csv, write csv (=a lot faster!)
        df_parceldata.to_csv(output_parcel_filepath)
    else:
        df_parceldata.to_file(output_parcel_filepath, index=False)

def create_train_test_sample(input_parcel_csv: str,
                             output_parcel_train_csv: str,
                             output_parcel_test_csv: str,
                             balancing_strategy: str,
                             force: bool = False):
    """ Create a seperate train and test sample from the general input file. """

    # If force == False Check and the output files exist already, stop.
    if(force is False
            and os.path.exists(output_parcel_train_csv) is True
            and os.path.exists(output_parcel_test_csv) is True):
        logger.warning(f"create_train_test_sample: output files already exist and force == False, so stop: {output_parcel_train_csv}, {output_parcel_test_csv}")
        return

    # Load input data...
    logger.info(f'Start create_train_test_sample with balancing_strategy {balancing_strategy}')
    logger.info(f'Read input file {input_parcel_csv}')
    df_in = pd.read_csv(input_parcel_csv)
    logger.debug(f'Read input file ready, shape: {df_in.shape}')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_in.groupby(gs.class_column, as_index=False).size()
        logger.info(f'Number of elements per classname in input dataset:\n{count_per_class}')

    # The test dataset should be as representative as possible for the entire dataset, so create
    # this first as a 20% sample of each class without any additional checks...
    # Remark: group_keys=False evades that apply creates an extra index-level of the groups above
    #         the data and evades having to do .reset_index(level=CLASS_COLUMN_NAME, drop=True)
    #         to get rid of the group level
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
        logger.info(f"Number of elements per classname for train dataset, before balancing:\n{count_per_class}")

    # Depending on the balancing_strategy, use different way to get a training sample
    if balancing_strategy == BALANCING_STRATEGY_NONE:
        # Just use 25% of all non-test data as train data -> 25% of 80% of data -> 20% of all data
        # will be training date
        # Remark: - this is very unbalanced, eg. classes with 10.000 times the input size than other
        #           classes
        #         - this results in a relatively high accuracy in overall numbers, but the small
        #           classes are not detected at all
        df_train = (df_train_base
                    .groupby(gs.class_column, group_keys=False)
                    .apply(pd.DataFrame.sample, frac=0.15))

    elif balancing_strategy == BALANCING_STRATEGY_MEDIUM:
        # Balance the train data, but still use some larger samples for the classes that have a lot
        # of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall, and also the
        #         smaller classes give some results with upper limit of 4000 results significantly
        #         less good.

        # For the larger classes, favor them by leaving the samples larger but cap at upper_limit
        upper_limit = 10000
        lower_limit = 1000
        logger.info(f"Cap over {upper_limit}, keep the full number of training sample till {lower_limit}, samples smaller than that are oversampled")
        df_train = (df_train_base.groupby(gs.class_column).filter(lambda x: len(x) >= upper_limit)
                    .groupby(gs.class_column, group_keys=False)
                    .apply(pd.DataFrame.sample, upper_limit))
        # Middle classes use the number as they are
        df_train = df_train.append(df_train_base
                                   .groupby(gs.class_column).filter(lambda x: len(x) < upper_limit)
                                   .groupby(gs.class_column).filter(lambda x: len(x) >= lower_limit))
        # For smaller classes, oversample...
        df_train = df_train.append(df_train_base
                                   .groupby(gs.class_column).filter(lambda x: len(x) < lower_limit)
                                   .groupby(gs.class_column, group_keys=False)
                                   .apply(pd.DataFrame.sample, lower_limit, replace=True))

    elif balancing_strategy == BALANCING_STRATEGY_EQUAL:
        # In theory the most logical way to balance: make sure all classes have the same amount of
        # training data by undersampling the largest classes and oversampling the small classes.
        df_train = (df_train_base.groupby(gs.class_column, group_keys=False)
                    .apply(pd.DataFrame.sample, 2000, replace=True))

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
    df_train.to_csv(output_parcel_train_csv, index=False)    # The ID column is the index...
    df_test.to_csv(output_parcel_test_csv, index=False)      # The ID column is the index...

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
