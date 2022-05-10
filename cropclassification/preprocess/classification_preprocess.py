# -*- coding: utf-8 -*-
"""
Module with helper functions to preprocess the data to use for the classification.
"""

import logging
import os
from pathlib import Path
import shutil

import pandas as pd

import cropclassification.preprocess.classification_preprocess_BEFL as befl
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

def prepare_input(input_parcel_filepath: Path,
                  input_parcel_filetype: str,
                  input_parcel_pixcount_filepath: Path,
                  classtype_to_prepare: str,
                  classes_refe_filepath: Path,
                  output_parcel_filepath: Path,
                  force: bool = False):
    """
    Prepare a raw input file by eg. adding the classification classes to use for the
    classification,...
    """

    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_parcel_filepath) is True:
        logger.warning(f"prepare_input: output file already exists and force == False, so stop: {output_parcel_filepath}")
        return

    # If it exists, copy the refe file to the run dir, so we always keep knowing which refe was used
    if classes_refe_filepath is not None:
        shutil.copy(classes_refe_filepath, output_parcel_filepath.parent)        

    if input_parcel_filetype == 'BEFL':
        # classes_refe_filepath must exist for BEFL!
        if not classes_refe_filepath.exists():
            raise Exception(f"Input classes file doesn't exist: {classes_refe_filepath}")
        
        df_parceldata = befl.prepare_input(input_parcel_filepath=input_parcel_filepath,
                                           classtype_to_prepare=classtype_to_prepare,
                                           classes_refe_filepath=classes_refe_filepath,
                                           output_dir=output_parcel_filepath.parent)
    else:
        message = f"Unknown value for parameter input_parcel_filetype: {input_parcel_filetype}"
        logger.critical(message)
        raise Exception(message)

    # Load pixcount data and join it
    logger.info(f"Read pixcount file {input_parcel_pixcount_filepath}")
    df_pixcount = pdh.read_file(input_parcel_pixcount_filepath)
    logger.debug(f"Read pixcount file ready, shape: {df_pixcount.shape}")
    if df_pixcount.index.name != conf.columns['id']: 
        df_pixcount.set_index(conf.columns['id'], inplace=True)

    df_parceldata.set_index(conf.columns['id'], inplace=True)
    df_parceldata = df_parceldata.join(df_pixcount[conf.columns['pixcount_s1s2']], how='left')
    df_parceldata[conf.columns['pixcount_s1s2']].fillna(value=0, inplace=True)

    # Export result to file
    output_ext = os.path.splitext(output_parcel_filepath)[1]
    for column in df_parceldata.columns:
        # if the output asked is a csv... we don't need the geometry...
        if column == conf.columns['geom'] and output_ext == '.csv':
            df_parceldata.drop(column, axis=1, inplace=True)

    logger.info(f"Write output to {output_parcel_filepath}")
    # If extension is not .shp, write using pandas (=a lot faster!)
    if output_ext.lower() != '.shp': 
        pdh.to_file(df_parceldata, output_parcel_filepath)
    else:
        df_parceldata.to_file(output_parcel_filepath, index=False)

def create_train_test_sample(input_parcel_filepath: Path,
                             output_parcel_train_filepath: Path,
                             output_parcel_test_filepath: Path,
                             balancing_strategy: str,
                             force: bool = False):
    """ Create a seperate train and test sample from the general input file. """

    # If force == False Check and the output files exist already, stop.
    if(force is False
            and output_parcel_train_filepath.exists() is True
            and output_parcel_test_filepath.exists() is True):
        logger.warning(f"create_train_test_sample: output files already exist and force == False, so stop: {output_parcel_train_filepath}, {output_parcel_test_filepath}")
        return

    # Load input data...
    logger.info(f"Start create_train_test_sample with balancing_strategy {balancing_strategy}")
    logger.info(f"Read input file {input_parcel_filepath}")
    df_in = pdh.read_file(input_parcel_filepath)
    logger.debug(f"Read input file ready, shape: {df_in.shape}")

    # Init some many-used variables from config
    class_balancing_column = conf.columns['class_balancing']
    class_column = conf.columns['class']

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_in.groupby(class_balancing_column, as_index=False).size()
        logger.info(f"Number of elements per classname in input dataset:\n{count_per_class}")

    # The test dataset should be as representative as possible for the entire dataset, so create
    # this first as a 20% sample of each class without any additional checks...
    # Remark: group_keys=False evades that apply creates an extra index-level of the groups above
    #         the data and evades having to do .reset_index(level=class_balancing_column_NAME, drop=True)
    #         to get rid of the group level
    df_test = df_in.groupby(class_balancing_column, group_keys=False).apply(pd.DataFrame.sample, frac=0.20)
    logger.debug(f"df_test after sampling 20% of data per class, shape: {df_test.shape}")

    # The candidate parcel for training are all non-test parcel
    df_train_base = df_in[~df_in.index.isin(df_test.index)]
    logger.debug(f"df_train_base after isin\n{df_train_base}")

    # Remove parcel with too few pixels from the train sample
    min_pixcount = conf.marker.getfloat('min_nb_pixels_train')
    df_train_base = df_train_base[df_train_base[conf.columns['pixcount_s1s2']] >= min_pixcount]
    logger.debug(f"Number of parcels in df_train_base after filter on pixcount >= {min_pixcount}: {len(df_train_base)}")

    # Some classes shouldn't be used for training... so remove them!
    logger.info(f"Remove 'classes_to_ignore_for_train' from train sample (= where {class_column} is in: {conf.marker.getlist('classes_to_ignore_for_train')}")
    df_train_base = df_train_base[~df_train_base[class_column].isin(conf.marker.getlist('classes_to_ignore_for_train'))]

    # All classes_to_ignore aren't meant for training either...
    logger.info(f"Remove 'classes_to_ignore' from train sample (= where {class_column} is in: {conf.marker.getlist('classes_to_ignore')}")
    df_train_base = df_train_base[~df_train_base[class_column].isin(conf.marker.getlist('classes_to_ignore'))]

    # Print the train base result before applying any balancing
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_train_base.groupby(class_balancing_column, as_index=False).size()
        logger.info(f"Number of elements per classname for train dataset, before balancing:\n{count_per_class}")

    # Depending on the balancing_strategy, use different way to get a training sample
    if balancing_strategy == 'BALANCING_STRATEGY_NONE':
        # Just use 25% of all non-test data as train data -> 25% of 80% of data -> 20% of all data
        # will be training date
        # Remark: - this is very unbalanced, eg. classes with 10.000 times the input size than other
        #           classes
        #         - this results in a relatively high accuracy in overall numbers, but the small
        #           classes are not detected at all
        df_train = (df_train_base
                    .groupby(class_balancing_column, group_keys=False)
                    .apply(pd.DataFrame.sample, frac=0.25))

    elif balancing_strategy == 'BALANCING_STRATEGY_MEDIUM':
        # Balance the train data, but still use some larger samples for the classes that have a lot
        # of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall, and also the
        #         smaller classes give some results with upper limit of 4000 results significantly
        #         less good.

        # For the larger classes, favor them by leaving the samples larger but cap at upper_limit
        upper_limit = 10000
        lower_limit = 1000
        logger.info(f"Cap over {upper_limit}, keep the full number of training sample till {lower_limit}, samples smaller than that are oversampled")
        df_train = (df_train_base.groupby(class_balancing_column).filter(lambda x: len(x) >= upper_limit)
                    .groupby(class_balancing_column, group_keys=False)
                    .apply(pd.DataFrame.sample, upper_limit))
        # Middle classes use the number as they are
        df_train = df_train.append(df_train_base
                                   .groupby(class_balancing_column).filter(lambda x: len(x) < upper_limit)
                                   .groupby(class_balancing_column).filter(lambda x: len(x) >= lower_limit))
        # For smaller classes, oversample...
        df_train = df_train.append(df_train_base
                                   .groupby(class_balancing_column).filter(lambda x: len(x) < lower_limit)
                                   .groupby(class_balancing_column, group_keys=False)
                                   .apply(pd.DataFrame.sample, lower_limit, replace=True))

    elif balancing_strategy == 'BALANCING_STRATEGY_MEDIUM2':
        # Balance the train data, but still use some larger samples for the classes that have a lot
        # of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall, and also the
        #         smaller classes give some results with upper limit of 4000 results significantly
        #         less good.

        # For the larger classes, leave the samples larger but cap
        cap_count_limit1 = 100000
        cap_train_limit1 = 30000
        logger.info(f"Cap balancing classes over {cap_count_limit1} to {cap_train_limit1}")
        df_train = (df_train_base
                    .groupby(class_balancing_column).filter(lambda x: len(x) >= cap_count_limit1)
                    .groupby(class_balancing_column, group_keys=False)
                    .apply(pd.DataFrame.sample, cap_train_limit1))
        cap_count_limit2 = 50000
        cap_train_limit2 = 20000
        logger.info(f"Cap balancing classes between {cap_count_limit2} and {cap_count_limit1} to {cap_train_limit2}")
        train_cap2_df = (df_train_base
                .groupby(class_balancing_column).filter(lambda x: len(x) < cap_count_limit1)
                .groupby(class_balancing_column).filter(lambda x: len(x) >= cap_count_limit2)
                .groupby(class_balancing_column, group_keys=False))
        if len(train_cap2_df) > 0:
            df_train = df_train.append(train_cap2_df.apply(pd.DataFrame.sample, cap_train_limit2))
        cap_count_limit3 = 20000
        cap_train_limit3 = 10000
        logger.info(f"Cap balancing classes between {cap_count_limit3} and {cap_count_limit2} to {cap_train_limit3}")
        df_train = df_train.append(df_train_base
                .groupby(class_balancing_column).filter(lambda x: len(x) < cap_count_limit2)
                .groupby(class_balancing_column).filter(lambda x: len(x) >= cap_count_limit3)
                .groupby(class_balancing_column, group_keys=False)
                .apply(pd.DataFrame.sample, cap_train_limit3))
        cap_count_limit4 = 10000
        cap_train_limit4 = 10000
        logger.info(f"Cap balancing classes between {cap_count_limit4} and {cap_count_limit3} to {cap_train_limit4}")
        df_train = df_train.append(df_train_base
                .groupby(class_balancing_column).filter(lambda x: len(x) < cap_count_limit3)
                .groupby(class_balancing_column).filter(lambda x: len(x) >= cap_count_limit4)
                .groupby(class_balancing_column, group_keys=False)
                .apply(pd.DataFrame.sample, cap_train_limit4))
        oversample_count = 1000
        # Middle classes use the number as they are
        logger.info(f"For classes between {cap_count_limit4} and {oversample_count}, just use all samples")
        df_train = df_train.append(df_train_base
                                   .groupby(class_balancing_column).filter(lambda x: len(x) < cap_count_limit4)
                                   .groupby(class_balancing_column).filter(lambda x: len(x) >= oversample_count))
        # For smaller classes, oversample...
        logger.info(f"For classes smaller than {oversample_count}, oversample to {oversample_count}")
        df_train = df_train.append(df_train_base
                                   .groupby(class_balancing_column).filter(lambda x: len(x) < oversample_count)
                                   .groupby(class_balancing_column, group_keys=False)
                                   .apply(pd.DataFrame.sample, oversample_count, replace=True))

    elif balancing_strategy == 'BALANCING_STRATEGY_PROPORTIONAL_GROUPS':
        # Balance the train data, but still use some larger samples for the classes that have a lot
        # of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall, and also the
        #         smaller classes give some results with upper limit of 4000 results significantly
        #         less good.

        # For the larger classes, leave the samples larger but cap
        upper_count_limit1 = 100000
        upper_train_limit1 = 30000
        logger.info(f"Cap balancing classes over {upper_count_limit1} to {upper_train_limit1}")
        df_train = (df_train_base
                    .groupby(class_balancing_column).filter(lambda x: len(x) >= upper_count_limit1)
                    .groupby(class_balancing_column, group_keys=False)
                    .apply(pd.DataFrame.sample, upper_train_limit1))
        upper_count_limit2 = 50000
        upper_train_limit2 = 20000
        logger.info(f"Cap balancing classes between {upper_count_limit2} and {upper_count_limit1} to {upper_train_limit2}")
        df_train = df_train.append(df_train_base
                .groupby(class_balancing_column).filter(lambda x: len(x) < upper_count_limit1)
                .groupby(class_balancing_column).filter(lambda x: len(x) >= upper_count_limit2)
                .groupby(class_balancing_column, group_keys=False)
                .apply(pd.DataFrame.sample, upper_train_limit2))
        upper_count_limit3 = 20000
        upper_train_limit3 = 10000
        logger.info(f"Cap balancing classes between {upper_count_limit3} and {upper_count_limit2} to {upper_train_limit3}")
        df_train = df_train.append(df_train_base
                .groupby(class_balancing_column).filter(lambda x: len(x) < upper_count_limit2)
                .groupby(class_balancing_column).filter(lambda x: len(x) >= upper_count_limit3)
                .groupby(class_balancing_column, group_keys=False)
                .apply(pd.DataFrame.sample, upper_train_limit3))
        upper_count_limit4 = 10000
        upper_train_limit4 = 5000
        logger.info(f"Cap balancing classes between {upper_count_limit4} and {upper_count_limit3} to {upper_train_limit4}")
        df_train = df_train.append(df_train_base
                .groupby(class_balancing_column).filter(lambda x: len(x) < upper_count_limit3)
                .groupby(class_balancing_column).filter(lambda x: len(x) >= upper_count_limit4)
                .groupby(class_balancing_column, group_keys=False)
                .apply(pd.DataFrame.sample, upper_train_limit4))

        # For smaller balancing classes, just use all samples
        df_train = df_train.append(
                df_train_base.groupby(class_balancing_column).filter(lambda x: len(x) < upper_count_limit4))

    elif balancing_strategy == 'BALANCING_STRATEGY_UPPER_LIMIT':
        # Balance the train data, but still use some larger samples for the classes that have a lot
        # of members in the input dataset
        # Remark: with the upper limit of 10.000 this gives still OK results overall, and also the
        #         smaller classes give some results with upper limit of 4000 results significantly
        #         less good.

        # For the larger classes, favor them by leaving the samples larger but cap at upper_limit
        upper_limit = 10000
        logger.info(f"Cap over {upper_limit}...")
        df_train = (df_train_base.groupby(class_balancing_column).filter(lambda x: len(x) >= upper_limit)
                    .groupby(class_balancing_column, group_keys=False)
                    .apply(pd.DataFrame.sample, upper_limit))
        # For smaller classes, just use all samples
        df_train = df_train.append(df_train_base
                                   .groupby(class_balancing_column).filter(lambda x: len(x) < upper_limit))

    elif balancing_strategy == 'BALANCING_STRATEGY_EQUAL':
        # In theory the most logical way to balance: make sure all classes have the same amount of
        # training data by undersampling the largest classes and oversampling the small classes.
        df_train = (df_train_base.groupby(class_balancing_column, group_keys=False)
                    .apply(pd.DataFrame.sample, 2000, replace=True))

    else:
        raise Exception(f"Unknown balancing strategy, STOP!: {balancing_strategy}")

    # Log the resulting numbers per class in the train sample
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_train.groupby(class_balancing_column, as_index=False).size()
        logger.info(f'Number of elements per class_balancing_column in train dataset:\n{count_per_class}')
        if class_balancing_column != class_column:
            count_per_class = df_train.groupby(class_column, as_index=False).size()
            logger.info(f'Number of elements per class_column in train dataset:\n{count_per_class}')

    # Log the resulting numbers per class in the test sample
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        count_per_class = df_test.groupby(class_balancing_column, as_index=False).size()
        logger.info(f'Number of elements per class_balancing_column in test dataset:\n{count_per_class}')
        if class_balancing_column != class_column:
            count_per_class = df_test.groupby(class_column, as_index=False).size()
            logger.info(f'Number of elements per class_column in test dataset:\n{count_per_class}')

    # Write to output files
    logger.info('Write the output files')
    df_train.set_index(conf.columns['id'], inplace=True)
    df_test.set_index(conf.columns['id'], inplace=True)
    pdh.to_file(df_train, output_parcel_train_filepath)    # The ID column is the index...
    pdh.to_file(df_test, output_parcel_test_filepath)      # The ID column is the index...

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
