# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.
"""

import logging
import os

import pandas as pd

import cropclassification.predict.classification_sklearn as class_core
import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train_test_predict(input_parcel_train_filepath: str,
                       input_parcel_test_filepath: str,
                       input_parcel_all_filepath: str,
                       input_parcel_classification_data_filepath: str,
                       output_classifier_filepath: str,
                       output_predictions_test_filepath: str,
                       output_predictions_all_filepath: str,
                       force: bool = False):
    """ Train a classifier, test it and do full predictions.

    Args
        input_parcel_classes_train_filepath: the list of parcels with classes to train the classifier, without data!
        input_parcel_classes_test_filepath: the list of parcels with classes to test the classifier, without data!
        input_parcel_classes_all_filepath: the list of parcels with classes that need to be classified, without data!
        input_parcel_classification_data_filepath: the data to be used for the classification for all parcels.
        output_classifier_filepath: the file path where to save the classifier.
        output_predictions_test_filepath: the file path where to save the test predictions.
        output_predictions_all_filepath: the file path where to save the predictions for all parcels.
        force: if True, overwrite all existing output files, if False, don't overwrite them.
    """

    logger.info("train_test_predict: Start")

    if(force is False
       and os.path.exists(output_classifier_filepath)
       and os.path.exists(output_predictions_test_filepath)
       and os.path.exists(output_predictions_all_filepath)):
        logger.warning(f"predict: output files exist and force is False, so stop: {output_classifier_filepath}, {output_predictions_test_filepath}, {output_predictions_all_filepath}")
        return

    # Read the classification data from the csv so we can pass it on to the other functione to improve performance...
    logger.info(f"Read classification data file: {input_parcel_classification_data_filepath}")
    df_input_parcel_classification_data = pdh.read_file(input_parcel_classification_data_filepath)
    
    if df_input_parcel_classification_data.index.name != conf.columns['id']: 
        df_input_parcel_classification_data.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read classification data file ready')

    # Train the classification
    train(input_parcel_train_filepath=input_parcel_train_filepath,
          input_parcel_classification_data_filepath=input_parcel_classification_data_filepath,
          output_classifier_filepath=output_classifier_filepath,
          force=force,
          df_input_parcel_classification_data=df_input_parcel_classification_data)

    # Predict the test parcels
    predict(input_parcel_filepath=input_parcel_test_filepath,
            input_parcel_classification_data_filepath=input_parcel_classification_data_filepath,
            input_classifier_filepath=output_classifier_filepath,
            output_predictions_filepath=output_predictions_test_filepath,
            force=force,
            df_input_parcel_classification_data=df_input_parcel_classification_data)

    # Predict all parcels
    predict(input_parcel_filepath=input_parcel_all_filepath,
            input_parcel_classification_data_filepath=input_parcel_classification_data_filepath,
            input_classifier_filepath=output_classifier_filepath,
            output_predictions_filepath=output_predictions_all_filepath,
            force=force,
            df_input_parcel_classification_data=df_input_parcel_classification_data)

def train(input_parcel_train_filepath: str,
          input_parcel_classification_data_filepath: str,
          output_classifier_filepath: str,
          force: bool = False,
          df_input_parcel_classification_data: pd.DataFrame = None):
    """ Train a classifier and test it by predicting the test cases. """

    logger.info("train_and_test: Start")
    if(force is False
       and os.path.exists(output_classifier_filepath)):
        logger.warning(f"predict: classifier already exist and force == False, so don't retrain: {output_classifier_filepath}")
        return

    # If the classification data isn't passed as dataframe, read it from the csv
    if df_input_parcel_classification_data is None:
        logger.info(f"Read classification data file: {input_parcel_classification_data_filepath}")
        df_input_parcel_classification_data = pdh.read_file(input_parcel_classification_data_filepath)
        if df_input_parcel_classification_data.index.name != conf.columns['id']:
            df_input_parcel_classification_data.set_index(conf.columns['id'], inplace=True)
        logger.debug('Read classification data file ready')

    # First train the classifier if needed
    logger.info(f"Read train file: {input_parcel_train_filepath}")
    df_train = pdh.read_file(input_parcel_train_filepath, 
                             columns=[conf.columns['id'], conf.columns['class']])
    if df_train.index.name != conf.columns['id']:
        df_train.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read train file ready')

    # Join the columns of df_classification_data that aren't yet in df_train
    logger.info("Join train sample with the classification data")
    df_train = (df_train.join(df_input_parcel_classification_data, how='inner'))

    class_core.train(df_train=df_train, output_classifier_filepath=output_classifier_filepath)

def predict(input_parcel_filepath: str,
            input_parcel_classification_data_filepath: str,
            input_classifier_filepath: str,
            output_predictions_filepath: str,
            force: bool = False,
            df_input_parcel_classification_data: pd.DataFrame = None):
    """ Predict the classes for the input data. """

    # If force is False, and the output file exist already, return
    if(force is False
       and os.path.exists(output_predictions_filepath)):
        logger.warning(f"predict: predictions output file already exists and force is false, so stop: {output_predictions_filepath}")
        return

    # Read the input parcels
    logger.info(f"Read input file: {input_parcel_filepath}")
    df_input_parcel = pdh.read_file(input_parcel_filepath,
                                    columns=[conf.columns['id'], conf.columns['class']])
    if df_input_parcel.index.name != conf.columns['id']:
        df_input_parcel.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read train file ready')

    # For parcels of a class that should be ignored, don't predict
    df_input_parcel = df_input_parcel.loc[~df_input_parcel[conf.columns['class']]
                                           .isin(conf.marker.getlist('classes_to_ignore'))]

    # If the classification data isn't passed as dataframe, read it from the csv
    if df_input_parcel_classification_data is None:
        logger.info(f"Read classification data file: {input_parcel_classification_data_filepath}")
        df_input_parcel_classification_data = pdh.read_file(input_parcel_classification_data_filepath)
        if df_input_parcel_classification_data.index.name != conf.columns['id']:
            df_input_parcel_classification_data.set_index(conf.columns['id'], inplace=True)
        logger.debug('Read classification data file ready')

    # Join the data to send to prediction logic...
    logger.info("Join train sample with the classification data")
    df_input_parcel_for_predict = df_input_parcel.join(df_input_parcel_classification_data, how='inner')

    class_core.predict_proba(df_input_parcel=df_input_parcel_for_predict,
                             input_classifier_filepath=input_classifier_filepath,
                             output_parcel_predictions_filepath=output_predictions_filepath)

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
