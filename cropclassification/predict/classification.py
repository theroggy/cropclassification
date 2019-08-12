# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.
"""

import logging
import os

import pandas as pd

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
                       output_classifier_basefilepath: str,
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
       #and os.path.exists(output_classifier_basefilepath)
       and os.path.exists(output_predictions_test_filepath)
       and os.path.exists(output_predictions_all_filepath)):
        logger.warning(f"predict: output files exist and force is False, so stop: {output_classifier_filepath}, {output_predictions_test_filepath}, {output_predictions_all_filepath}")
        return

    # Read the classification data from the csv so we can pass it on to the other functione to improve performance...
    logger.info(f"Read classification data file: {input_parcel_classification_data_filepath}")
    input_parcel_classification_data_df = pdh.read_file(input_parcel_classification_data_filepath)    
    if input_parcel_classification_data_df.index.name != conf.columns['id']: 
        input_parcel_classification_data_df.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read classification data file ready')

    # Train the classification
    output_classifier_filepath = train(input_parcel_train_filepath=input_parcel_train_filepath,
          input_parcel_test_filepath=input_parcel_test_filepath,
          input_parcel_classification_data_filepath=input_parcel_classification_data_filepath,
          output_classifier_basefilepath=output_classifier_basefilepath,
          force=force,
          input_parcel_classification_data_df=input_parcel_classification_data_df)

    # Predict the test parcels
    predict(input_parcel_filepath=input_parcel_test_filepath,
            input_parcel_classification_data_filepath=input_parcel_classification_data_filepath,
            input_classifier_basefilepath=output_classifier_basefilepath,
            input_classifier_filepath=output_classifier_filepath,
            output_predictions_filepath=output_predictions_test_filepath,
            force=force,
            input_parcel_classification_data_df=input_parcel_classification_data_df)

    # Predict all parcels
    predict(input_parcel_filepath=input_parcel_all_filepath,
            input_parcel_classification_data_filepath=input_parcel_classification_data_filepath,
            input_classifier_basefilepath=output_classifier_basefilepath,
            input_classifier_filepath=output_classifier_filepath,
            output_predictions_filepath=output_predictions_all_filepath,
            force=force,
            input_parcel_classification_data_df=input_parcel_classification_data_df)

def train(input_parcel_train_filepath: str,
          input_parcel_test_filepath: str,
          input_parcel_classification_data_filepath: str,
          output_classifier_basefilepath: str,
          force: bool = False,
          input_parcel_classification_data_df: pd.DataFrame = None):
    """ Train a classifier and test it by predicting the test cases. """

    logger.info("train_and_test: Start")
    if(force is False
       and os.path.exists(output_classifier_basefilepath)):
        logger.warning(f"predict: classifier already exist and force == False, so don't retrain: {output_classifier_basefilepath}")
        return output_classifier_basefilepath

    # If the classification data isn't passed as dataframe, read it from file
    if input_parcel_classification_data_df is None:
        logger.info(f"Read classification data file: {input_parcel_classification_data_filepath}")
        input_parcel_classification_data_df = pdh.read_file(input_parcel_classification_data_filepath)
        if input_parcel_classification_data_df.index.name != conf.columns['id']:
            input_parcel_classification_data_df.set_index(conf.columns['id'], inplace=True)
        logger.debug('Read classification data file ready')

    # Read the train parcels
    logger.info(f"Read train file: {input_parcel_train_filepath}")
    train_df = pdh.read_file(input_parcel_train_filepath, 
                             columns=[conf.columns['id'], conf.columns['class']])
    if train_df.index.name != conf.columns['id']:
        train_df.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read train file ready')

    # Join the columns of input_parcel_classification_data_df that aren't yet in train_df
    logger.info("Join train sample with the classification data")
    train_df = (train_df.join(input_parcel_classification_data_df, how='inner'))

    # Read the test/validation data
    logger.info(f"Read test file: {input_parcel_test_filepath}")
    test_df = pdh.read_file(input_parcel_test_filepath, 
                            columns=[conf.columns['id'], conf.columns['class']])
    if test_df.index.name != conf.columns['id']:
        test_df.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read test file ready')

    # Join the columns of input_parcel_classification_data_df that aren't yet in test_df
    logger.info("Join test sample with the classification data")
    test_df = (test_df.join(input_parcel_classification_data_df, how='inner'))

    # Train
    if conf.classifier['classifier_type'].lower() == 'keras_multilayer_perceptron':
        import cropclassification.predict.classification_keras as class_core_keras
        return class_core_keras.train(
                train_df=train_df, 
                test_df=test_df,
                output_classifier_basefilepath=output_classifier_basefilepath)
    else:
        import cropclassification.predict.classification_sklearn as class_core_sklearn
        return class_core_sklearn.train(
                train_df=train_df, 
                output_classifier_basefilepath=output_classifier_basefilepath)

def predict(input_parcel_filepath: str,
            input_parcel_classification_data_filepath: str,
            input_classifier_basefilepath: str,
            input_classifier_filepath: str,
            output_predictions_filepath: str,
            force: bool = False,
            input_parcel_classification_data_df: pd.DataFrame = None):
    """ Predict the classes for the input data. """

    # If force is False, and the output file exist already, return
    if(force is False
       and os.path.exists(output_predictions_filepath)):
        logger.warning(f"predict: predictions output file already exists and force is false, so stop: {output_predictions_filepath}")
        return

    # Read the input parcels
    logger.info(f"Read input file: {input_parcel_filepath}")
    input_parcel_df = pdh.read_file(input_parcel_filepath,
            columns=[conf.columns['id'], conf.columns['class'], conf.columns['class_declared']])
    if input_parcel_df.index.name != conf.columns['id']:
        input_parcel_df.set_index(conf.columns['id'], inplace=True)
    logger.debug('Read train file ready')

    # For parcels of a class that should be ignored, don't predict
    input_parcel_df = input_parcel_df.loc[~input_parcel_df[conf.columns['class_declared']]
                                           .isin(conf.marker.getlist('classes_to_ignore'))]

    # If the classification data isn't passed as dataframe, read it from the csv
    if input_parcel_classification_data_df is None:
        logger.info(f"Read classification data file: {input_parcel_classification_data_filepath}")
        input_parcel_classification_data_df = pdh.read_file(input_parcel_classification_data_filepath)
        if input_parcel_classification_data_df.index.name != conf.columns['id']:
            input_parcel_classification_data_df.set_index(conf.columns['id'], inplace=True)
        logger.debug('Read classification data file ready')

    # Join the data to send to prediction logic...
    logger.info("Join input parcels with the classification data")
    input_parcel_for_predict_df = input_parcel_df.join(input_parcel_classification_data_df, how='inner')

    # Predict!
    logger.info(f"Predict using this model: {input_classifier_filepath}")
    if conf.classifier['classifier_type'].lower() == 'keras_multilayer_perceptron':
        import cropclassification.predict.classification_keras as class_core_keras
        class_core_keras.predict_proba(
                parcel_df=input_parcel_for_predict_df,
                classifier_basefilepath=input_classifier_basefilepath,
                classifier_filepath=input_classifier_filepath,
                output_parcel_predictions_filepath=output_predictions_filepath)
    else:
        import cropclassification.predict.classification_sklearn as class_core_sklearn
        class_core_sklearn.predict_proba(
                parcel_df=input_parcel_for_predict_df,
                classifier_basefilepath=input_classifier_basefilepath,
                classifier_filepath=input_classifier_filepath,
                output_parcel_predictions_filepath=output_predictions_filepath)

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
