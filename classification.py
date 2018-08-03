# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.

@author: Pieter Roggemans
"""

import logging
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import global_settings as gs

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get the logger, at the moment just use the root logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train_and_test(input_parcel_classes_train_csv: str
                   , input_parcel_classes_test_csv: str
                   , input_parcel_classification_data_csv: str
                   , output_classifier_filepath: str
                   , output_test_predictions_csv: str
                   , force: bool = False):
    """ Train a classifier and test it by predicting the test cases. """

    # If force == False Check and the output file exists already, stop.
    if(force is False
       and os.path.exists(output_classifier_filepath)
       and os.path.exists(output_test_predictions_csv)):
        logger.warning(f"predict: output files already exist and force == False, so stop: {output_classifier_filepath}, {output_test_predictions_csv}")
        return

#    logger.setLevel(logging.DEBUG)
    logger.info("train_and_test: Start")

    logger.info(f"Read train file: {input_parcel_classes_train_csv}")
    df_train = pd.read_csv(input_parcel_classes_train_csv, low_memory=False)
    logger.debug('Read train file ready')

    logger.info(f"Read classification data file: {input_parcel_classification_data_csv}")
    df_classification_data = pd.read_csv(input_parcel_classification_data_csv, low_memory=False)
    df_classification_data.set_index(gs.id_column, inplace=True)
    logger.debug('Read classification data file ready')

    # Join the columns of df_classification_data that aren't yet in df_train
    logger.info("Join train sample with the classification data")
    df_train = (df_train[[gs.id_column, gs.class_column]]
                .join(df_classification_data, how='inner', on=gs.id_column))

    # First train the classifier
    _train(df_train=df_train, output_classifier_filepath=output_classifier_filepath, force=force)

    logger.info(f"Read test file: {input_parcel_classes_test_csv}")
    df_test = pd.read_csv(input_parcel_classes_test_csv, low_memory=False)

    logger.info("Join test sample with the classification data")
    df_test = (df_test[[gs.id_column, gs.class_column]]
               .join(df_classification_data, how='inner', on=gs.id_column))

    logger.info(f"Test file processed and rows with missing data removed, data shape: {df_test.shape}")

    _predict(df_input=df_test
             , input_classifier_filepath=output_classifier_filepath
             , output_parcel_predictions_csv=output_test_predictions_csv
             , force=force)

def predict(input_parcel_classes_csv: str
            , input_parcel_classification_data_csv: str
            , input_classifier_filepath: str
            , output_predictions_csv: str
            , force: bool = False):
    """ Predict the classes for the input data. """

    # If force is False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_predictions_csv):
        logger.warning(f"predict: output file already exists and force is False, so stop: {output_predictions_csv}")
        return

    logger.info(f"Read input file: {input_parcel_classes_csv}")
    df_in = pd.read_csv(input_parcel_classes_csv, low_memory=False)
    logger.debug('Read train file ready')

    logger.info(f"Read classification data file: {input_parcel_classification_data_csv}")
    df_classification_data = pd.read_csv(input_parcel_classification_data_csv, low_memory=False)
    df_classification_data.set_index(gs.id_column, inplace=True)
    logger.debug('Read classification data file ready')

    # Join the columns of df_classification_data that aren't yet in df_train
    logger.info("Join train sample with the classification data")
    df_in = (df_in[[gs.id_column, gs.class_column]]
             .join(df_classification_data, how='inner', on=gs.id_column))

    _predict(df_input=df_in
             , input_classifier_filepath=input_classifier_filepath
             , output_parcel_predictions_csv=output_predictions_csv
             , force=force)

def _train(df_train: pd.DataFrame
           , output_classifier_filepath: str
           , force: bool = False):
    """
    Train a classifier, apply it to the test dataset and write the results of the classification
    to the output file.
    """

    # If force is False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_classifier_filepath):
        logger.warning(f"train: output file already exists and force == False, so stop: {output_classifier_filepath}")
        return

    df_train_classes = df_train[gs.class_column]
    cols_to_keep = df_train.columns.difference([gs.id_column, gs.class_column])
    df_train_data = df_train[cols_to_keep]

    logger.info(f"Train file processed and rows with missing data removed, data shape: {df_train_data.shape}, labels shape: {df_train_classes.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for training data: {df_train_data.columns}")

    logger.info('Start training')

    # Using almost all defaults for the classifier seems to work best...
    classifier = MLPClassifier(max_iter=100, hidden_layer_sizes=(150, ))
    logger.info(f"Classifier info:\n{classifier}")
    classifier.fit(df_train_data, df_train_classes)

    # Write the learned model to a file...
    logger.info(f"Write the learned model file to {output_classifier_filepath}")
    joblib.dump(classifier, output_classifier_filepath)

def _predict(df_input: pd.DataFrame
             , input_classifier_filepath: str
             , output_parcel_predictions_csv: str
             , force: bool = False):
    """ Function that does the actual prediction for all input data. """

    # If force is False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_parcel_predictions_csv):
        logger.warning(f"predict: output file already exists and force is False, so stop: {output_parcel_predictions_csv}")
        return

    # Some basic checks that input is ok...
    if (gs.id_column not in df_input.columns
            or gs.class_column not in df_input.columns):
        message = f"Columns {gs.id_column} and {gs.class_column} are mandatory for input parameter df_input!"
        logger.critical(message)
        raise Exception(message)

    # Now do final preparation for the classification
    df_input_classes = df_input[gs.class_column]
    cols_to_keep = df_input.columns.difference([gs.id_column, gs.class_column])
    df_input_data = df_input[cols_to_keep]

    logger.info(f"Train file processed and rows with missing data removed, data shape: {df_input_data.shape}, labels shape: {df_input_classes.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for training data: {df_input_data.columns}")

    # Load the classifier
    classifier = joblib.load(input_classifier_filepath)

    logger.info(f"Predict classes with probabilities: {len(df_input)} rows")
    class_proba = classifier.predict_proba(df_input_data)
    logger.info(f"Predict classes with probabilities ready")

    # Get the top 3 predictions for each row
    # First get the indeces of the top 3 predictions for each row
    # Remark: argsort sorts ascending, so we need to take:
    #     - "[:,": for all rows
    #     - ":-4": the last 3 elements of the values
    #     - ":-1]": and than reverse the order with a negative step
    top3_pred_classes_idx = np.argsort(class_proba, axis=1)[:, :-4:-1]
    # Convert the indeces to classes
    top3_pred_classes = np.take(classifier.classes_, top3_pred_classes_idx)
    # Get the values of the top 3 predictions
    top3_pred_values = np.sort(class_proba, axis=1)[:, :-4:-1]
    # Concatenate both
    top3_pred = np.concatenate([top3_pred_classes, top3_pred_values], axis=1)
    # Concatenate the ids, the classes and the top3 predictions
    id_class_top3 = np.concatenate([df_input[[gs.id_column, gs.class_column]].values, top3_pred]
                                   , axis=1)

    # Convert to dataframe, combine with input data and write to file
    df_top3 = pd.DataFrame(id_class_top3
                           , columns=[gs.id_column, gs.class_column, 'pred1', 'pred2', 'pred3'
                                      , 'pred1_prob', 'pred2_prob', 'pred3_prob'])

    # Add the consolidated prediction
    def calculate_consolidated_prediction(row):
        # For some reason the row['pred2_prob'] is sometimes seen as string, and so 2* gives a
        # repetition of the string value instead of a mathematic multiplication... so cast to float!
        if float(row['pred1_prob']) >= 2.0 * float(row['pred2_prob']):
            return row['pred1']
        else:
            return 'DOUBT'

    values = df_top3.apply(calculate_consolidated_prediction, axis=1)
    df_top3.insert(loc=2, column='pred_consolidated', value=values)

    logger.info("Write top3 predictions to file")
    df_top3.to_csv(output_parcel_predictions_csv
                   , float_format='%.10f', encoding='utf-8', index=False)

    # Convert detailed results to dataframe, combine with input data and write to file
    # Combine the ids, the classes and the probabilities
    id_class_proba = np.concatenate([df_input[[gs.id_column, gs.class_column]].values, class_proba], axis=1)
    cols = [gs.id_column, gs.class_column].append(classifier.classes_)
    df_proba = pd.DataFrame(id_class_proba, columns=cols)
    df_proba.insert(loc=2, column='pred1', value=df_top3['pred1'])

    logger.info("Write detailed predictions to file")
    filepath_noext, ext = os.path.splitext(output_parcel_predictions_csv)
    df_proba.to_csv(f"{filepath_noext}_detailed{ext}"
                    , float_format='%.10f', encoding='utf-8', index=False)

    logger.info("Predictions written to file")

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
