# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.

@author: Pieter Roggemans
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import cropclassification.helpers.config_helper as conf

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get the logger, at the moment just use the root logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(df_train: pd.DataFrame,
          output_classifier_filepath: str):
    """
    Train a classifier and output the trained classifier to the output file.

    Args
        df_train: pandas DataFrame containing the train data. Columns:
            * global_settings.id_column: the id of the parcel
            * global_settings.class_column: the class of the parcel
            * ... all columns that will be used as classification data
        output_classifier_filepath: the filepath where the classifier can be written
    """

    df_train_classes = df_train[conf.csv['class_column']]
    cols_to_keep = df_train.columns.difference([conf.csv['id_column'], conf.csv['class_column']])
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

def predict_proba(df_input_parcel: pd.DataFrame,
                  input_classifier_filepath: str,
                  output_parcel_predictions_csv: str):
    """
    Predict the probabilities for all input data using the classifier provided and write it
    to the output file.

    Args
        df_input_parcel: pandas DataFrame containing the data to classify. Columns:
            * global_settings.id_column: the id of the parcel.
            * global_settings.class_column: the class of the parcel. Isn't really used.
            * ... all columns that will be used as classification data.
        output_classifier_filepath: the filepath where the classifier can be written.
    """

    # Some basic checks that input is ok
    if(conf.csv['id_column'] not in df_input_parcel.columns
       or conf.csv['class_column'] not in df_input_parcel.columns):
        message = f"Columns {conf.csv['id_column']} and {conf.csv['class_column']} are mandatory for input parameter df_input!"
        logger.critical(message)
        raise Exception(message)

    # Now do final preparation for the classification
    df_input_classes = df_input_parcel[conf.csv['class_column']]
    cols_to_keep = df_input_parcel.columns.difference([conf.csv['id_column'], conf.csv['class_column']])
    df_input_data = df_input_parcel[cols_to_keep]

    logger.info(f"Train file processed and rows with missing data removed, data shape: {df_input_data.shape}, labels shape: {df_input_classes.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for training data: {df_input_data.columns}")

    # Load the classifier
    classifier = joblib.load(input_classifier_filepath)
    logger.info(f"Classifier has the following columns: {classifier.classes_}")

    logger.info(f"Predict classes with probabilities: {len(df_input_parcel)} rows")
    class_proba = classifier.predict_proba(df_input_data)
    logger.info(f"Predict classes with probabilities ready")

    # Convert probabilities to dataframe, combine with input data and write to file
    id_class_proba = np.concatenate([df_input_parcel[[conf.csv['id_column'], conf.csv['class_column']]].values, class_proba], axis=1)
    cols = [conf.csv['id_column'], conf.csv['class_column']]
    cols.extend(classifier.classes_)
    df_proba = pd.DataFrame(id_class_proba, columns=cols)

    return df_proba

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
