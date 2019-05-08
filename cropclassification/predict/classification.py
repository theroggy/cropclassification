# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.

@author: Pieter Roggemans
"""

import logging
import os
import numpy as np
import pandas as pd
import cropclassification.predict.classification_sklearn as class_core
import cropclassification.helpers.config_helper as conf

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get the logger, at the moment just use the root logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train_test_predict(input_parcel_train_csv: str,
                       input_parcel_test_csv: str,
                       input_parcel_all_csv: str,
                       input_parcel_classification_data_csv: str,
                       output_classifier_filepath: str,
                       output_predictions_test_csv: str,
                       output_predictions_all_csv: str,
                       force: bool = False):
    """ Train a classifier, test it and do full predictions.

    Args
        input_parcel_classes_train_csv: the list of parcels with classes to train the classifier, without data!
        input_parcel_classes_test_csv: the list of parcels with classes to test the classifier, without data!
        input_parcel_classes_all_csv: the list of parcels with classes that need to be classified, without data!
        input_parcel_classification_data_csv: the data to be used for the classification for all parcels.
        output_classifier_filepath: the file path where to save the classifier.
        output_predictions_test_csv: the file path where to save the test predictions as csv.
        output_predictions_all_csv: the file path where to save the predictions for all parcels as csv.
        force: if True, overwrite all existing output files, if False, don't overwrite them.
    """

    logger.info("train_test_predict: Start")

    if(force is False
       and os.path.exists(output_classifier_filepath)
       and os.path.exists(output_predictions_test_csv)
       and os.path.exists(output_predictions_all_csv)):
        logger.warning(f"predict: output files exist and force is False, so stop: {output_classifier_filepath}, {output_predictions_test_csv}, {output_predictions_all_csv}")
        return

    # Read the classification data from the csv so we can pass it on to the other functione to improve performance...
    logger.info(f"Read classification data file: {input_parcel_classification_data_csv}")
    df_input_parcel_classification_data = pd.read_csv(input_parcel_classification_data_csv, low_memory=False)
    df_input_parcel_classification_data.set_index(conf.csv['id_column'], inplace=True)
    logger.debug('Read classification data file ready')

    # Train the classifiaction
    train(input_parcel_train_csv=input_parcel_train_csv,
          input_parcel_classification_data_csv=input_parcel_classification_data_csv,
          output_classifier_filepath=output_classifier_filepath,
          force=force,
          df_input_parcel_classification_data=df_input_parcel_classification_data)

    # Predict the test parcels
    predict(input_parcel_csv=input_parcel_test_csv,
            input_parcel_classification_data_csv=input_parcel_classification_data_csv,
            input_classifier_filepath=output_classifier_filepath,
            output_predictions_csv=output_predictions_test_csv,
            force=force,
            df_input_parcel_classification_data=df_input_parcel_classification_data)

    # Predict all parcels
    predict(input_parcel_csv=input_parcel_all_csv,
            input_parcel_classification_data_csv=input_parcel_classification_data_csv,
            input_classifier_filepath=output_classifier_filepath,
            output_predictions_csv=output_predictions_all_csv,
            force=force,
            df_input_parcel_classification_data=df_input_parcel_classification_data)

def train(input_parcel_train_csv: str,
          input_parcel_classification_data_csv: str,
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
        logger.info(f"Read classification data file: {input_parcel_classification_data_csv}")
        df_input_parcel_classification_data = pd.read_csv(input_parcel_classification_data_csv, low_memory=False)
        df_input_parcel_classification_data.set_index(conf.csv['id_column'], inplace=True)
        logger.debug('Read classification data file ready')

    # First train the classifier if needed
    logger.info(f"Read train file: {input_parcel_train_csv}")
    df_train = pd.read_csv(input_parcel_train_csv, low_memory=False)
    #df_train.set_index(gs.id_column, inplace=True)
    logger.debug('Read train file ready')

    # Join the columns of df_classification_data that aren't yet in df_train
    logger.info("Join train sample with the classification data")
    df_train = (df_train[[conf.csv['id_column'], conf.csv['class_column']]]
                .join(df_input_parcel_classification_data, how='inner', on=conf.csv['id_column']))

    class_core.train(df_train=df_train, output_classifier_filepath=output_classifier_filepath)

def predict(input_parcel_csv: str,
            input_parcel_classification_data_csv: str,
            input_classifier_filepath: str,
            output_predictions_csv: str,
            force: bool = False,
            df_input_parcel_classification_data: pd.DataFrame = None):
    """ Predict the classes for the input data. """

    # If force is False, and the output file doesn't exist yet, stop
    if(force is False
       and os.path.exists(output_predictions_csv)):
        logger.warning(f"predict: predictions output file already exists and force is false, so stop: {output_predictions_csv}")
        return

    logger.info(f"Read input file: {input_parcel_csv}")
    df_input_parcel = pd.read_csv(input_parcel_csv, low_memory=False)
    logger.debug('Read train file ready')

    # If the classification data isn't passed as dataframe, read it from the csv
    if df_input_parcel_classification_data is None:
        logger.info(f"Read classification data file: {input_parcel_classification_data_csv}")
        df_input_parcel_classification_data = pd.read_csv(input_parcel_classification_data_csv, low_memory=False)
        df_input_parcel_classification_data.set_index(conf.csv['id_column'], inplace=True)
        logger.debug('Read classification data file ready')

    # Prepare the data to send to prediction logic...
    logger.info("Join train sample with the classification data")
    df_input_parcel_for_predict = (df_input_parcel[[conf.csv['id_column'], conf.csv['class_column']]]
                                   .join(df_input_parcel_classification_data,
                                         how='inner', on=conf.csv['id_column']))

    df_proba = class_core.predict_proba(df_input_parcel=df_input_parcel_for_predict,
                                        input_classifier_filepath=input_classifier_filepath,
                                        output_parcel_predictions_csv=output_predictions_csv)

    # Calculate the top 3 predictions
    df_top3 = _get_top_3_prediction(df_proba)

    # Add the consolidated prediction
    def calculate_consolidated_prediction(row):
        # For some reason the row['pred2_prob'] is sometimes seen as string, and so 2* gives a
        # repetition of the string value instead of a mathematic multiplication... so cast to float!

# float(row['pred1_prob']) > 30
        if ((float(row['pred1_prob']) >= 2.0 * float(row['pred2_prob']))):
            return row[conf.csv['prediction_column']]
        else:
            return 'DOUBT'

    values = df_top3.apply(calculate_consolidated_prediction, axis=1)
    df_top3.insert(loc=2, column=conf.csv['prediction_cons_column'], value=values)

    # Make sure all input parcels are in the output. If there was no prediction, it means that there
    # was no data available for a classification, so set prediction to NODATA
    df_top3.set_index(conf.csv['id_column'], inplace=True)
    df_input_parcel.set_index(conf.csv['id_column'], inplace=True)

    # Add a column with the prediction status... and all parcels in df_top3 got a prediction
    df_top3[conf.csv['prediction_status']] = 'OK'
    df_top3.loc[(df_top3[conf.csv['prediction_cons_column']] == 'DOUBT'),
                conf.csv['prediction_status']] = 'DOUBT'

    cols_to_join = df_top3.columns.difference(df_input_parcel.columns)
    df_pred = df_input_parcel.join(df_top3[cols_to_join], how='left')
    df_pred[conf.csv['prediction_column']].fillna('NODATA', inplace=True)
    df_pred[conf.csv['prediction_cons_column']].fillna('NODATA', inplace=True)
    df_pred[conf.csv['prediction_status']].fillna('NODATA', inplace=True)

    logger.info(f"Columns of df_pred: {df_pred.columns}")
    # Parcels with too few pixels don't have a good accuracy and give many alfa errors...
    df_pred.loc[(df_pred[conf.csv['pixcount_s1s2_column']] <= conf.marker.getint('min_nb_pixels'))
                 & (df_pred[conf.csv['prediction_status']] != 'NODATA')
                 & (df_pred[conf.csv['prediction_status']] != 'DOUBT'),
                [conf.csv['prediction_cons_column'], conf.csv['prediction_status']]] = 'NOT_ENOUGH_PIXELS'

    df_pred.loc[df_pred[conf.csv['class_column']] == 'UNKNOWN', [conf.csv['prediction_status']]] = 'UNKNOWN'
    df_pred.loc[df_pred[conf.csv['class_column']].str.startswith('IGNORE_'), [conf.csv['prediction_status']]] = df_pred[conf.csv['class_column']]

    logger.info("Write final prediction data to file")
    df_pred.to_csv(output_predictions_csv, float_format='%.10f', encoding='utf-8')

def _get_top_3_prediction(df_probabilities):
    """ Returns the top 3 predictions for each parcel.

    The return value will be a dataset with the following columns:
        - global_settings.id_column: id of the parcel as in the input
        - global_settings.class_column: class of the parcel as in the input
        - global_settings.prediction_cons_column: the consolidated prediction, will be 'DOUBT'
          if the prediction had a relatively low probability.
        - global_settings.prediction_columns: prediction with the highest probability
        - pred1_prob: probability of the best prediction
        - pred2
        - pred2_prob
        - pred3
        - pred3_prob
    """

    logger.info("get_top_3_predictions: start")
    df_probabilities_tmp = df_probabilities.copy()
    for column in df_probabilities_tmp.columns:
        if column in conf.csv['dedicated_data_columns']:
            df_probabilities_tmp.drop(column, axis=1, inplace=True)

    # Get the top 3 predictions for each row
    # First get the indeces of the top 3 predictions for each row
    # Remark: argsort sorts ascending, so we need to take:
    #     - "[:,": for all rows
    #     - ":-4": the last 3 elements of the values
    #     - ":-1]": and than reverse the order with a negative step
    top3_pred_classes_idx = np.argsort(df_probabilities_tmp.values, axis=1)[:, :-4:-1]
    # Convert the indeces to classes
    top3_pred_classes = np.take(df_probabilities_tmp.columns, top3_pred_classes_idx)
    # Get the values of the top 3 predictions
    top3_pred_values = np.sort(df_probabilities_tmp.values, axis=1)[:, :-4:-1]
    # Concatenate both
    top3_pred = np.concatenate([top3_pred_classes, top3_pred_values], axis=1)
    # Concatenate the ids, the classes and the top3 predictions
    id_class_top3 = np.concatenate([df_probabilities[[conf.csv['id_column'], conf.csv['class_column']]].values, top3_pred]
                                   , axis=1)

    # Convert to dataframe, combine with input data and write to file
    df_top3 = pd.DataFrame(id_class_top3,
                           columns=[conf.csv['id_column'], conf.csv['class_column'],
                                    conf.csv['prediction_column'], 'pred2', 'pred3',
                                    'pred1_prob', 'pred2_prob', 'pred3_prob'])

    return df_top3

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
