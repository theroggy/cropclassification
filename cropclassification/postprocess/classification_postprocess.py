# -*- coding: utf-8 -*-
"""
Module with postprocessing functions on classification results.
"""

import logging

import numpy as np
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

def calc_top3_and_consolidation(input_parcel_filepath: str,
                                input_parcel_probabilities_filepath: str,
                                output_predictions_filepath: str):

    # Read input files
    df_input_parcel = pdh.read_file(input_parcel_filepath)
    df_proba = pdh.read_file(input_parcel_probabilities_filepath)

    # Calculate the top 3 predictions
    df_top3 = _get_top_3_prediction(df_proba)

    # Add the consolidated prediction
    # TODO: try to rewrite using native pandas commands to improve performance
    def calculate_consolidated_prediction(row):
        # For some reason the row['pred2_prob'] is sometimes seen as string, and so 2* gives a
        # repetition of the string value instead of a mathematic multiplication... so cast to float!

        # float(row['pred1_prob']) > 30
        if ((float(row['pred1_prob']) >= 2.0 * float(row['pred2_prob']))):
            return row[conf.csv['prediction_column']]
        else:
            return 'DOUBT'

    values = df_top3.apply(calculate_consolidated_prediction, axis=1)
    df_top3.insert(loc=2, column=conf.csv['prediction_withdoubt_column'], value=values)

    # Make sure all input parcels are in the output. If there was no prediction, it means that there
    # was no data available for a classification, so set prediction to NODATA
    df_top3.set_index(conf.csv['id_column'], inplace=True)
    df_input_parcel.set_index(conf.csv['id_column'], inplace=True)

    # Add a column with the prediction status... and all parcels in df_top3 got a prediction
    df_top3[conf.csv['prediction_status']] = 'OK'
    df_top3.loc[(df_top3[conf.csv['prediction_withdoubt_column']] == 'DOUBT'),
                conf.csv['prediction_status']] = 'DOUBT'

    cols_to_join = df_top3.columns.difference(df_input_parcel.columns)
    df_pred = df_input_parcel.join(df_top3[cols_to_join], how='left')
    df_pred[conf.csv['prediction_column']].fillna('NODATA', inplace=True)
    df_pred[conf.csv['prediction_withdoubt_column']].fillna('NODATA', inplace=True)
    df_pred[conf.csv['prediction_status']].fillna('NODATA', inplace=True)

    logger.info(f"Columns of df_pred: {df_pred.columns}")

    # Now calculate the full consolidated prediction: 
    #    * Can be doubt if probability too low
    #    * Parcels with too few pixels don't have a good accuracy and give many alfa errors...
    df_pred[conf.csv['prediction_cons_column']] = df_pred[conf.csv['prediction_withdoubt_column']]
    df_pred.loc[(df_pred[conf.csv['pixcount_s1s2_column']] <= conf.marker.getint('min_nb_pixels'))
                 & (df_pred[conf.csv['prediction_status']] != 'NODATA')
                 & (df_pred[conf.csv['prediction_status']] != 'DOUBT'),
                [conf.csv['prediction_cons_column'], conf.csv['prediction_status']]] = 'NOT_ENOUGH_PIXELS'

    # Set the prediction status for classes that should be ignored
    df_pred.loc[df_pred[conf.csv['class_column']].isin(conf.marker.getlist('classes_to_ignore_for_train')), 
                [conf.csv['prediction_status']]] = 'UNKNOWN'
    df_pred.loc[df_pred[conf.csv['class_column']].isin(conf.marker.getlist('classes_to_ignore')), 
                [conf.csv['prediction_status']]] = df_pred[conf.csv['class_column']]

    logger.info("Write final prediction data to file")
    pdh.to_file(df_pred, output_predictions_filepath)

def _get_top_3_prediction(df_probabilities):
    """ Returns the top 3 predictions for each parcel.

    The return value will be a dataset with the following columns:
        - id_column: id of the parcel as in the input
        - class_column: class of the parcel as in the input
        - prediction_cons_column: the consolidated prediction, will be 'DOUBT'
          if the prediction had a relatively low probability.
        - prediction_columns: prediction with the highest probability
        - pred1_prob: probability of the best prediction
        - pred2
        - pred2_prob
        - pred3
        - pred3_prob
    """

    logger.info("get_top_3_predictions: start")
    df_probabilities_tmp = df_probabilities.copy()
    for column in df_probabilities_tmp.columns:
        if column in conf.csv.getlist('dedicated_data_columns'):
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

    # Convert to dataframe and return
    df_top3 = pd.DataFrame(id_class_top3,
                           columns=[conf.csv['id_column'], conf.csv['class_column'],
                                    conf.csv['prediction_column'], 'pred2', 'pred3',
                                    'pred1_prob', 'pred2_prob', 'pred3_prob'])

    return df_top3