# -*- coding: utf-8 -*-
"""
Module with postprocessing functions on classification results.
"""

import logging
import os

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
                                output_predictions_filepath: str,
                                force: bool = False):

    if(force is False
       and os.path.exists(output_predictions_filepath)):
        logger.warning(f"calc_top3_and_consolidation: output file exist and force is False, so stop: {output_predictions_filepath}")
        return

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
            return row[conf.columns['prediction']]
        else:
            return 'DOUBT'

    values = df_top3.apply(calculate_consolidated_prediction, axis=1)
    df_top3.insert(loc=2, column=conf.columns['prediction_withdoubt'], value=values)

    # Make sure all input parcels are in the output. If there was no prediction, it means that there
    # was no data available for a classification, so set prediction to NODATA
    df_top3.set_index(conf.columns['id'], inplace=True)
    if df_input_parcel.index.name != conf.columns['id']:
        df_input_parcel.set_index(conf.columns['id'], inplace=True)

    # Add a column with the prediction status... and all parcels in df_top3 got a prediction
    df_top3[conf.columns['prediction_status']] = 'OK'
    df_top3.loc[(df_top3[conf.columns['prediction_withdoubt']] == 'DOUBT'),
                conf.columns['prediction_status']] = 'DOUBT'

    cols_to_join = df_top3.columns.difference(df_input_parcel.columns)
    df_pred = df_input_parcel.join(df_top3[cols_to_join], how='left')
    df_pred[conf.columns['prediction']].fillna('NODATA', inplace=True)
    df_pred[conf.columns['prediction_withdoubt']].fillna('NODATA', inplace=True)
    df_pred[conf.columns['prediction_status']].fillna('NODATA', inplace=True)

    logger.info(f"Columns of df_pred: {df_pred.columns}")

    # Now calculate the full consolidated prediction: 
    #    * Can be doubt if probability too low
    #    * Parcels with too few pixels don't have a good accuracy and give many alfa errors...
    df_pred[conf.columns['prediction_cons']] = df_pred[conf.columns['prediction_withdoubt']]
    df_pred.loc[(df_pred[conf.columns['pixcount_s1s2']] <= conf.marker.getint('min_nb_pixels'))
                 & (df_pred[conf.columns['prediction_status']] != 'NODATA')
                 & (df_pred[conf.columns['prediction_status']] != 'DOUBT'),
                [conf.columns['prediction_cons'], conf.columns['prediction_status']]] = 'NOT_ENOUGH_PIXELS'

    # Set the prediction status for classes that should be ignored
    df_pred.loc[df_pred[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore_for_train')), 
                [conf.columns['prediction_status']]] = 'UNKNOWN'
    df_pred.loc[df_pred[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore')), 
                [conf.columns['prediction_status']]] = df_pred[conf.columns['class']]

    # Calculate consequences for the predictions
    logger.info("Calculate the consequences for the predictions")

    def add_prediction_conclusion(in_df,
                                  new_columnname, 
                                  prediction_column_to_use,
                                  detailed: bool):
        """
        Calculate the "conclusions" for the predictions 

        REMARK: calculating it like this, using native pandas operations, is 300 times faster than
                using DataFrame.apply() with a function!!!
        """
        # Add the new column with a fixed value first 
        in_df[new_columnname] = 'UNDEFINED'

        # Get a list of the classes to ignore
        all_classes_to_ignore = conf.marker.getlist('classes_to_ignore_for_train') + conf.marker.getlist('classes_to_ignore')

        # Some conclusions are different is detailed info is asked...
        if detailed == True:
            # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
            in_df.loc[in_df[conf.columns['class']].isin(all_classes_to_ignore),
                      new_columnname] = 'IGNORE:INPUTCLASSNAME=' + in_df[conf.columns['class']].map(str)
            # If conclusion still UNDEFINED, check if doubt 
            in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(['DOUBT', 'NODATA', 'NOT_ENOUGH_PIXELS'])),
                    new_columnname] = 'DOUBT:REASON=' + in_df[prediction_column_to_use].map(str)
        else:
            # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
            in_df.loc[in_df[conf.columns['class']].isin(all_classes_to_ignore),
                      new_columnname] = 'IGNORE'
            # If conclusion still UNDEFINED, check if doubt 
            in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(['DOUBT', 'NODATA', 'NOT_ENOUGH_PIXELS'])),
                    new_columnname] = 'DOUBT'

        # If conclusion still UNDEFINED, check if prediction equals the input class 
        in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                & (in_df[conf.columns['class']] == in_df[prediction_column_to_use]),
                new_columnname] = 'OK:PREDICTION=INPUT_CLASS'
        # If conclusion still UNDEFINED, prediction is different from input 
        in_df.loc[in_df[new_columnname] == 'UNDEFINED',
                new_columnname] = 'NOK:PREDICTION<>INPUT_CLASS'

    # Calculate detailed conclusions
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_detail'],
                              prediction_column_to_use=conf.columns['prediction'],
                              detailed=True)
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_detail_withdoubt'],
                              prediction_column_to_use=conf.columns['prediction_withdoubt'],
                              detailed=True)
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_detail_cons'],
                              prediction_column_to_use=conf.columns['prediction_cons'],
                              detailed=True)

    # Calculate general conclusions for cons as well
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_cons'],
                              prediction_column_to_use=conf.columns['prediction_cons'],
                              detailed=False)

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
        if column in conf.preprocess.getlist('dedicated_data_columns'):
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
    id_class_top3 = np.concatenate([df_probabilities[[conf.columns['id'], conf.columns['class']]].values, top3_pred]
                                   , axis=1)

    # Convert to dataframe and return
    df_top3 = pd.DataFrame(id_class_top3,
                           columns=[conf.columns['id'], conf.columns['class'],
                                    conf.columns['prediction'], 'pred2', 'pred3',
                                    'pred1_prob', 'pred2_prob', 'pred3_prob'])

    return df_top3