# -*- coding: utf-8 -*-
"""
Module with some helper functions to report on the classification results.

@author: Pieter Roggemans
"""

import logging
import os
import pandas as pd
import numpy as np
import sklearn
import global_settings as gs

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
PRED_QUALITY_COLUMN = f"prediction_quality_{gs.prediction_column}"
PRED_QUALITY_CONS_COLUMN = f"prediction_quality_{gs.prediction_cons_column}"

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

# TODO: improve reporting to devide between eligible versus ineligible classes?
# TODO: report based on area instead of number parcel
#     -> seems like being a bit "detached from reality", as for RFV the most important parameter is the number of parcels
# TODO: UNKNOWN class, and definitely IGNORE_ classes now also contain classes that are difficult to distinguish to "ignore" them, to be able to compare them
#       to ground truth they should somehow keep their original class (as well?)...
def write_OA_per_pixcount(parcel_predictions_csv: str
                          , parcel_pixcount_csv: str
                          , output_report_txt: str
                          , force: bool = False):
    """ Write a report of the overall accuracy that parcels per pixcount get. """
    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    logger.info('Start write_OA_per_pixcount')
    df_predict = pd.read_csv(parcel_predictions_csv, low_memory=False)
    df_pixcount = pd.read_csv(parcel_pixcount_csv, low_memory=False)

    # Set the index to the id_columnname, and join the data to the result...
    df_predict.set_index(gs.id_column, inplace=True)
    df_pixcount.set_index(gs.id_column, inplace=True)
    df_result = df_predict.join(df_pixcount, how='inner')
    nb_predictions_total = len(df_result.index)

    # Write output...
    with open(output_report_txt, 'w') as outputfile:
        for i in range(40):

            df_result_cur_pixcount = df_result[df_result[gs.pixcount_s1s2_column] == i]
            nb_predictions_pixcount = len(df_result_cur_pixcount.index)
            if nb_predictions_pixcount == 0:
                continue

            overall_accuracy = 100.0*sklearn.metrics.accuracy_score(df_result_cur_pixcount[gs.class_column], df_result_cur_pixcount[gs.prediction_column], normalize=True, sample_weight=None)
            message = f"OA for pixcount {i:2}: {overall_accuracy:3.2f} %, with {nb_predictions_pixcount} elements ({100*(nb_predictions_pixcount/nb_predictions_total):.4f} % of {nb_predictions_total})"
            logger.info(message)
            outputfile.write(f"{message}\n")

def write_full_report(parcel_predictions_csv: str
                      , output_report_txt: str
                      , parcel_classes_to_report_on_csv: str = None
                      , parcel_ground_truth_csv: str = None
                      , parcel_pixcount_csv: str = None
                      , force: bool = None):
    """Writes a report about the accuracy of the predictions to a file.

    Args:
        parcel_predictions_csv: File name of csv file with the parcel with their predictions.
        prediction_columnname: Column name of the column that contains the predictions.
        output_report_txt: File name of txt file the report will be written to.
        parcel_classes_test_csv: The original full list of test parcel, including parcel that
            couldn't get a prediction due to eg. missing classification data. If None, the
            part of the report that is based on this data is skipped.
        parcel_ground_truth_csv: List of parcels with ground truth to calculate eg. alfa and
            beta errors. If None, the part of the report that is based on this data is skipped

    TODO: refactor function to split logic more...
    """
    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    logger.info('Start write_full_report')

    logger.info('Read csv with probabilities')
    df_predict = pd.read_csv(parcel_predictions_csv, low_memory=False)
    df_predict.set_index(gs.id_column, inplace=True)

    df_parcel_to_report = None
    if parcel_classes_to_report_on_csv is not None:
        logger.info('Read csv with all test parcel (with their classes)')
        df_parcel_to_report = pd.read_csv(parcel_classes_to_report_on_csv, low_memory=False)
        df_parcel_to_report.set_index(gs.id_column, inplace=True)

    # Build and write report...
    with open(output_report_txt, 'w') as outputfile:
        # Output general accuracies
        # Remark: the titles are 60 karakters wide...
        outputfile.write("************************************************************\n")
        outputfile.write("***************** GENERAL ACCURACIES ***********************\n")
        outputfile.write("************************************************************\n")

        # First report the general accuracies for the entire list to report on...
        if df_parcel_to_report is not None:
            # Join parcel classes_to_report_on with predictions
            df_parcel_to_report_pred = (df_parcel_to_report
                                        .join(df_predict[[gs.prediction_column
                                                         , gs.prediction_cons_column]], how='left'))
            df_parcel_to_report_pred[gs.prediction_column].fillna('NODATA', inplace=True)
            df_parcel_to_report_pred[gs.prediction_cons_column].fillna('NODATA', inplace=True)
            oa = sklearn.metrics.accuracy_score(df_parcel_to_report_pred[gs.class_column]
                                                , df_parcel_to_report_pred[gs.prediction_column]
                                                , normalize=True
                                                , sample_weight=None) * 100
            message = f"OA: {oa:.2f} for all parcels to report on, for standard prediction"
            logger.info(message)
            outputfile.write(f"{message}\n")

            oa = sklearn.metrics.accuracy_score(df_parcel_to_report_pred[gs.class_column]
                                                , df_parcel_to_report_pred[gs.prediction_cons_column]
                                                , normalize=True
                                                , sample_weight=None) * 100
            message = f"OA: {oa:.2f} for all parcels to report on, for consolidated prediction"
            logger.info(message)
            outputfile.write(f"{message}\n")

        # Now report on parcels that actually had a prediction
        oa = sklearn.metrics.accuracy_score(df_predict[gs.class_column]
                                            , df_predict[gs.prediction_column]
                                            , normalize=True
                                            , sample_weight=None) * 100
        message = f"OA: {oa:.2f} for parcels with a prediction (= excl. NODATA parcels), for standard prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")

        oa = sklearn.metrics.accuracy_score(df_predict[gs.class_column]
                                            , df_predict[gs.prediction_cons_column]
                                            , normalize=True
                                            , sample_weight=None) * 100
        message = f"OA: {oa:.2f} for parcels with a prediction (= excl. NODATA parcels), for consolidated prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")

        # Output best-case accuracy
        # Ignore the 'UNKNOWN' and 'IGNORE_' classes...
        logger.info("For best-case accuracy, remove the 'UNKNOWN' class and the classes that start with 'IGNORE_'")
        df_predict_accuracy_best_case = df_predict[df_predict[gs.class_column] != 'UNKNOWN']
        df_predict_accuracy_best_case = \
                df_predict_accuracy_best_case[~df_predict_accuracy_best_case[gs.class_column]
                                              .str.startswith('IGNORE_', na=True)]

        oa = sklearn.metrics.accuracy_score(df_predict_accuracy_best_case[gs.class_column]
                                            , df_predict_accuracy_best_case[gs.prediction_column]
                                            , normalize=True
                                            , sample_weight=None) * 100
        message = f"OA: {oa:.2f} for the parcels with a prediction, excl. classes UNKNOWN and IGNORE_*, for standard prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")

        oa = sklearn.metrics.accuracy_score(df_predict_accuracy_best_case[gs.class_column]
                                            , df_predict_accuracy_best_case[gs.prediction_cons_column]
                                            , normalize=True
                                            , sample_weight=None) * 100
        message = f"OA: {oa:.2f} for the parcels with a prediction, excl. classes UNKNOWN and IGNORE_*, for consolidated prediction"
        logger.info(message)
        outputfile.write(f"{message}\n\n")

        if parcel_classes_to_report_on_csv is not None:
            outputfile.write("************************************************************\n")
            outputfile.write("************* PARCELS WITHOUT PREDICTION (no data?) ********\n")
            outputfile.write("************************************************************\n")

            # Output info about parcel with no prediction
            # Get all test percels that are in the full list, but don't have a prediction
            df_parcel_to_report_no_predict = \
                    df_parcel_to_report[~df_parcel_to_report.index.isin(df_predict.index)]

            nb_parcel_test_no_predict = len(df_parcel_to_report_no_predict)
            message = f"Number of parcels that don't have a prediction: {nb_parcel_test_no_predict}"
            logger.info(message)
            outputfile.write(f"{message}\n")
            # If there are parcel that don't have a prediction, write the number of items per class
            # to reporting file
            if nb_parcel_test_no_predict > 0:
                count_per_class = (df_parcel_to_report_no_predict
                                   .groupby(gs.class_column, as_index=False).size())
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    outputfile.write(f"Number of parcels the don't have prediction per classname:\n{count_per_class}\n\n")

        # Write the recall, F1 score,... per class
#        message = sklearn.metrics.classification_report(df_predict[gs.class_column]
#                                                        , df_predict[gs.prediction_column]
#                                                        , labels=classes)
#        outputfile.write(message)

        outputfile.write("************************************************************\n")
        outputfile.write("***** CONFUSION MATRICES FOR PARCELS WITH PREDICTIONS ******\n")
        outputfile.write("************************************************************\n")
        # Calculate an extended confusion matrix with the standard prediction column and write
        # it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(df_predict, gs.prediction_column)
        outputfile.write("\nExtended confusion matrix of the predictions: Rows: true/input classes, columns: predicted classes\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            outputfile.write(f"{df_confmatrix_ext}\n")

        # Calculate an extended confusion matrix with the consolidated prediction column and write
        # it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(df_predict, gs.prediction_cons_column)
        outputfile.write("\nExtended confusion matrix of the consolidated predictions: Rows: true/input classes, columns: predicted classes\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            outputfile.write(f"{df_confmatrix_ext}\n\n")

        # If a ground truth file is provided, report on the ground truth
        if parcel_ground_truth_csv is not None:
            outputfile.write("************************************************************\n")
            outputfile.write("*** REPORTING ON PREDICTION QUALITY BASED ON GROUND TRUTH **\n")
            outputfile.write("************************************************************\n")

            # First check if the mandatory parcel_classes_sample_csv is provided as well...
            if parcel_classes_to_report_on_csv is None:
                message = "If parcel_ground_truth_csv is not None, parcel_classes_to_report_on_csv should be provided as well! STOP!"
                logger.critical(message)
                raise Exception(message)

            # Read ground truth
            logger.info(f"Read csv with ground truth (with their classes): {parcel_ground_truth_csv}")
            df_parcel_gt = pd.read_csv(parcel_ground_truth_csv, low_memory=False)
            df_parcel_gt.set_index(gs.id_column, inplace=True)
            logger.info(f"Read csv with ground truth ready, shape: {df_parcel_gt.shape}")

            # Rename the classname column in ground truth
            df_parcel_gt.rename(columns={gs.class_column: gs.class_column + '_GT'}
                                , inplace=True)

            # Join with the classes_sample, as we only want to report on the ground truth that
            # is also in the sample
            if parcel_classes_to_report_on_csv is not None:
                # Only add columns that aren't in the ground truth yet...
                cols_to_join = df_parcel_to_report.columns.difference(df_parcel_gt.columns)
                df_parcel_gt = df_parcel_gt.join(df_parcel_to_report[cols_to_join], how='inner')

            # If there is a pixcount file specified, add the pixcount column as well
            if(parcel_pixcount_csv is not None
               and gs.pixcount_s1s2_column not in df_parcel_gt.columns):
                logger.info(f"Read csv with pixcount: {parcel_pixcount_csv}")
                df_pixcount = pd.read_csv(parcel_pixcount_csv, low_memory=False)
                df_pixcount.set_index(gs.id_column, inplace=True)
                df_parcel_gt = (df_parcel_gt
                                .join(df_pixcount[gs.pixcount_s1s2_column], how='left'))

            # Join the prediction data and fill up missing predictions
            cols_to_join = df_predict.columns.difference(df_parcel_gt.columns)
            df_parcel_gt = df_parcel_gt.join(df_predict[cols_to_join], how='left')
            df_parcel_gt[gs.prediction_column].fillna('NODATA', inplace=True)
            df_parcel_gt[gs.prediction_cons_column].fillna('NODATA', inplace=True)

            logger.debug('Number of ground truth parcels that also have a prediction: {len(df_parcel_gt)}')

            # Add the alfa error
            # TODO: this needs to be reviewed, maybe need to compare with original classname
            #       instead of _IGNORE, UNKNOWN,...
            def get_prediction_quality(row, prediction_column_to_use):
                """ Get a string that gives a quality indicator of the prediction. """
                # For some reason the row['pred2_prob'] is sometimes seen as string, and
                # so 2* gives a repetition of the string value instead
                # of a mathematic multiplication... so cast to float!

                if row[gs.class_column] == row[gs.class_column + '_GT']:
                    # Farmer did a correct application
                    if row[gs.class_column + '_GT'] == row[prediction_column_to_use]:
                        # Prediction was the same as the ground truth, so correct!
                        return 'OK_EVERYTHING_CORRECT'
                    elif (row[gs.class_column] == 'UNKNOWN'
                          or row[gs.class_column].startswith('IGNORE_')):
                        # Input classname was special
                        return f"OK_BENEFIT_OF_DOUBT_CLASSNAME={row[gs.class_column]}"
                    elif row[prediction_column_to_use] in ['DOUBT', 'NODATA']:
                        # Prediction resulted in doubt or there was no/not enough data
                        return f"OK_BENEFIT_OF_DOUBT_PRED={row[prediction_column_to_use]}"
                    else:
                        # Prediction was wrong, and opposes the farmer!
                        return 'ERROR_ALFA'
                else:
                    # Farmer did an incorrect application
                    if row[gs.class_column + '_GT'] == row[prediction_column_to_use]:
                        # Prediction was the same as the ground truth, so correct!
                        return 'OK_FARMER_WRONG_PREDICTION_CORRECT'
                    elif row[gs.class_column] == row[prediction_column_to_use]:
                        # Prediction was wrong, but same as the farmer!
                        return 'ERROR_BETA_FARMER_WRONG_PREDICTION_DIDNT_OPPOSE'
                    elif (row[gs.class_column] == 'UNKNOWN'
                          or row[gs.class_column].startswith('IGNORE_')):
                        # Input classname was special
                        return f"ERROR_BETA_FARMER_WRONG_CLASSNAME={row[gs.class_column]}"
                    elif row[prediction_column_to_use] in ['DOUBT', 'NODATA']:
                        # Prediction resulted in doubt or there was no/not enough data
                        return f"OK_BENEFIT_OF_DOUBT_PRED={row[prediction_column_to_use]}"
                    else:
                        # Prediction was wrong, but same as the farmer!
                        return 'OK_FARMER_WRONG_PREDICTION_DIFFERENT'

            values = df_parcel_gt.apply(get_prediction_quality, args=[gs.prediction_column], axis=1)
            df_parcel_gt.insert(loc=2, column=PRED_QUALITY_COLUMN, value=values)
            values = df_parcel_gt.apply(get_prediction_quality, args=[gs.prediction_cons_column], axis=1)
            df_parcel_gt.insert(loc=2, column=PRED_QUALITY_CONS_COLUMN, value=values)
            df_parcel_gt.to_csv(output_report_txt + "groundtruth_pred_quality_details.csv")

            # First write the result for the standard predictions
            count_per_class = (df_parcel_gt.groupby(PRED_QUALITY_COLUMN, as_index=False)
                               .size().to_frame('count'))
            values = 100*count_per_class['count']/count_per_class['count'].sum()
            count_per_class.insert(loc=1, column='pct', value=values)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                message = f"Prediction quality overview, for {len(df_parcel_gt)} predicted cases in ground truth:\n{count_per_class}"
                logger.info(message)
                outputfile.write(f"{message}\n")

            # Now write the result for the consolidated predictions
            count_per_class = (df_parcel_gt.groupby(PRED_QUALITY_CONS_COLUMN, as_index=False)
                               .size().to_frame('count'))
            values = 100*count_per_class['count']/count_per_class['count'].sum()
            count_per_class.insert(loc=1, column='pct', value=values)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                message = f"Prediction quality cons overview, for {len(df_parcel_gt)} predicted cases in ground truth:\n{count_per_class}"
                logger.info(message)
                outputfile.write(f"{message}\n")

            # If the pixcount is available, write the number of ALFA errors per pixcount
            if gs.pixcount_s1s2_column in df_parcel_gt.columns:
                # Get data, drop empty lines and write
                df_per_pixcount = _get_alfa_errors_per_pixcount(df_parcel_gt)
                df_per_pixcount.dropna(inplace=True)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
                    message = f"Number of ERROR_ALFA parcels for the consolidated prediction per pixcount for the ground truth parcels:\n{df_per_pixcount}"
                    logger.info(message)
                    outputfile.write(f"{message}\n")

def _get_confusion_matrix_ext(df_predict, prediction_column_to_use: str):
    """ Returns a dataset with an extended confusion matrix. """

    classes = sorted(np.unique(np.append(df_predict[prediction_column_to_use].unique()
                                         , df_predict[gs.class_column].unique())))
    logger.debug(f"Input shape: {df_predict.shape}, Unique classes found: {classes}")

    # Calculate standard confusion matrix
    np_confmatrix = sklearn.metrics.confusion_matrix(df_predict[gs.class_column]
                                                     , df_predict[prediction_column_to_use]
                                                     , labels=classes)
    df_confmatrix_ext = pd.DataFrame(np_confmatrix, classes, classes)

    # Add some more columns to the confusion matrix
    # Insert column with the total nb of parcel for each class that were the input values
    # (=sum of the row of each class)
    values = df_confmatrix_ext[df_confmatrix_ext.columns].sum(axis=1)
    df_confmatrix_ext.insert(loc=0, column='nb_input', value=values)
    # Insert column with the total nb of parcel for each class that were predicted to have
    # this value (=sum of the column of each class)
    values = df_confmatrix_ext[df_confmatrix_ext.columns].sum(axis=0)
    df_confmatrix_ext.insert(loc=1, column='nb_predicted', value=values)
    # Insert column with the total nb of correctly predicted classes
    df_confmatrix_ext.insert(loc=2, column='nb_predicted_correct', value=0)
    for column in df_confmatrix_ext.columns:
        if column not in ['nb_input', 'nb_predicted', 'nb_predicted_correct']:
            df_confmatrix_ext['nb_predicted_correct'][column] = df_confmatrix_ext[column][column]
    # Insert column with the total nb of parcel for each class that were predicted to have
    # this value (=sum of the column of each class)
    values = df_confmatrix_ext['nb_predicted'] - df_confmatrix_ext['nb_predicted_correct']
    df_confmatrix_ext.insert(loc=3, column='nb_predicted_wrong', value=values)
    # Insert columns with percentages
    values = df_confmatrix_ext['nb_predicted_correct']*100/df_confmatrix_ext['nb_input']
    df_confmatrix_ext.insert(loc=4, column='pct_predicted_correct', value=values)
    values = ((df_confmatrix_ext['nb_predicted'] - df_confmatrix_ext['nb_predicted_correct'])
              * 100 / df_confmatrix_ext['nb_input'])
    df_confmatrix_ext.insert(loc=5, column='pct_predicted_wrong', value=values)

    return df_confmatrix_ext

def _get_alfa_errors_per_pixcount(df_predquality_pixcount):
    """ Returns a dataset with detailed information about the number of alfa errors per pixcount """

    # Calculate the number of parcels per pixcount, the cumulative sum + the pct of all
    df_count_per_pixcount = (df_predquality_pixcount.groupby(gs.pixcount_s1s2_column, as_index=False)
                             .size().to_frame('count_all'))
    values = df_count_per_pixcount['count_all'].cumsum(axis=0)
    df_count_per_pixcount.insert(loc=len(df_count_per_pixcount.columns)
                                 , column='count_all_cumulative'
                                 , value=values)
    values = (100 * df_count_per_pixcount['count_all_cumulative']
              / df_count_per_pixcount['count_all'].sum())
    df_count_per_pixcount.insert(loc=len(df_count_per_pixcount.columns)
                                 , column='pct_all_cumulative'
                                 , value=values)

    # Now calculate the number of alfa errors per pixcount
    df_alfa_error = df_predquality_pixcount[df_predquality_pixcount[PRED_QUALITY_CONS_COLUMN] == 'ERROR_ALFA']
    df_alfa_per_pixcount = (df_alfa_error.groupby(gs.pixcount_s1s2_column
                                                  , as_index=False)
                            .size().to_frame('count_error_alfa'))

    # Join them together, and calculate the alfa error percentages
    df_alfa_per_pixcount = df_count_per_pixcount.join(df_alfa_per_pixcount
                                                      , how='left')
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns)
                                , column='count_error_alfa_cumulative'
                                , value=df_alfa_per_pixcount['count_error_alfa'].cumsum(axis=0))
    values = 100 * df_alfa_per_pixcount['count_error_alfa'] / df_alfa_per_pixcount['count_all']
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns)
                                , column='pct_error_alfa_of_all', value=values)
    values = (100 * df_alfa_per_pixcount['count_error_alfa_cumulative']
              / df_alfa_per_pixcount['count_error_alfa'].sum())
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns)
                                , column='pct_error_alfa_of_alfa_cumulative', value=values)
    values = (100 * df_alfa_per_pixcount['count_error_alfa_cumulative']
              / df_alfa_per_pixcount['count_all'].sum())
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns)
                                , column='pct_error_alfa_of_all_cumulative', value=values)

    return df_alfa_per_pixcount

# If the script is run directly...
if __name__ == "__main__":

    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
