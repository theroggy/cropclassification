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

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
import global_settings as gs

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

# TODO: report on alfa and beta error using test data
# TODO: report on alfa and beta error using ground truth data
# TODO: improve reporting to devide between eligible versus ineligible classes?
# TODO: report based on area instead of number parcel?

def write_OA_per_pixcount(parcel_predictions_csv: str
                         ,parcel_pixcount_csv: str
                         ,output_report_txt: str
                         ,force: bool = False):

    # If force == False Check and the output file exists already, stop.
    if force == False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    logger.info('Start write_OA_per_pixcount')
    df_predict = pd.read_csv(parcel_predictions_csv, low_memory = False)
    df_pixcount = pd.read_csv(parcel_pixcount_csv, low_memory = False)

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
                     ,output_report_txt: str
                     ,parcel_classes_test_csv: str = None
                     ,parcel_ground_truth_csv: str = None
                     ,force: bool = None):
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
    """
    # If force == False Check and the output file exists already, stop.
    if force == False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    logger.info('Start write_full_report')

    logger.info('Read csv with probabilities')
    df_predict = pd.read_csv(parcel_predictions_csv, low_memory = False)
    df_predict.set_index(gs.id_column, inplace=True)

    classes = sorted(np.unique(np.append(df_predict[gs.class_column].unique(), df_predict[gs.prediction_column].unique())))
    logger.debug('Input shape: (index: {}, columns: {}), Unique classes found: {}'.format(df_predict.shape[0], df_predict.shape[1], classes))

    # Build and write report...
    with open(output_report_txt, 'w') as outputfile:
        # Output general accuracy
        overall_accuracy = 100.0*sklearn.metrics.accuracy_score(df_predict[gs.class_column], df_predict[gs.prediction_column], normalize=True, sample_weight=None)
        message = f"General OA: {overall_accuracy} of the items with a prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")

        # Output best-case accuracy
        # The 'UNKNOWN' and 'IGNORE_' classes aren't meant for training... so remove them!
        logger.info("For best-case accuracy, remove the 'UNKNOWN' class and the classes that start with 'IGNORE_'")
        df_predict_accuracy_best_case = df_predict[df_predict[gs.class_column] != 'UNKNOWN']
        df_predict_accuracy_best_case = df_predict_accuracy_best_case[~df_predict_accuracy_best_case[gs.class_column].str.startswith('IGNORE_', na=True)]

        overall_accuracy = 100.0*sklearn.metrics.accuracy_score(df_predict_accuracy_best_case[gs.class_column], df_predict_accuracy_best_case[gs.prediction_column], normalize=True, sample_weight=None)
        message = f"Best case OA: {overall_accuracy} of the items with a prediction that are not in classes UNKNOWN nor IGNORE_*"
        logger.info(message)
        outputfile.write(f"{message}\n")

        if parcel_classes_test_csv is not None:
            logger.info('Read csv with all test parcel (with their classes)')
            df_parcel_all_test = pd.read_csv(parcel_classes_test_csv, low_memory = False)
            df_parcel_all_test.set_index(gs.id_column, inplace=True)

            # Output info about parcel with no prediction
            # Get all test percels that are in the full list, but don't have a prediction
            df_parcel_all_test_no_predict = df_parcel_all_test[~df_parcel_all_test.index.isin(df_predict.index)]

            nb_parcel_test_no_predict = len(df_parcel_all_test_no_predict)
            message = f"Number of parcels that don't have a prediction: {nb_parcel_test_no_predict}"
            logger.info(message)
            outputfile.write(f"{message}\n")
            # If there are parcel that don't have a prediction, write the number of items per class to reporting file
            if nb_parcel_test_no_predict > 0:
                count_per_class = df_parcel_all_test_no_predict.groupby(gs.class_column, as_index = False).size()
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    outputfile.write(f"Number of parcels the don't have prediction per classname:\n{count_per_class}\n")

            if parcel_ground_truth_csv is not None:
                # Output alfa error
                # TODO: to implement correctly ;-)...
                logger.info('Read csv with ground truth (with their classes)')
                df_parcel_ground_truth = pd.read_csv(parcel_ground_truth_csv, low_memory = False)
                df_parcel_ground_truth.set_index(gs.id_column, inplace=True)
                logger.info('Read csv with ground truth ready, shape: {df_parcel_ground_truth.shape}')

                # Only join the columns of df_classification_data that aren't yet in df_train
                cols_to_join = df_predict.columns.difference(df_parcel_all_test.columns)
                df_all_with_pred = df_parcel_all_test.join(df_predict[cols_to_join], how = 'left')
                df_all_with_pred[gs.prediction_column].fillna('NODATA', inplace=True)
                df_all_with_pred[gs.prediction_cons_column].fillna('NODATA', inplace=True)

                df = df_all_with_pred.join(df_parcel_ground_truth, how = 'inner', rsuffix = '_GT')
                logger.debug('Number of ground truth parcels that also have a prediction: {len(df)}')

                # Add the alfa error
                def get_error(row, prediction_column_to_use):
                    # For some reason the row['pred2_prob'] is sometimes seen as string, and so 2* gives a repetition of the string value instead
                    # of a mathematic multiplication... so cast to float!

                    if row[gs.class_column] == row[gs.class_column + '_GT']:
                        # Farmer did a correct application
                        if (row[gs.class_column + '_GT'] == row[prediction_column_to_use]):
                            # Prediction was the same as the ground truth, so correct!
                            return 'OK_EVERYTHING_CORRECT'
                        elif (row[gs.class_column] == 'UNKNOWN'
                                      or row[gs.class_column].startswith('IGNORE_')):
                            # Input classname was special
                            return f"OK_BENEFIT_OF_DOUBT_CLASSNAME={row[gs.class_column]}"
                        elif (row[prediction_column_to_use] in ['DOUBT', 'NODATA']):
                            # Prediction resulted in doubt or there was no/not enough data
                            return f"OK_BENEFIT_OF_DOUBT_PRED={row[prediction_column_to_use]}"
                        else:
                            # Prediction was wrong, and opposes the farmer!
                            return 'ERROR_ALFA'
                    else:
                        # Farmer did an incorrect application
                        if (row[gs.class_column + '_GT'] == row[prediction_column_to_use]):
                            # Prediction was the same as the ground truth, so correct!
                            return 'OK_FARMER_WRONG_PREDICTION_CORRECT'
                        elif (row[gs.class_column] == row[prediction_column_to_use]):
                            # Prediction was wrong, but same as the farmer!
                            return 'ERROR_BETA_FARMER_WRONG_PREDICTION_DIDNT_OPPOSE'
                        elif (row[gs.class_column] == 'UNKNOWN'
                                      or row[gs.class_column].startswith('IGNORE_')):
                            # Input classname was special
                            return f"ERROR_BETA_BENEFIT_OF_DOUBT_CLASSNAME={row[gs.class_column]}"
                        elif (row[prediction_column_to_use] in ['DOUBT', 'NODATA']):
                            # Prediction resulted in doubt or there was no/not enough data
                            return f"OK_BENEFIT_OF_DOUBT_PRED={row[prediction_column_to_use]}"
                        else:
                            # Prediction was wrong, but same as the farmer!
                            return 'OK_FARMER_WRONG_PREDICTION_DIFFERENT'

                values = df.apply(get_error, args = [gs.prediction_column], axis=1)
                prediction_quality_column_name = f"prediction_quality_{gs.prediction_column}"
                df.insert(loc = 2, column = prediction_quality_column_name, value = values)

                prediction_quality_cons_column_name = f"prediction_quality_{gs.prediction_cons_column}"
                values = df.apply(get_error, args = [gs.prediction_cons_column], axis=1)
                df.insert(loc = 2, column = prediction_quality_cons_column_name, value = values)
                df.to_csv(output_report_txt + "groundtruth_pred_quality_details.csv")

                # First write the result for the standard predictions
                count_per_class = df.groupby(prediction_quality_column_name, as_index = False).size().to_frame('count')
                values = 100*count_per_class['count']/count_per_class['count'].sum()
                count_per_class.insert(loc = 1, column = 'pct', value = values)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    message = f"Prediction quality overview, for {len(df)} predicted cases in ground truth:\n{count_per_class}"
                    logger.info(message)
                    outputfile.write(f"{message}\n")

                # Now write the result for the consolidated predictions
                count_per_class = df.groupby(prediction_quality_cons_column_name, as_index = False).size().to_frame('count')
                values = 100*count_per_class['count']/count_per_class['count'].sum()
                count_per_class.insert(loc = 1, column = 'pct', value = values)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    message = f"Prediction quality cons overview, for {len(df)} predicted cases in ground truth:\n{count_per_class}"
                    logger.info(message)
                    outputfile.write(f"{message}\n")

        # Write the recall, F1 score,... per class
        message = sklearn.metrics.classification_report(df_predict[gs.class_column], df_predict[gs.prediction_column], labels = classes)
        outputfile.write(message)

        # Write extended confusion matrix and make sure all rows and collumns are written to the output file...
        # Calculate confusion matrix on the predictions
        np_confmatrix = sklearn.metrics.confusion_matrix(df_predict[gs.class_column], df_predict[gs.prediction_column], labels = classes)
        df_confmatrix = pd.DataFrame(np_confmatrix, classes, classes)

        df_confmatrix_ext = df_confmatrix.copy()

        # Add some more columns to the confusion matrix
        # Insert column with the total nb of parcel for each class that were the input values (=sum of the row of each class)
        values = df_confmatrix_ext[df_confmatrix_ext.columns].sum(axis=1)
        df_confmatrix_ext.insert(loc = 0, column = 'nb_input', value = values)
        # Insert column with the total nb of parcel for each class that were predicted to have this value (=sum of the column of each class)
        values = df_confmatrix_ext[df_confmatrix_ext.columns].sum(axis=0)
        df_confmatrix_ext.insert(loc = 1, column = 'nb_predicted', value = values)
        # Insert column with the total nb of correctly predicted classes
        df_confmatrix_ext.insert(loc = 2, column = 'nb_predicted_correct', value = 0)
        for column in df_confmatrix_ext.columns:
            if column not in ['nb_input', 'nb_predicted', 'nb_predicted_correct']:
                df_confmatrix_ext['nb_predicted_correct'][column] = df_confmatrix_ext[column][column]
        # Insert column with the total nb of parcel for each class that were predicted to have this value (=sum of the column of each class)
        values = df_confmatrix_ext['nb_predicted'] - df_confmatrix_ext['nb_predicted_correct']
        df_confmatrix_ext.insert(loc = 3, column = 'nb_predicted_wrong', value = values)
        # Insert columns with percentages
        values = df_confmatrix_ext['nb_predicted_correct']*100/df_confmatrix_ext['nb_input']
        df_confmatrix_ext.insert(loc = 4, column = 'pct_predicted_correct', value = values)
        values = (df_confmatrix_ext['nb_predicted'] - df_confmatrix_ext['nb_predicted_correct'])*100/df_confmatrix_ext['nb_input']
        df_confmatrix_ext.insert(loc = 5, column = 'pct_predicted_wrong', value = values)

        outputfile.write("Confusion matrix of the predictions: Rows: true/input classes, columns: predicted classes\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            outputfile.write(str(df_confmatrix_ext))

# If the script is run directly...
if __name__ == "__main__":

    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')

    '''
    local_base_dir = 'X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-05-04_Monitoring_Classificatie' # Local base dir
    local_class_dir = os.path.join(local_base_dir, 'class_maincrops_mon')                       # Specific dir for this classification
    local_class_dir = os.path.join(local_class_dir, '2018-07-20_Run3_newsampling_S2')
    base_filename = 'BEVL2017_weekly_bufm5'

    class_predictions_filepath = os.path.join(local_class_dir, "{base_fn}_predict.csv".format(base_fn=base_filename))

    print(predictions_csv = class_predictions_filepath
         ,class_columnname = 'classname'
         ,prediction_columnname = 'pred1')
    '''