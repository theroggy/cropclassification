# -*- coding: utf-8 -*-
"""
Module with some helper functions to report on the classification results.

@author: Pieter Roggemans
"""

import logging
import os
import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
import global_settings as gs
from string import Template

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
# TODO?: report based on area instead of number parcel
#     -> seems like being a bit "detached from reality", as for RFV the most important parameter is the number of parcels

def write_full_report(parcel_predictions_csv: str,
                      output_report_txt: str,
                      parcel_ground_truth_csv: str = None,
                      force: bool = None):
    """Writes a report about the accuracy of the predictions to a file.

    Args:
        parcel_predictions_csv: File name of csv file with the parcel with their predictions.
        prediction_columnname: Column name of the column that contains the predictions.
        output_report_txt: File name of txt file the report will be written to.
        parcel_ground_truth_csv: List of parcels with ground truth to calculate eg. alfa and
            beta errors. If None, the part of the report that is based on this data is skipped

    TODO: refactor function to split logic more...
    """
    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    logger.info("Start write_full_report")

    logger.info(f"Read csv with predictions: {parcel_predictions_csv}")
    df_predict = pd.read_csv(parcel_predictions_csv, low_memory=False)
    df_predict.set_index(gs.id_column, inplace=True)

    # alle waarde moeten gekend zijn voor de template engine, dus defaulten op niets
    empty_string = "''";
    html_data = {
       'GENERAL_ACCURACIES_TABLE': empty_string,
       'GENERAL_ACCURACIES_TEXT': empty_string,
       'GENERAL_ACCURACIES_DATA': empty_string,
       'CONFUSION_MATRICES_TABLE': empty_string,
       'CONFUSION_MATRICES_DATA': empty_string,
       'CONFUSION_MATRICES_CONSOLIDATED_TABLE': empty_string,
       'CONFUSION_MATRICES_CONSOLIDATED_DATA': empty_string,
       'PREDICTION_QUALITY_OVERVIEW_TEXT': empty_string,
       'PREDICTION_QUALITY_OVERVIEW_TABLE': empty_string,
       'PREDICTION_QUALITY_CONS_OVERVIEW_TEXT': empty_string,
       'PREDICTION_QUALITY_CONS_OVERVIEW_TABLE': empty_string,
       'PREDICTION_QUALITY_ALPHA_ERROR_TEXT': empty_string,
       'PREDICTION_QUALITY_ALPHA_ERROR_TABLE': empty_string
    }
    
    # Build and write report...
    with open(output_report_txt, 'w') as outputfile:
        # Remark: the titles are 60 karakters wide...
        # Output general classification status
        outputfile.write("************************************************************\n")
        outputfile.write("***************** GENERAL ACCURACIES ***********************\n")
        outputfile.write("************************************************************\n")
        count_per_pred_status = (df_predict
                                 .groupby(gs.prediction_status, as_index=False)
                                 .size().to_frame('count'))
        values = 100 * count_per_pred_status['count'] / count_per_pred_status['count'].sum()
        count_per_pred_status.insert(loc=1, column='pct', value=values)

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            outputfile.write(f"Number of parcels per prediction status:\n{count_per_pred_status}\n\n")       
            html_data['GENERAL_ACCURACIES_TABLE'] = count_per_pred_status.to_html()
            html_data['GENERAL_ACCURACIES_DATA'] = count_per_pred_status.to_dict();
            
        # Output general accuracies
        outputfile.write("************************************************************\n")
        outputfile.write("***************** GENERAL ACCURACIES ***********************\n")
        outputfile.write("************************************************************\n")

        # First report the general accuracies for the entire list to report on...
        oa = skmetrics.accuracy_score(df_predict[gs.class_column],
                                            df_predict[gs.prediction_column],
                                            normalize=True,
                                            sample_weight=None) * 100
        message = f"OA: {oa:.2f} for all parcels to report on, for standard prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")        
        html_data['GENERAL_ACCURACIES_TEXT'] = f"<li>{message}</li>\n";

        oa = skmetrics.accuracy_score(df_predict[gs.class_column],
                                            df_predict[gs.prediction_cons_column],
                                            normalize=True,
                                            sample_weight=None) * 100
        message = f"OA: {oa:.2f} for all parcels to report on, for consolidated prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")        
        html_data['GENERAL_ACCURACIES_TEXT'] += f"<li>{message}</li>\n";

        # Now report on parcels that actually had a prediction
        df_predict_has_prediction = df_predict.loc[(df_predict[gs.prediction_status] != 'NODATA')
                                                   & (df_predict[gs.prediction_status] != 'NOT_ENOUGH_PIXELS')]
        oa = skmetrics.accuracy_score(df_predict_has_prediction[gs.class_column],
                                            df_predict_has_prediction[gs.prediction_column],
                                            normalize=True,
                                            sample_weight=None) * 100
        message = f"OA: {oa:.2f} for parcels with a prediction (= excl. NODATA, NOT_ENOUGH_PIXELS parcels), for standard prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")
        html_data['GENERAL_ACCURACIES_TEXT'] += f"<li>{message}</li>\n";


        oa = skmetrics.accuracy_score(df_predict_has_prediction[gs.class_column],
                                            df_predict_has_prediction[gs.prediction_cons_column],
                                            normalize=True,
                                            sample_weight=None) * 100
        message = f"OA: {oa:.2f} for parcels with a prediction (= excl. NODATA, NOT_ENOUGH_PIXELS parcels), for consolidated prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")
        html_data['GENERAL_ACCURACIES_TEXT'] += f"<li>{message}</li>\n";

        # Output best-case accuracy
        # Ignore the 'UNKNOWN' and 'IGNORE_' classes...
        logger.info("For best-case accuracy, remove the 'UNKNOWN' class and the classes that start with 'IGNORE_'")
        df_predict_accuracy_best_case = df_predict_has_prediction[df_predict_has_prediction[gs.class_column] != 'UNKNOWN']
        df_predict_accuracy_best_case = \
                df_predict_accuracy_best_case[~df_predict_accuracy_best_case[gs.class_column]
                                              .str.startswith('IGNORE_', na=True)]

        oa = skmetrics.accuracy_score(df_predict_accuracy_best_case[gs.class_column],
                                            df_predict_accuracy_best_case[gs.prediction_column],
                                            normalize=True,
                                            sample_weight=None) * 100
        message = f"OA: {oa:.2f} for the parcels with a prediction, excl. classes UNKNOWN and IGNORE_*, for standard prediction"
        logger.info(message)
        outputfile.write(f"{message}\n")
        html_data['GENERAL_ACCURACIES_TEXT'] += f"<li>{message}</li>\n";

        oa = skmetrics.accuracy_score(df_predict_accuracy_best_case[gs.class_column],
                                            df_predict_accuracy_best_case[gs.prediction_cons_column],
                                            normalize=True,
                                            sample_weight=None) * 100
        message = f"OA: {oa:.2f} for the parcels with a prediction, excl. classes UNKNOWN and IGNORE_*, for consolidated prediction"
        logger.info(message)
        outputfile.write(f"{message}\n\n")
        html_data['GENERAL_ACCURACIES_TEXT'] += f"<li>{message}</li>\n";

        df_pixcount = None

        # Write the recall, F1 score,... per class
#        message = skmetrics.classification_report(df_predict[gs.class_column]
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
            html_data['CONFUSION_MATRICES_TABLE'] = df_confmatrix_ext.to_html()
            html_data['CONFUSION_MATRICES_DATA'] = df_confmatrix_ext.to_json()

        # Calculate an extended confusion matrix with the consolidated prediction column and write
        # it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(df_predict, gs.prediction_cons_column)
        outputfile.write("\nExtended confusion matrix of the consolidated predictions: Rows: true/input classes, columns: predicted classes\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            outputfile.write(f"{df_confmatrix_ext}\n\n")
            html_data['CONFUSION_MATRICES_CONSOLIDATED_TABLE'] = df_confmatrix_ext.to_html()
            html_data['CONFUSION_MATRICES_CONSOLIDATED_DATA'] = df_confmatrix_ext.to_json()

        # If a ground truth file is provided, report on the ground truth
        if parcel_ground_truth_csv is not None:
            outputfile.write("************************************************************\n")
            outputfile.write("*** REPORTING ON PREDICTION QUALITY BASED ON GROUND TRUTH **\n")
            outputfile.write("************************************************************\n")

            # Read ground truth
            logger.info(f"Read csv with ground truth (with their classes): {parcel_ground_truth_csv}")
            df_parcel_gt = pd.read_csv(parcel_ground_truth_csv, low_memory=False)
            df_parcel_gt.set_index(gs.id_column, inplace=True)
            logger.info(f"Read csv with ground truth ready, shape: {df_parcel_gt.shape}")

            # Rename the classname column in ground truth
            df_parcel_gt.rename(columns={gs.class_column: gs.class_column + '_GT'},
			                    inplace=True)

            # Join the prediction data
            cols_to_join = df_predict.columns.difference(df_parcel_gt.columns)
            df_parcel_gt = df_parcel_gt.join(df_predict[cols_to_join], how='inner')
            logger.info(f"After join of ground truth with predictions, shape: {df_parcel_gt.shape}")

            if len(df_parcel_gt) == 0:
                message = "After join of ground truth with predictions the result was empty, so probably a wrong ground truth file was used!"
                logger.critical(message)
                raise Exception(message)

            # Add the alfa error
            # TODO: this needs to be reviewed, maybe need to compare with original classname
            #       instead of _IGNORE, UNKNOWN,...
            def get_prediction_quality(row, prediction_column_to_use) -> str:
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
                    elif row[prediction_column_to_use] in ['DOUBT', 'NODATA', 'NOT_ENOUGH_PIXELS']:
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
                    elif row[prediction_column_to_use] in ['DOUBT', 'NODATA', 'NOT_ENOUGH_PIXELS']:
                        # Prediction resulted in doubt or there was no/not enough data
                        return f"OK_BENEFIT_OF_DOUBT_PRED={row[prediction_column_to_use]}"
                    else:
                        # Prediction was wrong, but same as the farmer!
                        return 'OK_FARMER_WRONG_PREDICTION_DIFFERENT'

#            df_parcel_gt[PRED_QUALITY_COLUMN] = df_parcel_gt.apply(get_prediction_quality
#                                                                   , args=[gs.prediction_cons_column]
#                                                                   , axis=1).tolist()
            values = df_parcel_gt.apply(get_prediction_quality, args=[gs.prediction_column], axis=1)
            df_parcel_gt.insert(loc=2, column=PRED_QUALITY_COLUMN, value=values)
            values = df_parcel_gt.apply(get_prediction_quality, args=[gs.prediction_cons_column], axis=1)
            df_parcel_gt.insert(loc=2, column=PRED_QUALITY_CONS_COLUMN, value=values)
            df_parcel_gt.to_csv(output_report_txt + "groundtruth_pred_quality_details.csv")

            # First write the result for the standard predictions
            message = f"Prediction quality overview, for {len(df_parcel_gt)} predicted cases in ground truth:"
            outputfile.write(f"{message}\n")
            html_data['PREDICTION_QUALITY_OVERVIEW_TEXT'] = message
            
            count_per_class = (df_parcel_gt.groupby(PRED_QUALITY_COLUMN, as_index=False)
                               .size().to_frame('count'))
            values = 100*count_per_class['count']/count_per_class['count'].sum()
            count_per_class.insert(loc=1, column='pct', value=values)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                outputfile.write(f"{count_per_class}\n")
				logger.info(f"{count_per_class}\n")
                html_data['PREDICTION_QUALITY_OVERVIEW_TABLE'] = count_per_class.to_html()

            # Now write the result for the consolidated predictions
            message = f"Prediction quality cons overview, for {len(df_parcel_gt)} predicted cases in ground truth:"
            outputfile.write(f"{message}\n")
            html_data['PREDICTION_QUALITY_CONS_OVERVIEW_TEXT'] = message
            
            count_per_class = (df_parcel_gt.groupby(PRED_QUALITY_CONS_COLUMN, as_index=False)
                               .size().to_frame('count'))
            values = 100*count_per_class['count']/count_per_class['count'].sum()
            count_per_class.insert(loc=1, column='pct', value=values)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):                                
                outputfile.write(f"{count_per_class}\n")
				logger.info(f"{count_per_class}\n")
                html_data['PREDICTION_QUALITY_CONS_OVERVIEW_TABLE'] = count_per_class.to_html()

            # If the pixcount is available, write the number of ALFA errors per pixcount
            if gs.pixcount_s1s2_column in df_parcel_gt.columns:
                # Get data, drop empty lines and write
                message = f"Number of ERROR_ALFA parcels for the consolidated prediction per pixcount for the ground truth parcels:"
                outputfile.write(f"{message}\n")            
                html_data['PREDICTION_QUALITY_ALPHA_ERROR_TEXT'] = message
                
                df_per_pixcount = _get_alfa_errors_per_pixcount(df_parcel_gt)
                df_per_pixcount.dropna(inplace=True, how='all') # de eerste 10 records worden normaal weg gefilterd, omdat daar geen data voor is vanwege de negatieve 10m buffer
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):                    
                    outputfile.write(f"{df_per_pixcount}\n")
					logger.info(f"{df_per_pixcount}\n")
                    html_data['PREDICTION_QUALITY_ALPHA_ERROR_TABLE'] = df_per_pixcount.to_html()
                        
    with open(output_report_txt.replace(".txt", ".html"), 'w') as outputfile:           
        html_template_file = open('html_rapport_template.html').read()                        
        src = Template(html_template_file)
        # replace strings and write to file
        output = src.substitute(html_data)
        outputfile.write(output)

    if df_pixcount is not None:
        pixcount_output_report_txt = output_report_txt + '_pixcount.txt'
        _write_OA_per_pixcount(df_parcel_predictions=df_predict,
                               df_parcel_pixcount=df_pixcount,
                               output_report_txt=pixcount_output_report_txt,
                               force=force)

def _get_confusion_matrix_ext(df_predict, prediction_column_to_use: str):
    """ Returns a dataset with an extended confusion matrix. """

    classes = sorted(np.unique(np.append(df_predict[prediction_column_to_use].unique(),
                                         df_predict[gs.class_column].unique())))
    logger.debug(f"Input shape: {df_predict.shape}, Unique classes found: {classes}")

    # Calculate standard confusion matrix
    np_confmatrix = skmetrics.confusion_matrix(df_predict[gs.class_column],
                                               df_predict[prediction_column_to_use],
                                               labels=classes)
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
    df_count_per_pixcount.insert(loc=len(df_count_per_pixcount.columns),
                                 column='count_all_cumulative',
                                 value=values)
    values = (100 * df_count_per_pixcount['count_all_cumulative']
              / df_count_per_pixcount['count_all'].sum())
    df_count_per_pixcount.insert(loc=len(df_count_per_pixcount.columns),
                                 column='pct_all_cumulative',
                                 value=values)

    # Now calculate the number of alfa errors per pixcount
    df_alfa_error = df_predquality_pixcount[df_predquality_pixcount[PRED_QUALITY_CONS_COLUMN] == 'ERROR_ALFA']
    df_alfa_per_pixcount = (df_alfa_error.groupby(gs.pixcount_s1s2_column,
                                                  as_index=False)
                            .size().to_frame('count_error_alfa'))

    # Join them together, and calculate the alfa error percentages
    df_alfa_per_pixcount = df_count_per_pixcount.join(df_alfa_per_pixcount
                                                      , how='left')
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns),
                                column='count_error_alfa_cumulative',
                                value=df_alfa_per_pixcount['count_error_alfa'].cumsum(axis=0))
    values = 100 * df_alfa_per_pixcount['count_error_alfa'] / df_alfa_per_pixcount['count_all']
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns),
                                column='pct_error_alfa_of_all', value=values)
    values = (100 * df_alfa_per_pixcount['count_error_alfa_cumulative']
              / df_alfa_per_pixcount['count_error_alfa'].sum())
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns),
                                column='pct_error_alfa_of_alfa_cumulative', value=values)
    values = (100 * df_alfa_per_pixcount['count_error_alfa_cumulative']
              / df_alfa_per_pixcount['count_all'].sum())
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns),
                                column='pct_error_alfa_of_all_cumulative', value=values)

    return df_alfa_per_pixcount

def _write_OA_per_pixcount(df_parcel_predictions: pd.DataFrame,
                           df_parcel_pixcount: pd.DataFrame,
                           output_report_txt: str,
                           force: bool = False):
    """ Write a report of the overall accuracy that parcels per pixcount get. """
    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    # Set the index to the id_columnname, and join the data to the result...
    df_parcel_predictions.set_index(gs.id_column, inplace=True)
    df_parcel_pixcount.set_index(gs.id_column, inplace=True)
    df_result = df_parcel_predictions.join(df_parcel_pixcount, how='inner')
    nb_predictions_total = len(df_result.index)

    # Write output...
    with open(output_report_txt, 'w') as outputfile:
        for i in range(40):

            df_result_cur_pixcount = df_result[df_result[gs.pixcount_s1s2_column] == i]
            nb_predictions_pixcount = len(df_result_cur_pixcount.index)
            if nb_predictions_pixcount == 0:
                continue

            overall_accuracy = 100.0*skmetrics.accuracy_score(df_result_cur_pixcount[gs.class_column], df_result_cur_pixcount[gs.prediction_column], normalize=True, sample_weight=None)
            message = f"OA for pixcount {i:2}: {overall_accuracy:3.2f} %, with {nb_predictions_pixcount} elements ({100*(nb_predictions_pixcount/nb_predictions_total):.4f} % of {nb_predictions_total})"
            logger.info(message)
            outputfile.write(f"{message}\n")

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
