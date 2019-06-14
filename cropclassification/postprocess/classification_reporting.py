# -*- coding: utf-8 -*-
"""
Module with some helper functions to report on the classification results.
"""

import logging
import os

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from string import Template

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

# TODO: improve reporting to devide between eligible versus ineligible classes?
# TODO?: report based on area instead of number parcel
#     -> seems like being a bit "detached from reality", as for RFV the most important parameter is the number of parcels

def write_full_report(parcel_predictions_filepath: str,
                      output_report_txt: str,
                      parcel_ground_truth_filepath: str = None,
                      force: bool = None):
    """Writes a report about the accuracy of the predictions to a file.

    Args:
        parcel_predictions_filepath: File name of csv file with the parcel with their predictions.
        prediction_columnname: Column name of the column that contains the predictions.
        output_report_txt: File name of txt file the report will be written to.
        parcel_ground_truth_filepath: List of parcels with ground truth to calculate eg. alfa and
            beta errors. If None, the part of the report that is based on this data is skipped

    TODO: refactor function to split logic more...
    """

    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    logger.info("Start write_full_report")

    pandas_option_context_list = ['display.max_rows', None, 'display.max_columns', None, 
                                  'display.max_colwidth', 300, 'display.width', 2000, 
                                  'display.colheader_justify', 'left']
    logger.info(f"Read file with predictions: {parcel_predictions_filepath}")
    df_predict = pdh.read_file(parcel_predictions_filepath)
    df_predict.set_index(conf.columns['id'], inplace=True)
        
    # Python template engine expects all values to be present, so initialize to empty
    empty_string = "''"
    html_data = {
       'GENERAL_ACCURACIES_TABLE': empty_string,
       'GENERAL_ACCURACIES_TEXT': empty_string,
       'GENERAL_ACCURACIES_DATA': empty_string,
       'CONFUSION_MATRICES_TABLE': empty_string,
       'CONFUSION_MATRICES_DATA': empty_string,
       'CONFUSION_MATRICES_CONSOLIDATED_TABLE': empty_string,
       'CONFUSION_MATRICES_CONSOLIDATED_DATA': empty_string,
       'PREDICTION_QUALITY_CONS_OVERVIEW_TEXT': empty_string,
       'PREDICTION_QUALITY_CONS_OVERVIEW_TABLE': empty_string,
       'PREDICTION_QUALITY_ALPHA_TEXT': empty_string,
       'PREDICTION_QUALITY_BETA_TEXT': empty_string,
       'PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TEXT': empty_string,
       'PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TABLE': empty_string
    }
    
    # Build and write report...
    with open(output_report_txt, 'w') as outputfile:

        outputfile.write("************************************************************\n")
        outputfile.write("********************* PARAMETERS USED **********************\n")
        outputfile.write("************************************************************\n")
        outputfile.write("\n")
        message = "Main parameters used for the marker"
        outputfile.write(f"\n{message}\n")
        html_data['PARAMETERS_USED_TEXT'] = message

        logger.info(f"{dict(conf.marker)}")
        parameter_list = [['marker', key, value] for key, value in conf.marker.items()]
        parameter_list += [['preprocess', key, value] for key, value in conf.preprocess.items()]
        parameter_list += [['classifier', key, value] for key, value in conf.classifier.items()]
        parameter_list += [['postprocess', key, value] for key, value in conf.postprocess.items()]
        
        parameters_used_df = pd.DataFrame(parameter_list, columns=['parameter_type', 'parameter', 'value'])
        with pd.option_context(*pandas_option_context_list):
            outputfile.write(f"\n{parameters_used_df}\n")
            logger.info(f"{parameters_used_df}\n")
            html_data['PARAMETERS_USED_TABLE'] = parameters_used_df.to_html(index=False) 
              
        outputfile.write("************************************************************\n")
        outputfile.write("**************** RECAP OF GENERAL RESULTS ******************\n")
        outputfile.write("************************************************************\n")
        outputfile.write("\n")
        outputfile.write("************************************************************\n")
        outputfile.write("*             GENERAL CONSOLIDATED CONCLUSIONS             *\n")
        outputfile.write("************************************************************\n")
        # Calculate + write general conclusions for consolidated prediction
        _add_prediction_conclusion(in_df=df_predict,
                                   new_columnname=conf.columns['prediction_conclusion_cons'],
                                   prediction_column_to_use=conf.columns['prediction_cons'],
                                   detailed=False)
        message = f"Prediction conclusions cons (doubt + not_enough_pixels) overview, for {len(df_predict.index)} predicted cases:"
        outputfile.write(f"\n{message}\n")
        html_data['GENERAL_PREDICTION_CONCLUSION_CONS_OVERVIEW_TEXT'] = message
        
        count_per_class = (df_predict.groupby(conf.columns['prediction_conclusion_cons'], as_index=False)
                           .size().to_frame('count'))
        values = 100*count_per_class['count']/count_per_class['count'].sum()
        count_per_class.insert(loc=1, column='pct', value=values)
        
        with pd.option_context(*pandas_option_context_list):                                
            outputfile.write(f"\n{count_per_class}\n")
            logger.info(f"{count_per_class}\n")
            html_data['GENERAL_PREDICTION_CONCLUSION_CONS_OVERVIEW_TABLE'] = count_per_class.to_html()
            html_data['GENERAL_PREDICTION_CONCLUSION_CONS_OVERVIEW_DATA'] = count_per_class.to_dict()

        # Output general accuracies
        outputfile.write("************************************************************\n")
        outputfile.write("*                   OVERALL ACCURACIES                     *\n")
        outputfile.write("************************************************************\n")
        overall_accuracies_list = []

        # Calculate overall accuracies for all parcels
        oa = skmetrics.accuracy_score(df_predict[conf.columns['class']],
                                      df_predict[conf.columns['prediction']],
                                      normalize=True,
                                      sample_weight=None) * 100
        overall_accuracies_list.append({'parcels': 'All', 
                                        'prediction_type': 'standard', 
                                        'accuracy': oa})

        oa = skmetrics.accuracy_score(df_predict[conf.columns['class']],
                                      df_predict[conf.columns['prediction_cons']],
                                      normalize=True,
                                      sample_weight=None) * 100
        overall_accuracies_list.append({'parcels': 'All', 
                                        'prediction_type': 'consolidated', 
                                        'accuracy': oa})

        # Calculate while ignoring the classes to be ignored...
        df_predict_accuracy_no_ignore = df_predict[
                ~df_predict[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore_for_train'))]
        df_predict_accuracy_no_ignore = df_predict_accuracy_no_ignore[
                ~df_predict_accuracy_no_ignore[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore'))]
        
        oa = skmetrics.accuracy_score(df_predict_accuracy_no_ignore[conf.columns['class']],
                                      df_predict_accuracy_no_ignore[conf.columns['prediction']],
                                      normalize=True,
                                      sample_weight=None) * 100
        overall_accuracies_list.append({'parcels': 'Exclude classes_to_ignore(_for_train) classes', 
                                        'prediction_type': 'standard', 
                                        'accuracy': oa})

        oa = skmetrics.accuracy_score(df_predict_accuracy_no_ignore[conf.columns['class']],
                                      df_predict_accuracy_no_ignore[conf.columns['prediction_cons']],
                                      normalize=True,
                                      sample_weight=None) * 100
        overall_accuracies_list.append({'parcels': 'Exclude classes_to_ignore(_for_train) classes', 
                                        'prediction_type': 'consolidated', 
                                        'accuracy': oa})

        # Calculate ignoring both classes to ignored + parcels not having a valid prediction
        df_predict_no_ignore_has_prediction = df_predict_accuracy_no_ignore.loc[
                (df_predict_accuracy_no_ignore[conf.columns['prediction_status']] != 'NODATA')
                        & (df_predict_accuracy_no_ignore[conf.columns['prediction_status']] != 'NOT_ENOUGH_PIXELS')]
        oa = skmetrics.accuracy_score(df_predict_no_ignore_has_prediction[conf.columns['class']],
                                      df_predict_no_ignore_has_prediction[conf.columns['prediction']],
                                      normalize=True,
                                      sample_weight=None) * 100
        overall_accuracies_list.append({'parcels': 'Exclude ignored ones + with prediction (= excl. NODATA, NOT_ENOUGH_PIXELS)', 
                                        'prediction_type': 'standard', 'accuracy': oa})

        oa = skmetrics.accuracy_score(df_predict_no_ignore_has_prediction[conf.columns['class']],
                                      df_predict_no_ignore_has_prediction[conf.columns['prediction_cons']],
                                      normalize=True,
                                      sample_weight=None) * 100
        overall_accuracies_list.append({'parcels': 'Exclude ignored ones + with prediction (= excl. NODATA, NOT_ENOUGH_PIXELS)', 
                                        'prediction_type': 'consolidated', 'accuracy': oa})

        # Output the resulting overall accuracies
        message = 'Overall accuracies for different sub-groups of the data'
        outputfile.write(f"\n{message}\n")
        html_data['OVERALL_ACCURACIES_TEXT'] = message

        overall_accuracies_df = pd.DataFrame(overall_accuracies_list, 
                                             columns=['parcels', 'prediction_type', 'accuracy'])
        overall_accuracies_df.set_index(keys=['parcels', 'prediction_type'], inplace=True)
        with pd.option_context(*pandas_option_context_list):
            outputfile.write(f"\n{overall_accuracies_df}\n")
            logger.info(f"{overall_accuracies_df}\n")
            html_data['OVERALL_ACCURACIES_TABLE'] = overall_accuracies_df.to_html()        

        # Write the recall, F1 score,... per class
        #message = skmetrics.classification_report(df_predict[gs.class_column]
        #                                                , df_predict[gs.prediction_column]
        #                                                , labels=classes)
        #outputfile.write(message)

        outputfile.write("************************************************************\n")
        outputfile.write("********************* DETAILED RESULTS *********************\n")
        outputfile.write("************************************************************\n")
        outputfile.write("\n")
        outputfile.write("************************************************************\n")
        outputfile.write("*             DETAILED PREDICTION CONCLUSIONS              *\n")
        outputfile.write("************************************************************\n")

        # Calculate detailed conclusions for the predictions
        logger.info("Calculate the detailed conclusions for the predictions")

        # Write the conclusions for the consolidated predictions
        _add_prediction_conclusion(in_df=df_predict,
                                   new_columnname=conf.columns['prediction_conclusion_detail_cons'],
                                   prediction_column_to_use=conf.columns['prediction_cons'],
                                   detailed=True)
        message = f"Prediction conclusions cons (doubt + not_enough_pixels) overview, for {len(df_predict.index)} predicted cases:"
        outputfile.write(f"\n{message}\n")
        html_data['PREDICTION_CONCLUSION_DETAIL_CONS_OVERVIEW_TEXT'] = message
        
        count_per_class = (df_predict.groupby(conf.columns['prediction_conclusion_detail_cons'], as_index=False)
                            .size().to_frame('count'))
        values = 100*count_per_class['count']/count_per_class['count'].sum()
        count_per_class.insert(loc=1, column='pct', value=values)
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):                                
            outputfile.write(f"\n{count_per_class}\n")
            logger.info(f"{count_per_class}\n")
            html_data['PREDICTION_CONCLUSION_DETAIL_CONS_OVERVIEW_TABLE'] = count_per_class.to_html()

        outputfile.write("************************************************************\n")
        outputfile.write("*     CONFUSION MATRICES FOR PARCELS WITH PREDICTIONS      *\n")
        outputfile.write("************************************************************\n")
        # Calculate an extended confusion matrix with the standard prediction column and write
        # it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(df_predict, conf.columns['prediction'])
        outputfile.write("\nExtended confusion matrix of the predictions: Rows: true/input classes, columns: predicted classes\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            outputfile.write(f"{df_confmatrix_ext}\n")
            html_data['CONFUSION_MATRICES_TABLE'] = df_confmatrix_ext.to_html()
            html_data['CONFUSION_MATRICES_DATA'] = df_confmatrix_ext.to_json()

        # Calculate an extended confusion matrix with the consolidated prediction column and write
        # it to output...
        df_confmatrix_ext = _get_confusion_matrix_ext(df_predict, conf.columns['prediction_cons'])
        outputfile.write("\nExtended confusion matrix of the consolidated predictions: Rows: true/input classes, columns: predicted classes\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
            outputfile.write(f"{df_confmatrix_ext}\n\n")
            html_data['CONFUSION_MATRICES_CONSOLIDATED_TABLE'] = df_confmatrix_ext.to_html()
            html_data['CONFUSION_MATRICES_CONSOLIDATED_DATA'] = df_confmatrix_ext.to_json()

        # If the pixcount is available, write the OA per pixcount
        if conf.columns['pixcount_s1s2'] in df_predict.columns:
            pixcount_output_report_txt = output_report_txt + '_OA_per_pixcount.txt'
            _write_OA_per_pixcount(df_parcel_predictions=df_predict,
                                   output_report_txt=pixcount_output_report_txt,
                                   force=force)
        
        # If a ground truth file is provided, report on the ground truth
        if parcel_ground_truth_filepath is not None:
            outputfile.write("************************************************************\n")
            outputfile.write("*   REPORTING ON PREDICTION QUALITY BASED ON GROUND TRUTH  *\n")
            outputfile.write("************************************************************\n")

            # Read ground truth
            logger.info(f"Read csv with ground truth (with their classes): {parcel_ground_truth_filepath}")
            df_parcel_gt = pdh.read_file(parcel_ground_truth_filepath)
            df_parcel_gt.set_index(conf.columns['id'], inplace=True)
            logger.info(f"Read csv with ground truth ready, shape: {df_parcel_gt.shape}")

            # Join the prediction data
            cols_to_join = df_predict.columns.difference(df_parcel_gt.columns)
            df_parcel_gt = df_predict[cols_to_join].join(df_parcel_gt, how='inner')
            logger.info(f"After join of ground truth with predictions, shape: {df_parcel_gt.shape}")

            if len(df_parcel_gt.index) == 0:
                message = "After join of ground truth with predictions the result was empty, so probably a wrong ground truth file was used!"
                logger.critical(message)
                raise Exception(message)

            # General ground truth statistics
            # ******************************************************************
            # Calculate the conclusions based on ground truth
           
            # Calculate and write the result for the consolidated predictions
            _add_gt_conclusions(df_parcel_gt, conf.columns['prediction_cons'])
            message = f"Prediction quality cons (doubt + not_enough_pixels) overview, for {len(df_parcel_gt.index)} predicted cases in ground truth:"
            outputfile.write(f"\n{message}\n")
            html_data['PREDICTION_QUALITY_CONS_OVERVIEW_TEXT'] = message
            
            count_per_class = (df_parcel_gt.groupby(f"gt_conclusion_{conf.columns['prediction_cons']}", as_index=False)
                               .size().to_frame('count'))
            values = 100*count_per_class['count']/count_per_class['count'].sum()
            count_per_class.insert(loc=1, column='pct', value=values)
            
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):                                
                outputfile.write(f"\n{count_per_class}\n")
                logger.info(f"{count_per_class}\n")
                html_data['PREDICTION_QUALITY_CONS_OVERVIEW_TABLE'] = count_per_class.to_html()

            # Write the ground truth conclusions to file
            pdh.to_file(df_parcel_gt, output_report_txt + "_groundtruth_pred_quality_details.tsv")

            # Alpha and beta error statistics
            # ******************************************************************            
            # Pct Alpha errors=alpha errors/(alpha errors + real errors)  
            columnname = f"gt_conclusion_{conf.columns['prediction_cons']}"
            alpha_error_numerator = len(df_parcel_gt.loc[df_parcel_gt[columnname] == 'FARMER-OK_PRED-WRONG:ERROR_ALPHA'].index)
            alpha_error_denominator = (alpha_error_numerator 
                    + len(df_parcel_gt.loc[df_parcel_gt[columnname].isin(
                            ['FARMER-WRONG_PRED-OK', 'FARMER-WRONG_PRED-WRONG'])].index))
            if alpha_error_denominator > 0:
                message = (f"Alpha error: {alpha_error_numerator}/{alpha_error_denominator} = "
                        + f"{(alpha_error_numerator/alpha_error_denominator):.02f}")
            else:
                message = f"Alpha error: {alpha_error_numerator}/{alpha_error_denominator} = ?"

            outputfile.write(f"\n{message}\n")
            html_data['PREDICTION_QUALITY_ALPHA_TEXT'] = message

            beta_error_numerator = len(df_parcel_gt.loc[df_parcel_gt[columnname] == 'FARMER-WRONG_PRED-DOESNT_OPPOSE:ERROR_BETA'].index)
            beta_error_denominator = (beta_error_numerator 
                    + len(df_parcel_gt.loc[df_parcel_gt[columnname].str.startswith('FARMER-WRONG_PRED-')].index))
            if beta_error_denominator > 0:
                message = (f"Beta error: {beta_error_numerator}/{beta_error_denominator} = "
                           + f"{(beta_error_numerator/beta_error_denominator):.02f}")
            else:
                message = f"Beta error: {beta_error_numerator}/{beta_error_denominator} = ?"

            outputfile.write(f"\n{message}\n")
            html_data['PREDICTION_QUALITY_BETA_TEXT'] = message

            # If the pixcount is available, write the number of ALFA errors per pixcount (for the prediction with doubt)
            if conf.columns['pixcount_s1s2'] in df_parcel_gt.columns:
                # Get data, drop empty lines and write
                message = f"Number of ERROR_ALFA parcels for the 'prediction with doubt' per pixcount for the ground truth parcels:"
                outputfile.write(f"\n{message}\n")            
                html_data['PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TEXT'] = message
                
                # To get the number of alpha errors per pixcount, we also need alpha errors
                # also for parcels that had not_enough_pixels, so we need prediction_withdoubt
                # If they don't exist, calculate
                _add_gt_conclusions(df_parcel_gt, conf.columns['prediction_withdoubt'])
            
                df_per_pixcount = _get_alfa_errors_per_pixcount(
                        df_predquality_pixcount=df_parcel_gt,
                        pred_quality_column=f"gt_conclusion_{conf.columns['prediction_withdoubt']}",
                        error_alpha_code='FARMER-OK_PRED-WRONG:ERROR_ALPHA')
                df_per_pixcount.dropna(inplace=True)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):                    
                    outputfile.write(f"\n{df_per_pixcount}\n")
                    logger.info(f"{df_per_pixcount}\n")
                    html_data['PREDICTION_QUALITY_ALPHA_PER_PIXCOUNT_TABLE'] = df_per_pixcount.to_html()
                        
    with open(output_report_txt.replace('.txt', '.html'), 'w') as outputfile:           
        html_template_file = open('./cropclassification/postprocess/html_rapport_template.html').read()                        
        src = Template(html_template_file)
        # replace strings and write to file
        output = src.substitute(html_data)
        outputfile.write(output)

def _get_confusion_matrix_ext(df_predict, prediction_column_to_use: str):
    """ Returns a dataset with an extended confusion matrix. """

    classes = sorted(np.unique(np.append(df_predict[prediction_column_to_use],
                                         df_predict[conf.columns['class']])))
    logger.debug(f"Input shape: {df_predict.shape}, Unique classes found: {classes}")

    # Calculate standard confusion matrix
    np_confmatrix = skmetrics.confusion_matrix(df_predict[conf.columns['class']],
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

def _add_prediction_conclusion(in_df,
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
    all_classes_to_ignore = (conf.marker.getlist('classes_to_ignore_for_train') 
                             + conf.marker.getlist('classes_to_ignore'))

    # Some conclusions are different is detailed info is asked...
    if detailed == True:
        # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
        in_df.loc[in_df[conf.columns['class']].isin(all_classes_to_ignore),
                  new_columnname] = 'IGNORE:INPUTCLASSNAME=' + in_df[conf.columns['class']].map(str)
        # If conclusion still UNDEFINED, check if doubt 
        in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(conf.marker.getlist('classes_doubt'))),
                  new_columnname] = 'DOUBT:REASON=' + in_df[prediction_column_to_use].map(str)
    else:
        # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
        in_df.loc[in_df[conf.columns['class']].isin(all_classes_to_ignore),
                  new_columnname] = 'IGNORE'
        # If conclusion still UNDEFINED, check if doubt 
        in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(conf.marker.getlist('classes_doubt'))),
                  new_columnname] = 'DOUBT'

    # If conclusion still UNDEFINED, check if prediction equals the input class 
    in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                & (in_df[conf.columns['class']] == in_df[prediction_column_to_use]),
              new_columnname] = 'OK:PREDICTION=INPUT_CLASS'
    # If conclusion still UNDEFINED, prediction is different from input 
    in_df.loc[in_df[new_columnname] == 'UNDEFINED',
              new_columnname] = 'NOK:PREDICTION<>INPUT_CLASS'

def _add_gt_conclusions(in_df, 
                        prediction_column_to_use) -> str:
    """ Add some columns with groundtruth conclusions. """
    
    # Add the new column with a fixed value first
    gt_vs_input_column = f"gt_vs_input_{prediction_column_to_use}"
    gt_vs_prediction_column = f"gt_vs_prediction_{prediction_column_to_use}"
    gt_conclusion_column = f"gt_conclusion_{prediction_column_to_use}"
    all_classes_to_ignore = (conf.marker.getlist('classes_to_ignore_for_train') 
                                + conf.marker.getlist('classes_to_ignore'))

    # Calculate gt_vs_input_column
    # If ground truth same as input class, farmer OK, unless it is an ignore class
    in_df[gt_vs_input_column] = 'FARMER-WRONG'
    in_df.loc[(in_df[conf.columns['class_groundtruth_unverified']] == in_df[conf.columns['class_groundtruth_verified']]),
              gt_vs_input_column] = 'FARMER-OK'
    in_df.loc[(in_df[conf.columns['class_groundtruth_unverified']] == in_df[conf.columns['class_groundtruth_verified']])
                    & (in_df[conf.columns['class_groundtruth_verified']].isin(all_classes_to_ignore)),
              gt_vs_input_column] = 'FARMER-OK:IGNORE:VERIFIED=UNVERIFIED=' + in_df[conf.columns['class_groundtruth_verified']].map(str)
    in_df.loc[(in_df[gt_vs_input_column] == 'FARMER-WRONG')
                    & (in_df[conf.columns['class_groundtruth_verified']].isin(all_classes_to_ignore)),
              gt_vs_input_column] = 'FARMER-WRONG:IGNORE:VERIFIEDCLASSNAME=' + in_df[conf.columns['class_groundtruth_verified']].map(str)
    in_df.loc[(in_df[gt_vs_input_column] == 'FARMER-WRONG')
                    & (in_df[conf.columns['class_groundtruth_unverified']].isin(all_classes_to_ignore)),
              gt_vs_input_column] = 'FARMER-WRONG:IGNORE:UNVERIFIEDCLASSNAME=' + in_df[conf.columns['class_groundtruth_unverified']].map(str)

    # Calculate gt_vs_prediction_column
    # If ground truth same as prediction, prediction OK 
    in_df[gt_vs_prediction_column] = 'UNDEFINED'
    in_df.loc[(in_df[prediction_column_to_use] == in_df[conf.columns['class_groundtruth_verified']]),
              gt_vs_prediction_column] = 'PRED-OK'
    in_df.loc[(in_df[prediction_column_to_use] == in_df[conf.columns['class_groundtruth_verified']])
                    & (in_df[prediction_column_to_use].isin(all_classes_to_ignore)),
              gt_vs_prediction_column] = 'PRED-OK:IGNORE:PREDICTION=VERIFIED=' + in_df[prediction_column_to_use].map(str)                

    # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
    in_df.loc[(in_df[gt_vs_prediction_column] == 'UNDEFINED')
                    & (in_df[conf.columns['class_groundtruth_verified']].isin(all_classes_to_ignore)),
              gt_vs_prediction_column] = 'PRED-WRONG:IGNORE:VERIFIEDCLASSNAME=' + in_df[conf.columns['class_groundtruth_verified']].map(str)
    in_df.loc[(in_df[gt_vs_prediction_column] == 'UNDEFINED')
                    & (in_df[conf.columns['class_groundtruth_unverified']].isin(all_classes_to_ignore)),
              gt_vs_prediction_column] = 'PRED-WRONG:IGNORE:UNVERIFIEDCLASSNAME=' + in_df[conf.columns['class_groundtruth_unverified']].map(str)
    in_df.loc[(in_df[gt_vs_prediction_column] == 'UNDEFINED')
                    & (in_df[conf.columns['class']].isin(all_classes_to_ignore)),
              gt_vs_prediction_column] = 'PRED-NONE:IGNORE:INPUTCLASSNAME=' + in_df[conf.columns['class']].map(str)

    # If conclusion still UNDEFINED, check if doubt 
    in_df.loc[(in_df[gt_vs_prediction_column] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(conf.marker.getlist('classes_doubt'))),
              gt_vs_prediction_column] = 'PRED-DOUBT:REASON=' + in_df[prediction_column_to_use].map(str)

    # If conclusion still UNDEFINED, it was wrong 
    in_df.loc[in_df[gt_vs_prediction_column] == 'UNDEFINED',
              gt_vs_prediction_column] = 'PRED-WRONG'

    # Calculate gt_conclusion_column
    # Unverified class was correct  
    in_df[gt_conclusion_column] = 'UNDEFINED'
    in_df.loc[(in_df[gt_vs_input_column] == 'FARMER-OK')
                    & (in_df[gt_vs_prediction_column] == 'PRED-WRONG'),
              gt_conclusion_column] = 'FARMER-OK_PRED-WRONG:ERROR_ALPHA'
    in_df.loc[(in_df[gt_conclusion_column] == 'UNDEFINED')
                    & (in_df[gt_vs_input_column] == 'FARMER-OK'),
                gt_conclusion_column] = 'FARMER-OK_' + in_df[gt_vs_prediction_column].map(str)

    # Unverified class was not correct
    in_df.loc[(in_df[gt_conclusion_column] == 'UNDEFINED')
                    & (in_df[gt_vs_input_column] == 'FARMER-WRONG')
                    & (in_df[conf.columns['class_groundtruth_unverified']] == in_df[prediction_column_to_use]),
              gt_conclusion_column] = 'FARMER-WRONG_PRED-DOESNT_OPPOSE:ERROR_BETA'
    in_df.loc[(in_df[gt_conclusion_column] == 'UNDEFINED')
                    & (in_df[gt_vs_input_column] == 'FARMER-WRONG'),
              gt_conclusion_column] = 'FARMER-WRONG_' + in_df[gt_vs_prediction_column].map(str)
    
    # Unverified or verified class was ignore
    in_df.loc[(in_df[gt_conclusion_column] == 'UNDEFINED'),
              gt_conclusion_column] = in_df[gt_vs_input_column].map(str)
                            
def _get_alfa_errors_per_pixcount(df_predquality_pixcount,
                                  pred_quality_column: str,
                                  error_alpha_code: str):
    """ Returns a dataset with detailed information about the number of alfa errors per pixcount """

    # Calculate the number of parcels per pixcount, the cumulative sum + the pct of all
    df_count_per_pixcount = (df_predquality_pixcount.groupby(conf.columns['pixcount_s1s2'], as_index=False)
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
    df_alfa_error = df_predquality_pixcount[df_predquality_pixcount[pred_quality_column] == error_alpha_code]
    df_alfa_per_pixcount = (df_alfa_error.groupby(conf.columns['pixcount_s1s2'], as_index=False)
                            .size().to_frame('count_error_alfa'))

    # Join them together, and calculate the alfa error percentages
    df_alfa_per_pixcount = df_count_per_pixcount.join(df_alfa_per_pixcount, how='left')
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns),
                                column='count_error_alfa_cumulative',
                                value=df_alfa_per_pixcount['count_error_alfa'].cumsum(axis=0))
                                
    values = 100 * df_alfa_per_pixcount['count_error_alfa'] / df_alfa_per_pixcount['count_all']
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns), column='pct_error_alfa_of_all', value=values)
                                
    values = (100 * df_alfa_per_pixcount['count_error_alfa_cumulative'] / df_alfa_per_pixcount['count_error_alfa'].sum())
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns), column='pct_error_alfa_of_alfa_cumulative', value=values)

    values = (100 * df_alfa_per_pixcount['count_error_alfa_cumulative'] / df_alfa_per_pixcount['count_all'].sum())
    df_alfa_per_pixcount.insert(loc=len(df_alfa_per_pixcount.columns), column='pct_error_alfa_of_all_cumulative', value=values)

    return df_alfa_per_pixcount

def _write_OA_per_pixcount(df_parcel_predictions: pd.DataFrame,
                           output_report_txt: str,
                           force: bool = False):
    """ Write a report of the overall accuracy that parcels per pixcount get. """
    # If force == False Check and the output file exists already, stop.
    if force is False and os.path.exists(output_report_txt):
        logger.warning(f"collect_and_prepare_timeseries_data: output file already exists and force == False, so stop: {output_report_txt}")
        return

    # Write output...
    nb_predictions_total = len(df_parcel_predictions.index)
    with open(output_report_txt, 'w') as outputfile:
        for i in range(40):

            df_result_cur_pixcount = df_parcel_predictions[df_parcel_predictions[conf.columns['pixcount_s1s2']] == i]
            nb_predictions_pixcount = len(df_result_cur_pixcount.index)
            if nb_predictions_pixcount == 0:
                continue

            overall_accuracy = 100.0*skmetrics.accuracy_score(df_result_cur_pixcount[conf.columns['class']], df_result_cur_pixcount[conf.columns['prediction']], normalize=True, sample_weight=None)
            message = f"OA for pixcount {i:2}: {overall_accuracy:3.2f} %, with {nb_predictions_pixcount} elements ({100*(nb_predictions_pixcount/nb_predictions_total):.4f} % of {nb_predictions_total})"
            logger.info(message)
            outputfile.write(f"{message}\n")

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
