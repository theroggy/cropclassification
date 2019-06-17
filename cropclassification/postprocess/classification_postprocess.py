# -*- coding: utf-8 -*-
"""
Module with postprocessing functions on classification results.
"""

import datetime
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
                                output_predictions_output_filepath: str = None,
                                force: bool = False):
    """Calculate the top3 prediction and a consolidation prediction.
    
    Args:
        input_parcel_filepath (str): [description]
        input_parcel_probabilities_filepath (str): [description]
        output_predictions_filepath (str): [description]
        output_predictions_output_filepath (str, optional): [description]. Defaults to None.
        force (bool, optional): [description]. Defaults to False.
    """
    # If force is false and output exists, already, return
    if(force is False
       and os.path.exists(output_predictions_filepath)):
        logger.warning(f"calc_top3_and_consolidation: output file exist and force is False, so stop: {output_predictions_filepath}")
        return

    # Read input files
    logger.info("Read input file")
    proba_df = pdh.read_file(input_parcel_probabilities_filepath)

    top3_df = calc_top3(proba_df)

    # Read input files
    logger.info("Read input file")
    input_parcel_df = pdh.read_file(input_parcel_filepath)

    # All input parcels must stay in the output, so left join input with pred
    top3_df.set_index(conf.columns['id'], inplace=True)
    if input_parcel_df.index.name != conf.columns['id']:
        input_parcel_df.set_index(conf.columns['id'], inplace=True)
    cols_to_join = top3_df.columns.difference(input_parcel_df.columns)
    pred_df = input_parcel_df.join(top3_df[cols_to_join], how='left')

    add_doubt_columns(pred_df, classified_classes=proba_df.columns.to_list())

    logger.info("Write final prediction data to file")
    pdh.to_file(pred_df, output_predictions_filepath)

    # Create final output file with the most important info
    if output_predictions_output_filepath is not None:

        # First add some aditional columns specific for the export
        pred_df['markercode'] = conf.marker['markertype']
        pred_df['run_id'] = conf.general['run_id']
        today = datetime.date.today()
        pred_df['cons_date'] = today
        pred_df['modify_date'] = today
        logger.info("Write final output prediction data to file")
        pred_df.reset_index(inplace=True)
        pred_df = pred_df[conf.columns.getlist('output_columns')] 
        pdh.to_file(pred_df, output_predictions_output_filepath, index=False) 

        # Write oracle sqlldr file
        if conf.marker['markertype'] in ['LANDCOVER', 'LANDCOVER_EARLY']:
            table_name = 'mon_marker_landcover'
            table_columns = ("layer_id, prc_id, versienummer, markercode, run_id, cons_landcover, "
                          + "cons_status, cons_date date 'yyyy-mm-dd', landcover1, probability1, "
                          + "landcover2, probability2, landcover3, probability3, "
                          + "modify_date date 'yyyy-mm-dd'")
        elif conf.marker['markertype'] in ['CROPGROUP', 'CROPGROUP_EARLY']:
            table_name = 'mon_marker_cropgroup'
            table_columns = ("layer_id, prc_id, versienummer, markercode, run_id, cons_cropgroup, "
                          + "cons_status, cons_date date 'yyyy-mm-dd', cropgroup1, probability1, "
                          + "cropgroup2, probability2, cropgroup3, probability3, "
                          + "modify_date date 'yyyy-mm-dd'")
        else: 
            table_name = None
            logger.warning(f"Table unknown for marker type {conf.marker['markertype']}, so cannot write .ctl file")

        if table_name is not None:
            with open(output_predictions_output_filepath + '.ctl', 'w') as ctlfile:
                # SKIP=1 to skip the columns names line, the other ones to evade 
                # more commits than needed
                ctlfile.write("OPTIONS (SKIP=1, ROWS=10000, BINDSIZE=40000000, READSIZE=40000000)\n")     
                ctlfile.write("LOAD DATA\n")
                ctlfile.write(f"INFILE '{os.path.basename(output_predictions_output_filepath)}'  \"str '\\n'\"\n")
                ctlfile.write(f"INSERT INTO TABLE {table_name} APPEND\n")
                # A tab as seperator is apparently X'9'  
                ctlfile.write("FIELDS TERMINATED BY X'9'\n")
                ctlfile.write(f"({table_columns})\n")

def calc_top3(proba_df: pd.DataFrame) -> pd.DataFrame:

    # Calculate the top 3 predictions
    logger.info("Calculate top3")
    proba_tmp_df = proba_df.copy()
    for column in proba_tmp_df.columns:
        if column in conf.preprocess.getlist('dedicated_data_columns'):
            proba_tmp_df.drop(column, axis=1, inplace=True)

    # Get the top 3 predictions for each row
    # First get the indeces of the top 3 predictions for each row
    # Remark: argsort sorts ascending, so we need to take:
    #     - "[:,": for all rows
    #     - ":-4": the last 3 elements of the values
    #     - ":-1]": and than reverse the order with a negative step
    top3_pred_classes_idx = np.argsort(proba_tmp_df.values, axis=1)[:, :-4:-1]
    # Convert the indeces to classes
    top3_pred_classes = np.take(proba_tmp_df.columns, top3_pred_classes_idx)
    # Get the values of the top 3 predictions
    top3_pred_values = np.sort(proba_tmp_df.values, axis=1)[:, :-4:-1]
    # Concatenate both
    top3_pred = np.concatenate([top3_pred_classes, top3_pred_values], axis=1)
    # Concatenate the ids, the classes and the top3 predictions
    id_class_top3 = np.concatenate(
            [proba_df[[conf.columns['id'], conf.columns['class']]].values, top3_pred], axis=1)

    # Convert to dataframe
    top3_df = pd.DataFrame(id_class_top3,
                           columns=[conf.columns['id'], conf.columns['class'],
                                    conf.columns['prediction'], 'pred2', 'pred3',
                                    'pred1_prob', 'pred2_prob', 'pred3_prob'])

    return top3_df

def add_doubt_columns(pred_df: pd.DataFrame,
                      classified_classes: []):

    # For the ignore classes, set the prediction to the ignore type
    classes_to_ignore = conf.marker.getlist('classes_to_ignore')
    pred_df.loc[pred_df[conf.columns['class']].isin(classes_to_ignore), 
                [conf.columns['prediction']]] = pred_df[conf.columns['class']]

    # For all other parcels without prediction there must have been no data 
    # available for a classification, so set prediction to NODATA
    pred_df[conf.columns['prediction']].fillna('NODATA', inplace=True)
    logger.debug(f"Columns of pred_df: {pred_df.columns}")

    # Calculate predictions with doubt column
    doubt_proba1_st_2_x_proba2 = conf.postprocess.getboolean('doubt_proba1_st_2_x_proba2')
    doubt_pred_ne_input_proba1_st_thresshold = conf.postprocess.getfloat('doubt_pred_ne_input_proba1_st_thresshold')
    doubt_pred_eq_input_proba1_st_thresshold = conf.postprocess.getfloat('doubt_pred_eq_input_proba1_st_thresshold')

    # Init with the standard prediction 
    pred_df[conf.columns['prediction_withdoubt']] = pred_df[conf.columns['prediction']]

    # Apply doubt for parcels with prediction != unverified input
    if doubt_proba1_st_2_x_proba2 is True:
        pred_df.loc[(pred_df[conf.columns['prediction']] != 'NODATA')
                        & (~pred_df[conf.columns['class']].isin(classes_to_ignore))
                        & (pred_df['pred1_prob'].map(float) < 2.0 * pred_df['pred2_prob'].map(float)),
                    conf.columns['prediction_withdoubt']] = 'DOUBT:PROBA1<2*PROBA2'
    
    # Apply doubt for parcels with prediction != unverified input
    if doubt_pred_ne_input_proba1_st_thresshold > 0:
        pred_df.loc[(pred_df[conf.columns['prediction']] != 'NODATA')
                        & (~pred_df[conf.columns['class']].isin(classes_to_ignore))
                        & (pred_df[conf.columns['prediction']] != pred_df[conf.columns['class']])
                        & (pred_df['pred1_prob'].map(float) < doubt_pred_ne_input_proba1_st_thresshold),
                    conf.columns['prediction_withdoubt']] = 'DOUBT:PRED<>INPUT-PROBA1<X'

    # Apply doubt for parcels with prediction == unverified input
    if doubt_pred_eq_input_proba1_st_thresshold > 0:
        pred_df.loc[(pred_df[conf.columns['prediction']] != 'NODATA')
                        & (~pred_df[conf.columns['class']].isin(classes_to_ignore))
                        & (pred_df[conf.columns['prediction']] == pred_df[conf.columns['class']])
                        & (pred_df['pred1_prob'].map(float) < doubt_pred_eq_input_proba1_st_thresshold),
                    conf.columns['prediction_withdoubt']] = 'DOUBT:PRED=INPUT-PROBA1<X'

    # Apply some extra, marker-specific doubt algorythms
    if conf.marker['markertype'] in ('LANDCOVER', 'LANDCOVER_EARLY'):
        logger.info("Apply some marker-specific doubt algorythms")
        # If parcel was declared as grassland, and is classified as arable, set to doubt
        # Remark: those gave only false positives for LANDCOVER marker
        pred_df.loc[(pred_df[conf.columns['class']] == 'MON_LC_GRASSES')
                        & (pred_df[conf.columns['prediction_withdoubt']] == 'MON_LC_ARABLE'),
                    conf.columns['prediction_withdoubt']] = 'DOUBT:GRASS->ARABLE'

        # If parcel was declared as fallow, and is classified as something else, set to doubt
        # Remark: those gave 50% false positives for marker LANDCOVER
        pred_df.loc[(~pred_df[conf.columns['prediction_withdoubt']].str.startswith('DOUBT'))
                        & (pred_df[conf.columns['class']] == 'MON_LC_FALLOW')
                        & (pred_df[conf.columns['prediction_withdoubt']] != 'MON_LC_FALLOW'),
                    conf.columns['prediction_withdoubt']] = 'DOUBT:FALLOW-UNCONFIRMED'
        
        if conf.marker['markertype'] == 'LANDCOVER_EARLY':
            # If parcel was declared as winter grain, but is not classified as MON_LC_ARABLE: doubt
            # Remark: those gave > 50% false positives for marker LANDCOVER_EARLY
            pred_df.loc[(~pred_df[conf.columns['prediction_withdoubt']].str.startswith('DOUBT'))
                            & (pred_df[conf.columns['crop_declared']].isin(['311', '321', '331']))
                            & (pred_df[conf.columns['prediction_withdoubt']] != 'MON_LC_ARABLE'),
                        conf.columns['prediction_withdoubt']] = 'DOUBT:GRAIN-UNCONFIRMED'

    elif conf.marker['markertype'] in ('CROPGROUP', 'CROPGROUP_EARLY'):
        logger.info("Apply some marker-specific doubt algorythms")

    # Add a column with the prediction status... and all parcels in df_top3 got a prediction
    pred_df[conf.columns['prediction_status']] = 'OK'
    pred_df.loc[(pred_df[conf.columns['prediction_withdoubt']].str.startswith('DOUBT')),
                conf.columns['prediction_status']] = 'DOUBT'
    pred_df.loc[(pred_df[conf.columns['prediction']] == 'NODATA'),
                conf.columns['prediction_status']] = 'NODATA'                
                
    # Calculate consolidated prediction: accuracy with few pixels is lower
    pred_df[conf.columns['prediction_cons']] = pred_df[conf.columns['prediction_withdoubt']]
    pred_df.loc[(pred_df[conf.columns['pixcount_s1s2']] <= conf.marker.getint('min_nb_pixels'))
                    & (pred_df[conf.columns['prediction_status']] != 'NODATA')
                    & (pred_df[conf.columns['prediction_status']] != 'DOUBT'),
                conf.columns['prediction_cons']] = 'DOUBT:NOT_ENOUGH_PIXELS'
    pred_df.loc[(pred_df[conf.columns['pixcount_s1s2']] <= conf.marker.getint('min_nb_pixels'))
                    & (pred_df[conf.columns['prediction_status']] != 'NODATA')
                    & (pred_df[conf.columns['prediction_status']] != 'DOUBT'),
                conf.columns['prediction_status']] = 'DOUBT'

    # Set the prediction status for classes that should be ignored
    pred_df.loc[pred_df[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore_for_train')), 
                [conf.columns['prediction_status']]] = 'UNKNOWN'
    pred_df.loc[pred_df[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore')), 
                [conf.columns['prediction_status']]] = pred_df[conf.columns['class']]

    # Calculate the status of the consolidated prediction (OK=usable, NOK: not)
    pred_df.loc[pred_df[conf.columns['prediction_cons']].isin(classified_classes), 
                conf.columns['prediction_cons_status']] = 'OK'
    pred_df[conf.columns['prediction_cons_status']].fillna('NOK', inplace=True)    
