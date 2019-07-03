# -*- coding: utf-8 -*-
"""
Main script to do a classification.
"""

import datetime
import logging
import os

import cropclassification.helpers.config_helper as conf 
import cropclassification.helpers.dir_helper as dir_helper
import cropclassification.helpers.log_helper as log_helper
import cropclassification.preprocess.timeseries_util as ts_util
import cropclassification.preprocess.timeseries as ts
import cropclassification.preprocess.classification_preprocess as class_pre
import cropclassification.predict.classification as classification
import cropclassification.postprocess.classification_postprocess as class_post
import cropclassification.postprocess.classification_reporting as class_report

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
def run(markertype_to_calc: str,
        input_parcel_filename: str,
        input_parcel_filetype: str,
        country_code: str,
        year: int,
        input_groundtruth_filename: str,
        input_model_to_use_filepath: str):
    """
    Runs a marker for an input file. If no input model to use is specified,
    a new one will be trained.

    Args

    """

    # If a model to use is specified, check if it exists...
    if input_model_to_use_filepath is not None and not os.path.exists(input_model_to_use_filepath):
        raise Exception(f"Input file input_model_to_use_filepath doesn't exist: {input_model_to_use_filepath}")
    
    # Determine the config files to load depending on the marker_type
    marker_ini = f"config/{markertype_to_calc.lower()}.ini"
    config_filepaths = ["config/general.ini",
                        marker_ini,
                        "config/local_overrule.ini"]

    # Read the configuration files
    conf.read_config(config_filepaths, year=year)

    # Create run dir to be used for the results
    reuse_last_run_dir = conf.dirs.getboolean('reuse_last_run_dir')
    reuse_last_run_dir_config = conf.dirs.getboolean('reuse_last_run_dir_config')
    run_dir = dir_helper.create_run_dir(conf.dirs['marker_base_dir'], reuse_last_run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Main initialisation of the logging
    logger = log_helper.main_log_init(run_dir, __name__)      
    logger.info(f"Run dir with reuse_last_run_dir: {reuse_last_run_dir}, {run_dir}")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # If the config needs to be reused as well, load it, else write it
    config_used_filepath = os.path.join(run_dir, 'config_used.ini')
    if(reuse_last_run_dir 
       and reuse_last_run_dir_config
       and os.path.exists(run_dir)
       and os.path.exists(config_used_filepath)):
        config_filepaths.append(config_used_filepath)
        logger.info(f"Run dir config needs to be reused, so {config_filepaths}")
        conf.read_config(config_filepaths=config_filepaths, year=year)
        logger.info("Write new config_used.ini, because some parameters might have been added")
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)
    else:
        logger.info("Write config_used.ini, so it can be reused later on")
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)

    # Prepare input filepaths
    input_dir = conf.dirs['input_dir']    
    input_parcel_filepath = os.path.join(input_dir, input_parcel_filename)
    if input_groundtruth_filename is not None:
        input_groundtruth_filepath = os.path.join(input_dir, input_groundtruth_filename)
    else:
        input_groundtruth_filepath = None

    # Check if the necessary input files exist...
    if not os.path.exists(input_parcel_filepath):
        message = f"The parcel input file doesn't exist, so STOP: {input_parcel_filepath}"
        logger.critical(message)
        raise Exception(message)

    # Get some general config
    columndata_ext = conf.general['columndata_ext']
    rowdata_ext = conf.general['rowdata_ext']
    output_ext = conf.general['output_ext']
    geofile_ext = conf.general['geofile_ext']
       
    #-------------------------------------------------------------
    # The real work
    #-------------------------------------------------------------
    # STEP 1: prepare parcel data for classification and image data extraction
    #-------------------------------------------------------------

    # Prepare the input data for optimal image data extraction:
    #    1) apply a negative buffer on the parcel to evade mixels
    #    2) remove features that became null because of buffer
    input_preprocessed_dir = conf.dirs['input_preprocessed_dir']
    input_parcel_filename_noext, _ = os.path.splitext(input_parcel_filename)
    buffer = conf.marker.getint('buffer')       
    input_parcel_nogeo_filepath = os.path.join(
            input_preprocessed_dir, f"{input_parcel_filename_noext}{columndata_ext}")
    imagedata_input_parcel_filename_noext = f"{input_parcel_filename_noext}_bufm{buffer}"
    imagedata_input_parcel_filepath = os.path.join(
            input_preprocessed_dir, f"{imagedata_input_parcel_filename_noext}{geofile_ext}")
    ts_util.prepare_input(
            input_parcel_filepath=input_parcel_filepath,
            output_imagedata_parcel_input_filepath=imagedata_input_parcel_filepath,
            output_parcel_nogeo_filepath=input_parcel_nogeo_filepath)

    # STEP 2: Get the timeseries data needed for the classification
    #-------------------------------------------------------------
    # Get the time series data (S1 and S2) to be used for the classification 
    # Result: data is put in files in timeseries_periodic_dir, in one file per 
    #         date/period
    timeseries_periodic_dir = conf.dirs['timeseries_periodic_dir']
    start_date_str = conf.marker['start_date_str']
    end_date_str = conf.marker['end_date_str']
    sensordata_to_use = conf.marker.getlist('sensordata_to_use')
    parceldata_aggregations_to_use = conf.marker.getlist('parceldata_aggregations_to_use')
    base_filename = f"{input_parcel_filename_noext}_bufm{buffer}_weekly"
    ts.calc_timeseries_data(
            input_parcel_filepath=imagedata_input_parcel_filepath,
            input_country_code=country_code,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            sensordata_to_get=sensordata_to_use,
            base_filename=base_filename,
            dest_data_dir=timeseries_periodic_dir)

    # STEP 3: Preprocess all data needed for the classification
    #-------------------------------------------------------------
    # Prepare the basic input file with the classes that will be classified to.
    # Remarks:
    #    - this is typically specific for the input dataset and result wanted!!!
    #    - the result is/should be a file with the following columns
    #           - id (=global_settings.id_column): unique ID for each parcel
    #           - classname (=global_settings.class_column): the class that must 
    #             be classified to.
    #             Remarks: - if in classes_to_ignore_for_train, class won't be used for training
    #                      - if in classes_to_ignore, the class will be ignored
    #           - pixcount (=global_settings.pixcount_s1s2_column):  
    #             the number of S1/S2 pixels in the parcel.
    #             Is -1 if the parcel doesn't have any S1/S2 data.
    classtype_to_prepare = conf.preprocess['classtype_to_prepare']
    parcel_filepath = os.path.join(
            run_dir, f"{input_parcel_filename_noext}_parcel{columndata_ext}")
    parcel_pixcount_filepath = os.path.join(
            timeseries_periodic_dir, f"{base_filename}_pixcount{columndata_ext}")
    class_pre.prepare_input(
            input_parcel_filepath=input_parcel_nogeo_filepath,
            input_parcel_filetype=input_parcel_filetype,
            input_parcel_pixcount_filepath=parcel_pixcount_filepath,
            classtype_to_prepare=classtype_to_prepare,
            output_parcel_filepath=parcel_filepath)

    # Collect all data needed to do the classification in one input file
    parcel_classification_data_filepath = os.path.join(
            run_dir, f"{base_filename}_parcel_classdata{rowdata_ext}")
    ts.collect_and_prepare_timeseries_data(
            input_parcel_filepath=input_parcel_nogeo_filepath,
            timeseries_dir=timeseries_periodic_dir,
            base_filename=base_filename,
            output_filepath=parcel_classification_data_filepath,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            sensordata_to_use=sensordata_to_use,
            parceldata_aggregations_to_use=parceldata_aggregations_to_use)

    # STEP 4: Train and test if necessary... and predict
    #-------------------------------------------------------------
    parcel_predictions_proba_all_filepath = os.path.join(
            run_dir, f"{base_filename}_predict_proba_all{rowdata_ext}")

    # if there is no model to use specified, train one!
    if input_model_to_use_filepath is None:

        # Create the training sample...
        # Remark: this creates a list of representative test parcel + a list of (candidate) training parcel
        balancing_strategy = conf.marker['balancing_strategy']
        parcel_train_filepath = os.path.join(run_dir, 
                f"{base_filename}_parcel_train{columndata_ext}")
        parcel_test_filepath = os.path.join(
                run_dir, f"{base_filename}_parcel_test{columndata_ext}")
        class_pre.create_train_test_sample(
                input_parcel_filepath=parcel_filepath,
                output_parcel_train_filepath=parcel_train_filepath,
                output_parcel_test_filepath=parcel_test_filepath,
                balancing_strategy=balancing_strategy)

        # Train the classifier and output predictions
        classifier_ext = conf.classifier['classifier_ext']
        classifier_filepath = os.path.splitext(parcel_train_filepath)[0] + f"_classifier{classifier_ext}"
        parcel_predictions_proba_test_filepath = os.path.join(
                run_dir, f"{base_filename}_predict_proba_test{rowdata_ext}")
        classification.train_test_predict(
                input_parcel_train_filepath=parcel_train_filepath,
                input_parcel_test_filepath=parcel_test_filepath,
                input_parcel_all_filepath=parcel_filepath,
                input_parcel_classification_data_filepath=parcel_classification_data_filepath,
                output_classifier_filepath=classifier_filepath,
                output_predictions_test_filepath=parcel_predictions_proba_test_filepath,
                output_predictions_all_filepath=parcel_predictions_proba_all_filepath)
    else:
        # there is a classifier specified, so just use it!
        classification.predict(
                input_parcel_filepath=parcel_filepath,
                input_parcel_classification_data_filepath=parcel_classification_data_filepath,
                input_classifier_filepath=input_model_to_use_filepath,
                output_predictions_filepath=parcel_predictions_proba_all_filepath)

    # STEP 5: if necessary, do extra postprocessing
    #-------------------------------------------------------------    
    '''if postprocess_to_groups is not None:
        # TODO 
    '''

    # STEP 6: do the default, mandatory postprocessing
    #-------------------------------------------------------------
    # If it was necessary to train, there will be a test prediction... so postprocess it
    if input_model_to_use_filepath is None:
        parcel_predictions_test_filepath = os.path.join(
                run_dir, f"{base_filename}_predict_test{output_ext}")
        class_post.calc_top3_and_consolidation(
                input_parcel_filepath=parcel_test_filepath,
                input_parcel_probabilities_filepath=parcel_predictions_proba_test_filepath,
                output_predictions_filepath=parcel_predictions_test_filepath)
        
    # Postprocess predictions
    parcel_predictions_all_filepath = os.path.join(
            run_dir, f"{base_filename}_predict_all{output_ext}")
    parcel_predictions_all_output_filepath = os.path.join(
            run_dir, f"{base_filename}_predict_all_output{output_ext}")
    class_post.calc_top3_and_consolidation(
            input_parcel_filepath=parcel_filepath,
            input_parcel_probabilities_filepath=parcel_predictions_proba_all_filepath,
            output_predictions_filepath=parcel_predictions_all_filepath,
            output_predictions_output_filepath=parcel_predictions_all_output_filepath)

    # STEP 7: Report on the accuracy, incl. ground truth
    #-------------------------------------------------------------
    # Preprocess the ground truth data if it is provided
    groundtruth_filepath = None
    if input_groundtruth_filepath is not None:
            _, input_gt_filename = os.path.split(input_groundtruth_filepath)
            input_gt_filename_noext, input_gt_filename_ext = os.path.splitext(input_gt_filename)
            groundtruth_filepath = os.path.join(
                    run_dir, f"{input_gt_filename_noext}_classes{input_gt_filename_ext}")
            class_pre.prepare_input(
                    input_parcel_filepath=input_groundtruth_filepath,
                    input_parcel_filetype=input_parcel_filetype,
                    input_parcel_pixcount_filepath=parcel_pixcount_filepath,
                    classtype_to_prepare=conf.preprocess['classtype_to_prepare_groundtruth'],
                    output_parcel_filepath=groundtruth_filepath)

    # If we trained a model, there is a test prediction we want to report on
    if input_model_to_use_filepath is None:
        # Print full reporting on the accuracy of the test dataset
        report_txt = f"{parcel_predictions_test_filepath}_accuracy_report.txt"
        class_report.write_full_report(
                parcel_predictions_filepath=parcel_predictions_test_filepath,
                output_report_txt=report_txt,
                parcel_ground_truth_filepath=groundtruth_filepath)

    # Print full reporting on the accuracy of the full dataset
    report_txt = f"{parcel_predictions_all_filepath}_accuracy_report.txt"
    class_report.write_full_report(
            parcel_predictions_filepath=parcel_predictions_all_filepath,
            output_report_txt=report_txt,
            parcel_ground_truth_filepath=groundtruth_filepath)

    logging.shutdown()
    