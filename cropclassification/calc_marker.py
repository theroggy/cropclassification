# -*- coding: utf-8 -*-
"""
Main script to do a classification.
"""

import configparser
import logging
import os
from pathlib import Path
import shutil

import cropclassification.helpers.config_helper as conf 
import cropclassification.helpers.dir_helper as dir_helper
import cropclassification.helpers.log_helper as log_helper
import cropclassification.helpers.model_helper as mh
import cropclassification.preprocess.timeseries_util as ts_util
import cropclassification.preprocess.timeseries as ts
import cropclassification.preprocess.classification_preprocess as class_pre
import cropclassification.predict.classification as classification
import cropclassification.postprocess.classification_postprocess as class_post
import cropclassification.postprocess.classification_reporting as class_report

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
def calc_marker_job(job_path: Path):
    """
    Runs a marker for an job file. If no input model to use is specified,
    a new one will be trained.
    
    Args:
        job_path (Path): the job file where the input parameters to calculate 
            the marker can be found.
    
    Raises:
        Exception: [description]
        Exception: [description]
    """
    # Create configparser and read job file!
    job_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')]},
            allow_no_value=True)
    job_config.read(job_path)

    # Determine the config files to load 
    script_dir = Path(os.path.abspath(__file__)).parent
    config_filepaths = [script_dir / 'config' / 'general.ini']
    extra_config_files_to_load = job_config['job'].getlist('extra_config_files_to_load')
    if extra_config_files_to_load is not None:
        for config_file in extra_config_files_to_load:
            config_file_formatted = config_file.format(script_dir=script_dir.as_posix(), job_path=job_path)
            config_filepaths.append(Path(config_file_formatted))

    # Read the configuration files
    conf.read_config(config_filepaths, default_basedir=job_path.parent.parent)
    
    # Create run dir to be used for the results
    reuse_last_run_dir = conf.dirs.getboolean('reuse_last_run_dir')
    reuse_last_run_dir_config = conf.dirs.getboolean('reuse_last_run_dir_config')
    run_dir = dir_helper.create_run_dir(conf.dirs['marker_dir'], reuse_last_run_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Main initialisation of the logging
    logger = log_helper.main_log_init(run_dir, __name__)      
    logger.info(f"Run dir with reuse_last_run_dir: {reuse_last_run_dir}, {run_dir}")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # If the config needs to be reused as well, load it, else write it
    config_used_filepath = Path(run_dir) / 'config_used.ini'
    if(reuse_last_run_dir 
       and reuse_last_run_dir_config
       and os.path.exists(run_dir)
       and os.path.exists(config_used_filepath)):
        config_filepaths.append(config_used_filepath)
        logger.info(f"Run dir config needs to be reused, so {config_filepaths}")
        conf.read_config(config_filepaths=config_filepaths, default_basedir=job_path.parent.parent)
        logger.info("Write new config_used.ini, because some parameters might have been added")
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)
    else:
        logger.info("Write config_used.ini, so it can be reused later on")
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)

    # Also copy the job file used to the run dir
    shutil.copy(job_path, run_dir)

    # Read the info about the run
    runinfo = conf.config['runinfo']
    input_parcel_filename = runinfo['input_parcel_filename']
    input_parcel_filetype = runinfo['input_parcel_filetype']
    country_code = runinfo['country_code']
    classes_refe_filename = runinfo['classes_refe_filename']
    input_groundtruth_filename = runinfo['input_groundtruth_filename']
    input_model_to_use_relativepath = runinfo['input_model_to_use_relativepath']

    # Prepare input filepaths
    if input_model_to_use_relativepath is not None:
        input_model_to_use_filepath = os.path.join(conf.dirs.get('model_dir'), input_model_to_use_relativepath)
        if not os.path.exists(input_model_to_use_filepath):
            raise Exception(f"Input file input_model_to_use_filepath doesn't exist: {input_model_to_use_filepath}")
    else:
        input_model_to_use_filepath = None

    input_dir = conf.dirs['input_dir']
    input_parcel_filepath = os.path.join(input_dir, input_parcel_filename)
    if input_groundtruth_filename is not None:
        input_groundtruth_filepath = os.path.join(input_dir, input_groundtruth_filename)
    else:
        input_groundtruth_filepath = None
    
    refe_dir = conf.dirs.getpath('refe_dir')
    classes_refe_filepath = refe_dir / classes_refe_filename

    # Check if the necessary input files exist...
    for path in [classes_refe_filepath, input_parcel_filepath]:
        if path is not None and not os.path.exists(path):
            message = f"Input file doesn't exist, so STOP: {path}"
            logger.critical(message)
            raise Exception(message)

    # Get some general config
    data_ext = conf.general['data_ext']
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
            input_preprocessed_dir, f"{input_parcel_filename_noext}{data_ext}")
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
    #           - id (=id_column): unique ID for each parcel
    #           - classname (=class_column): the class that must 
    #             be classified to.
    #             Remarks: - if in classes_to_ignore_for_train, class won't be used for training
    #                      - if in classes_to_ignore, the class will be ignored
    #           - pixcount:  
    #             the number of S1/S2 pixels in the parcel.
    #             Is -1 if the parcel doesn't have any S1/S2 data.
    classtype_to_prepare = conf.preprocess['classtype_to_prepare']
    parcel_filepath = os.path.join(
            run_dir, f"{input_parcel_filename_noext}_parcel{data_ext}")
    parcel_pixcount_filepath = os.path.join(
            timeseries_periodic_dir, f"{base_filename}_pixcount{data_ext}")
    class_pre.prepare_input(
            input_parcel_filepath=input_parcel_nogeo_filepath,
            input_parcel_filetype=input_parcel_filetype,
            input_parcel_pixcount_filepath=parcel_pixcount_filepath,
            classtype_to_prepare=classtype_to_prepare,
            classes_refe_filepath=classes_refe_filepath,
            output_parcel_filepath=parcel_filepath)

    # Collect all data needed to do the classification in one input file
    parcel_classification_data_filepath = os.path.join(
            run_dir, f"{base_filename}_parcel_classdata{data_ext}")
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
    markertype = conf.marker.get('markertype')
    parcel_predictions_proba_all_filepath = os.path.join(
            run_dir, f"{base_filename}_predict_proba_all{data_ext}")
    classifier_ext = conf.classifier['classifier_ext']
    classifier_basefilepath = os.path.join(run_dir, f"{markertype}_01_mlp{classifier_ext}")    

    # Check if a model exists already
    if input_model_to_use_filepath is None:
        best_model = mh.get_best_model(run_dir, acc_metric_mode='min')
        if best_model is not None:
            input_model_to_use_filepath = best_model['filepath']
            
    # if there is no model to use specified, train one!
    if input_model_to_use_filepath is None:

        # Create the training sample...
        # Remark: this creates a list of representative test parcel + a list of (candidate) training parcel
        balancing_strategy = conf.marker['balancing_strategy']
        parcel_train_filepath = os.path.join(run_dir, 
                f"{base_filename}_parcel_train{data_ext}")
        parcel_test_filepath = os.path.join(
                run_dir, f"{base_filename}_parcel_test{data_ext}")
        class_pre.create_train_test_sample(
                input_parcel_filepath=parcel_filepath,
                output_parcel_train_filepath=parcel_train_filepath,
                output_parcel_test_filepath=parcel_test_filepath,
                balancing_strategy=balancing_strategy)

        # Train the classifier and output predictions
        parcel_predictions_proba_test_filepath = os.path.join(
                run_dir, f"{base_filename}_predict_proba_test{data_ext}")
        classification.train_test_predict(
                input_parcel_train_filepath=parcel_train_filepath,
                input_parcel_test_filepath=parcel_test_filepath,
                input_parcel_all_filepath=parcel_filepath,
                input_parcel_classification_data_filepath=parcel_classification_data_filepath,
                output_classifier_basefilepath=classifier_basefilepath,
                output_predictions_test_filepath=parcel_predictions_proba_test_filepath,
                output_predictions_all_filepath=parcel_predictions_proba_all_filepath)
    else:
        # there is a classifier specified, so just use it!
        classification.predict(
                input_parcel_filepath=parcel_filepath,
                input_parcel_classification_data_filepath=parcel_classification_data_filepath,
                input_classifier_basefilepath=classifier_basefilepath,
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
                run_dir, f"{base_filename}_predict_test{data_ext}")
        class_post.calc_top3_and_consolidation(
                input_parcel_filepath=parcel_test_filepath,
                input_parcel_probabilities_filepath=parcel_predictions_proba_test_filepath,
                output_predictions_filepath=parcel_predictions_test_filepath)
        
    # Postprocess predictions
    parcel_predictions_all_filepath = os.path.join(
            run_dir, f"{base_filename}_predict_all{data_ext}")
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
                    classes_refe_filepath=classes_refe_filepath,
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
    