# -*- coding: utf-8 -*-
"""
Main script to do a classification.
"""

import logging
import os
from pathlib import Path
import shutil
from typing import List

from cropclassification.helpers import config_helper as conf 
from cropclassification.helpers import dir_helper
from cropclassification.helpers import log_helper
from cropclassification.helpers import model_helper as mh
from cropclassification.preprocess import timeseries_util as ts_util
from cropclassification.preprocess import timeseries as ts
from cropclassification.preprocess import classification_preprocess as class_pre
from cropclassification.predict import classification
from cropclassification.postprocess import classification_postprocess as class_post
from cropclassification.postprocess import classification_reporting as class_report

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
def calc_marker_task(
        config_filepaths: List[Path],
        default_basedir: Path):
    """
    Runs a marker using the setting in the config_filepaths.
    
    Args:
        config_filepaths (List[Path]): the config files to load
        default_basedir (Path): the dir to resolve relative paths in the config
            file to.
    
    Raises:
        Exception: [description]
        Exception: [description]
    """
    # Read the configuration files
    conf.read_config(config_filepaths, default_basedir=default_basedir)
    
    # Create run dir to be used for the results
    reuse_last_run_dir = conf.calc_marker_params.getboolean('reuse_last_run_dir')
    reuse_last_run_dir_config = conf.calc_marker_params.getboolean('reuse_last_run_dir_config')
    run_dir = dir_helper.create_run_dir(conf.dirs.getpath('marker_dir'), reuse_last_run_dir)
    print(run_dir)
    if not run_dir.exists():
        os.makedirs(run_dir)

    # Main initialisation of the logging
    log_level = conf.general.get("log_level")
    logger = log_helper.main_log_init(run_dir, __name__, log_level)      
    logger.info(f"Run dir with reuse_last_run_dir: {reuse_last_run_dir}, {run_dir}")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # If running in conda, export the environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env is not None:
        environment_yml_path = run_dir / f"{conda_env}.yml"
        logger.info(f"Export conda environment used to {environment_yml_path}")
        os.system(f"conda env export > {environment_yml_path}")

    # If the config needs to be reused as well, load it, else write it
    config_used_filepath = run_dir / 'config_used.ini'
    if(reuse_last_run_dir 
       and reuse_last_run_dir_config
       and run_dir.exists()
       and config_used_filepath.exists()):
        config_filepaths.append(config_used_filepath)
        logger.info(f"Run dir config needs to be reused, so {config_filepaths}")
        conf.read_config(config_filepaths=config_filepaths, default_basedir=default_basedir)
        logger.info("Write new config_used.ini, because some parameters might have been added")
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)
    else:
        # Copy the config files to a config dir for later notice
        configfiles_used_dir = run_dir / "configfiles_used"
        if configfiles_used_dir.exists():
            shutil.rmtree(configfiles_used_dir)
        configfiles_used_dir.mkdir()
        for config_filepath in config_filepaths:
            shutil.copy(config_filepath, configfiles_used_dir)

        # Write the resolved complete config, so it can be reused 
        logger.info("Write config_used.ini, so it can be reused later on")
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)

    # Read the info about the run
    input_parcel_filename = conf.calc_marker_params.getpath('input_parcel_filename')
    input_parcel_filetype = conf.calc_marker_params['input_parcel_filetype']
    country_code = conf.calc_marker_params['country_code']
    classes_refe_filename = conf.calc_marker_params.getpath('classes_refe_filename')
    input_groundtruth_filename = conf.calc_marker_params.getpath('input_groundtruth_filename')
    input_model_to_use_relativepath = conf.calc_marker_params.getpath('input_model_to_use_relativepath')

    # Prepare input filepaths
    if input_model_to_use_relativepath is not None:
        input_model_to_use_filepath = conf.dirs.getpath('model_dir') / input_model_to_use_relativepath
        if not input_model_to_use_filepath.exists():
            raise Exception(f"Input file input_model_to_use_filepath doesn't exist: {input_model_to_use_filepath}")
    else:
        input_model_to_use_filepath = None

    input_dir = conf.dirs.getpath('input_dir')
    input_parcel_filepath = input_dir / input_parcel_filename
    if input_groundtruth_filename is not None:
        input_groundtruth_filepath = input_dir / input_groundtruth_filename
    else:
        input_groundtruth_filepath = None
    
    refe_dir = conf.dirs.getpath('refe_dir')
    classes_refe_filepath = refe_dir / classes_refe_filename

    # Check if the necessary input files exist...
    for path in [classes_refe_filepath, input_parcel_filepath]:
        if path is not None and not path.exists():
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
    input_preprocessed_dir = conf.dirs.getpath('input_preprocessed_dir')
    buffer = conf.marker.getint('buffer')       
    input_parcel_nogeo_filepath = input_preprocessed_dir / f"{input_parcel_filename.stem}{data_ext}"
    imagedata_input_parcel_filename = f"{input_parcel_filename.stem}_bufm{buffer}{geofile_ext}"
    imagedata_input_parcel_filepath = input_preprocessed_dir / imagedata_input_parcel_filename
    ts_util.prepare_input(
            input_parcel_filepath=input_parcel_filepath,
            output_imagedata_parcel_input_filepath=imagedata_input_parcel_filepath,
            output_parcel_nogeo_filepath=input_parcel_nogeo_filepath)

    # STEP 2: Get the timeseries data needed for the classification
    #-------------------------------------------------------------
    # Get the time series data (S1 and S2) to be used for the classification 
    # Result: data is put in files in timeseries_periodic_dir, in one file per 
    #         date/period
    timeseries_periodic_dir = conf.dirs.getpath('timeseries_periodic_dir')
    start_date_str = conf.marker['start_date_str']
    end_date_str = conf.marker['end_date_str']
    sensordata_to_use = conf.marker.getlist('sensordata_to_use')
    parceldata_aggregations_to_use = conf.marker.getlist('parceldata_aggregations_to_use')
    base_filename = f"{input_parcel_filename.stem}_bufm{buffer}_weekly"
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
    parcel_filepath = run_dir / f"{input_parcel_filename.stem}_parcel{data_ext}"
    parcel_pixcount_filepath = timeseries_periodic_dir / f"{base_filename}_pixcount{data_ext}"
    class_pre.prepare_input(
            input_parcel_filepath=input_parcel_nogeo_filepath,
            input_parcel_filetype=input_parcel_filetype,
            input_parcel_pixcount_filepath=parcel_pixcount_filepath,
            classtype_to_prepare=classtype_to_prepare,
            classes_refe_filepath=classes_refe_filepath,
            output_parcel_filepath=parcel_filepath)

    # Collect all data needed to do the classification in one input file
    parcel_classification_data_filepath = run_dir / f"{base_filename}_parcel_classdata{data_ext}"
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
    parcel_predictions_proba_all_filepath = run_dir / f"{base_filename}_predict_proba_all{data_ext}"
    classifier_ext = conf.classifier['classifier_ext']
    classifier_basefilepath = run_dir / f"{markertype}_01_mlp{classifier_ext}"    

    # Check if a model exists already
    if input_model_to_use_filepath is None:
        best_model = mh.get_best_model(run_dir, acc_metric_mode='min')
        if best_model is not None:
            input_model_to_use_filepath = best_model['filepath']
            
    # if there is no model to use specified, train one!
    parcel_test_filepath = None
    parcel_predictions_proba_test_filepath = None
    if input_model_to_use_filepath is None:

        # Create the training sample...
        # Remark: this creates a list of representative test parcel + a list of (candidate) training parcel
        balancing_strategy = conf.marker['balancing_strategy']
        parcel_train_filepath = run_dir / f"{base_filename}_parcel_train{data_ext}"
        parcel_test_filepath = run_dir / f"{base_filename}_parcel_test{data_ext}"
        class_pre.create_train_test_sample(
                input_parcel_filepath=parcel_filepath,
                output_parcel_train_filepath=parcel_train_filepath,
                output_parcel_test_filepath=parcel_test_filepath,
                balancing_strategy=balancing_strategy)

        # Train the classifier and output predictions
        parcel_predictions_proba_test_filepath = run_dir / f"{base_filename}_predict_proba_test{data_ext}"
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
    parcel_predictions_test_filepath = None
    parcel_predictions_test_geopath = None
    if(input_model_to_use_filepath is None 
       and parcel_test_filepath is not None 
       and parcel_predictions_proba_test_filepath is not None):
        parcel_predictions_test_filepath = run_dir / f"{base_filename}_predict_test{data_ext}"
        parcel_predictions_test_geopath = run_dir / f"{base_filename}_predict_test{geofile_ext}"
        class_post.calc_top3_and_consolidation(
                input_parcel_filepath=parcel_test_filepath,
                input_parcel_probabilities_filepath=parcel_predictions_proba_test_filepath,
                input_parcel_geofilepath=input_parcel_filepath,
                output_predictions_filepath=parcel_predictions_test_filepath,
                output_predictions_geofilepath=parcel_predictions_test_geopath)
        
    # Postprocess predictions
    parcel_predictions_all_filepath = run_dir / f"{base_filename}_predict_all{data_ext}"
    parcel_predictions_all_geopath = run_dir / f"{base_filename}_predict_all{geofile_ext}"
    parcel_predictions_all_output_filepath = run_dir / f"{base_filename}_predict_all_output{output_ext}"
    class_post.calc_top3_and_consolidation(
            input_parcel_filepath=parcel_filepath,
            input_parcel_probabilities_filepath=parcel_predictions_proba_all_filepath,
            input_parcel_geofilepath=input_parcel_filepath,
            output_predictions_filepath=parcel_predictions_all_filepath,
            output_predictions_geofilepath=parcel_predictions_all_geopath,
            output_predictions_output_filepath=parcel_predictions_all_output_filepath)

    # STEP 7: Report on the accuracy, incl. ground truth
    #-------------------------------------------------------------
    # Preprocess the ground truth data if it is provided
    groundtruth_filepath = None
    if input_groundtruth_filepath is not None:
            groundtruth_filepath = run_dir / f"{input_groundtruth_filepath.stem}_classes{input_groundtruth_filepath.suffix}"
            class_pre.prepare_input(
                    input_parcel_filepath=input_groundtruth_filepath,
                    input_parcel_filetype=input_parcel_filetype,
                    input_parcel_pixcount_filepath=parcel_pixcount_filepath,
                    classtype_to_prepare=conf.preprocess['classtype_to_prepare_groundtruth'],
                    classes_refe_filepath=classes_refe_filepath,
                    output_parcel_filepath=groundtruth_filepath)

    # If we trained a model, there is a test prediction we want to report on
    if input_model_to_use_filepath is None and parcel_predictions_test_geopath is not None:
        # Print full reporting on the accuracy of the test dataset
        report_txt = Path(f"{str(parcel_predictions_test_filepath)}_accuracy_report.txt")
        class_report.write_full_report(
                parcel_predictions_geopath=parcel_predictions_test_geopath,
                output_report_txt=report_txt,
                parcel_ground_truth_filepath=groundtruth_filepath)

    # Print full reporting on the accuracy of the full dataset
    report_txt = Path(f"{str(parcel_predictions_all_filepath)}_accuracy_report.txt")
    class_report.write_full_report(
            parcel_predictions_geopath=parcel_predictions_all_geopath,
            output_report_txt=report_txt,
            parcel_ground_truth_filepath=groundtruth_filepath)

    logging.shutdown()
    