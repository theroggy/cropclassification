# -*- coding: utf-8 -*-
"""
Main script to do a classification.

@author: Pieter Roggemans

"""

import logging
import os
import datetime
import cropclassification.preprocess.timeseries_calc_preprocess as ts_pre
import cropclassification.preprocess.timeseries_calc_gee as ts_calc_gee
import cropclassification.preprocess.timeseries as ts
import cropclassification.preprocess.classification_preprocess as class_pre
import cropclassification.predict.classification as classification
import cropclassification.postprocess.classification_reporting as class_report
import cropclassification.helpers.dir_helper as dir_helper
import cropclassification.helpers.log_helper as log_helper
import cropclassification.helpers.config_helper as conf 

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

def run(config_filepaths: [],
        reuse_last_run_dir: bool = None):
    # Read the configuration
    conf.read_config(config_filepaths)

    # Main initialisation of the logging
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)      
    logger.info(f"Config used: \n{conf.pformat_config()}")

    year = conf.marker.getint('year')
    base_dir = conf.dirs['base_dir']
    input_dir = conf.dirs['input_dir']
    input_preprocessed_dir = conf.dirs['input_preprocessed_dir']

    # Input file depends on the year
    if year:
        input_parcel_filename_noext = conf.config[f'marker:{year}']['input_parcel_filename_noext']
        input_groundtruth_csv = os.path.join(input_dir, conf.config[f'marker:{year}']['input_groundtruth_csv'])
    else:
        input_groundtruth_csv = None

    input_parcel_filepath = os.path.join(input_dir, f"{input_parcel_filename_noext}.shp")   # Input filepath of the parcel
    input_parcel_filetype = conf.marker['country_code']
    imagedata_dir = conf.dirs['imagedata_dir']
    start_date_str = conf.marker['start_date_str']
    end_date_str = conf.marker['end_date_str']
    buffer = conf.marker.getint('buffer')

    # REMARK: the column names that are used/expected can be found/changed in the config files!
    # Settings for monitoring landcover
    classtype_to_prepare = conf.marker['markertype']
    class_base_dir = conf.dirs['marker_base_dir']
    balancing_strategy = conf.marker['balancing_strategy']
    postprocess_to_groups = conf.marker['postprocess_to_groups']

    # Create run dir to be used for the results
    # If reuse_last_run_dir is not specified, look for it in the config
    if reuse_last_run_dir is None:
        reuse_last_run_dir = conf.dirs.getboolean('reuse_last_run_dir')
    marker_dir = dir_helper.create_run_dir(class_base_dir, reuse_last_run_dir)
    logger.info(f"Run dir with reuse_last_run_dir: {reuse_last_run_dir}, {marker_dir}")

    base_filename = f"{conf.marker['country_code']}{year}_bufm{buffer}_weekly"
    sensordata_to_use = [ts.SENSORDATA_S1_ASCDESC, ts.SENSORDATA_S2gt95]
    parceldata_aggregations_to_use = [ts.PARCELDATA_AGGRAGATION_MEAN]

    # Create the dir's if they don't exist yet...
    for dir in [base_dir, imagedata_dir, class_base_dir, marker_dir]:
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

    # Check if the necessary input files exist...
    if not os.path.exists(input_parcel_filepath):
        message = f"The parcel input file doesn't exist, so STOP: {input_parcel_filepath}"
        print(message)
        raise Exception(message)

    #-------------------------------------------------------------
    # The real work
    #-------------------------------------------------------------
    # STEP 1: prepare parcel data for classification and image data extraction
    #-------------------------------------------------------------

    # Prepare the input data for optimal image data extraction:
    #    TODO: 1) reproject to projection used in GEE: EPSG:4326
    #    2) apply a negative buffer on the parcel to evade mixels
    #    3) remove features that became null because of buffer
    imagedata_input_parcel_filename_noext = f"{input_parcel_filename_noext}_bufm{buffer}"
    imagedata_input_parcel_filepath = os.path.join(input_preprocessed_dir, f"{imagedata_input_parcel_filename_noext}.shp")
    ts_pre.prepare_input(input_parcel_filepath=input_parcel_filepath,
                         output_imagedata_parcel_input_filepath=imagedata_input_parcel_filepath) 

    # STEP 2: Get the timeseries data needed for the classification
    #-------------------------------------------------------------
    # Get the time series data (S1 and S2) to be used for the classification later on...
    # Result: data is put in csv files in imagedata_dir, in one csv file per date/period
    # Remarks:
    #    - the path to the imput data is specific for gee... so will need to be changed if another
    #      implementation is used
    #    - the upload to gee as an asset is not implemented, because it need a google cloud account...
    #      so upload needs to be done manually
    input_parcel_filepath_gee = f"{conf.dirs['gee']}{imagedata_input_parcel_filename_noext}"
    ts_calc_gee.calc_timeseries_data(input_parcel_filepath=input_parcel_filepath_gee,
                                     input_country_code=conf.marker['country_code'],
                                     start_date_str=start_date_str,
                                     end_date_str=end_date_str,
                                     sensordata_to_get=sensordata_to_use,
                                     base_filename=base_filename,
                                     dest_data_dir=imagedata_dir)

    # STEP 3: Preprocess all data needed for the classification
    #-------------------------------------------------------------
    # First prepare the basic input file with the classes that will be classified to.
    # Remarks:
    #    - this is typically specific for the input dataset and result wanted!!!
    #    - the result is/should be a csv file with the following columns
    #           - id (=global_settings.id_column): unique ID for each parcel
    #           - classname (=global_settings.class_column): the class that must be classified to.
    #             Remarks: - if the classname is listed in classes_to_ignore_for_train in the conf file, the parcel won't be used for training
    #                      - if the classname is listed in classes_to_ignore in the conf file, the parcel will be ignored
    #           - pixcount (=global_settings.pixcount_s1s2_column): the number of S1/S2 pixels in the
    #             parcel. Is -1 if the parcel doesn't have any S1/S2 data.
    #           - extra columns defined in the ini file under ${csv:extra_export_columns}
    parcel_csv = os.path.join(marker_dir, f"{input_parcel_filename_noext}_parcel.csv")
    parcel_pixcount_csv = os.path.join(imagedata_dir, f"{base_filename}_pixcount.csv")
    class_pre.prepare_input(input_parcel_filepath=input_parcel_filepath,
                            input_filetype=input_parcel_filetype,
                            input_parcel_pixcount_csv=parcel_pixcount_csv,
                            output_parcel_filepath=parcel_csv,
                            input_classtype_to_prepare=classtype_to_prepare)

    # Combine all data needed to do the classification in one input file
    parcel_classification_data_csv = os.path.join(marker_dir, f"{base_filename}_parcel_classdata.csv")
    ts.collect_and_prepare_timeseries_data(imagedata_dir=imagedata_dir,
                                           base_filename=base_filename,
                                           output_csv=parcel_classification_data_csv,
                                           start_date_str=start_date_str,
                                           end_date_str=end_date_str,
                                           min_fraction_data_in_column=0.9,
                                           sensordata_to_use=sensordata_to_use,
                                           parceldata_aggregations_to_use=parceldata_aggregations_to_use)

    # STEP 4: Train, test and classify
    #-------------------------------------------------------------
    # Create the training sample...
    # Remark: this creates a list of representative test parcel + a list of (candidate) training parcel
    parcel_train_csv = os.path.join(marker_dir, f"{base_filename}_parcel_train.csv")
    parcel_test_csv = os.path.join(marker_dir, f"{base_filename}_parcel_test.csv")
    class_pre.create_train_test_sample(input_parcel_csv=parcel_csv,
                                       output_parcel_train_csv=parcel_train_csv,
                                       output_parcel_test_csv=parcel_test_csv,
                                       balancing_strategy=balancing_strategy)

    # Train the classifier and output test predictions
    classifier_filepath = os.path.splitext(parcel_train_csv)[0] + "_classifier.pkl"
    parcel_predictions_test_csv = os.path.join(marker_dir, f"{base_filename}_predict_test.csv")
    parcel_predictions_all_csv = os.path.join(marker_dir, f"{base_filename}_predict_all.csv")
    classification.train_test_predict(input_parcel_train_csv=parcel_train_csv,
                                      input_parcel_test_csv=parcel_test_csv,
                                      input_parcel_all_csv=parcel_csv,
                                      input_parcel_classification_data_csv=parcel_classification_data_csv,
                                      output_classifier_filepath=classifier_filepath,
                                      output_predictions_test_csv=parcel_predictions_test_csv,
                                      output_predictions_all_csv=parcel_predictions_all_csv)
    
    # STEP 5: in necessary, postprocess results
    #-------------------------------------------------------------    
    '''if postprocess_to_groups is not None:
        # TODO 
    '''
    
    # STEP 6: Report on the test accuracy, incl. ground truth
    #-------------------------------------------------------------
    # Preprocess the ground truth data
    groundtruth_csv = None
    if input_groundtruth_csv is not None:
        _, input_gt_filename = os.path.split(input_groundtruth_csv)
        input_gt_filename_noext, input_gt_filename_ext = os.path.splitext(input_gt_filename)
        groundtruth_csv = os.path.join(marker_dir, f"{input_gt_filename_noext}_classes{input_gt_filename_ext}")
        class_pre.prepare_input(input_parcel_filepath=input_groundtruth_csv,
                                input_filetype=input_parcel_filetype,
                                input_parcel_pixcount_csv=parcel_pixcount_csv,
                                output_parcel_filepath=groundtruth_csv,
                                input_classtype_to_prepare=f"{classtype_to_prepare}_GROUNDTRUTH")

    # Print full reporting on the accuracy
    report_txt = f"{parcel_predictions_test_csv}_accuracy_report.txt"
    class_report.write_full_report(parcel_predictions_csv=parcel_predictions_test_csv,
                                   output_report_txt=report_txt,
                                   parcel_ground_truth_csv=groundtruth_csv)

    # STEP 7: Report on the full accuracy, incl. ground truth
    #-------------------------------------------------------------
    # Print full reporting on the accuracy
    report_txt = f"{parcel_predictions_all_csv}_accuracy_report.txt"
    class_report.write_full_report(parcel_predictions_csv=parcel_predictions_all_csv,
                                   output_report_txt=report_txt,
                                   parcel_ground_truth_csv=groundtruth_csv)

    logging.shutdown()