# -*- coding: utf-8 -*-
"""
Main script to do a classification.

@author: Pieter Roggemans

"""

import logging
import os
import datetime

import io_helper
import timeseries_calc_preprocess as ts_pre
import timeseries_calc_gee as ts_calc_gee
import timeseries as ts
import classification_preprocess as class_pre
import classification
import classification_reporting as class_report

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
year = 2018
country_code = 'BEFL'        # The region of the classification: typically country code

#base_dir = 'X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-05-04_Monitoring_Classificatie'                                               # Base dir
base_dir = 'x:\\Monitoring\\Markers\\PlayGround\\JanAnt'                                               # Base dir
input_dir = os.path.join(base_dir, 'InputData')                                         # Input dir
input_preprocessed_dir = os.path.join(input_dir, 'Preprocessed')

# Input file depends on the year
if year == 2017:
    input_parcel_filename_noext = 'Prc_flanders_2017_2018-01-09'                        # Input filename
elif year == 2018:
    input_parcel_filename_noext = 'Prc_BEFL_2018_2018-08-02'                            # Input filename
input_parcel_filepath = os.path.join(input_dir, f"{input_parcel_filename_noext}.shp")   # Input filepath of the parcel
if year == 2017:
    input_groundtruth_csv = os.path.join(input_dir, "Prc_BEFL_2017_groundtruth.csv")    # The ground truth
if year == 2018:
    input_groundtruth_csv = os.path.join(input_dir, "Prc_BEFL_2018_groundtruth.csv")    # The ground truth
else:
    input_groundtruth_csv = None

input_groundtruth_csv = None

input_parcel_filetype = country_code
imagedata_dir = os.path.join(base_dir, 'Timeseries_data')      # General data download dir
start_date_str = f"{year}-03-27"
end_date_str = f"{year}-08-10"                                 # End date is NOT inclusive for gee processing

# REMARK: the column names that are used/expected can be found/changed in global_constants.py!
'''
# Settings for 7 main crops
classtype_to_prepare = 'MOST_POPULAR_CROPS'
class_base_dir = os.path.join(base_dir, f"{year}_class_maincrops7")    # Dir for the classification type
balancing_strategy = class_pre.BALANCING_STRATEGY_MEDIUM
postprocess_to_groups = None
'''
'''
# Settings for monitoring landcover
classtype_to_prepare = 'MONITORING_LANDCOVER'
class_base_dir = os.path.join(base_dir, f"{year}_class_landcover_mon") # Dir for the classification type
balancing_strategy = class_pre.BALANCING_STRATEGY_MEDIUM
postprocess_to_groups = None
'''
#'''
# Settings for monitoring crop groups
classtype_to_prepare = 'MONITORING_CROPGROUPS'
class_base_dir = os.path.join(base_dir, f"{year}_class_maincrops_mon") # Dir for the classification type
balancing_strategy = class_pre.BALANCING_STRATEGY_MEDIUM
postprocess_to_groups = None
# manueel aanpassen bij
#'''
'''
# Settings for monitoring landcover via crop groups
#classtype_to_prepare = 'MONITORING_CROPGROUPS'
#class_base_dir = os.path.join(base_dir, f"{year}_class_landcover_via_cropgroup_mon") # Dir for the classification type
#balancing_strategy = class_pre.BALANCING_STRATEGY_MEDIUM
#postprocess_to_groups = 'MONITORING_LANDCOVER'
# manueel aanpassen bij
'''

# Get a (new) run dir
io_helper.get_run_dir(run_base_dir=class_base_dir,
                      reuse_last_run_dir=False)

log_dir = os.path.join(class_base_dir, 'log')
base_filename = f"{country_code}{year}_bufm10_weekly"
sensordata_to_use = [ts.SENSORDATA_S1_ASCDESC, ts.SENSORDATA_S2gt95]
parceldata_aggregations_to_use = [ts.PARCELDATA_AGGRAGATION_MEAN]

# Check if the necessary input files and directories exist...
if not os.path.exists(input_parcel_filepath):
    message = f"The parcel input file doesn't exist, so STOP: {input_parcel_filepath}"
    print(message)
    raise Exception(message)
if not os.path.exists(imagedata_dir):
    os.mkdir(imagedata_dir)
if not os.path.exists(class_base_dir):
    os.mkdir(class_base_dir)
if not os.path.exists(class_dir):
    os.mkdir(class_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#-------------------------------------------------------------
# Init logging
#-------------------------------------------------------------
# Get root logger
logger = logging.getLogger('')

# Set the general maximum log level...
logger.setLevel(logging.INFO)
for handler in logger.handlers:
    handler.flush()
    handler.close()

# Remove all handlers and add the ones I want again, so a new log file is created for each run
# Remark: the function removehandler doesn't seem to work?
logger.handlers = []

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
#ch.setFormatter(logging.Formatter('%(levelname)s|%(name)s|%(message)s'))
ch.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
logger.addHandler(ch)

log_filepath = os.path.join(log_dir, f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_class_maincrop.log")
fh = logging.FileHandler(filename=log_filepath)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
logger.addHandler(fh)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------
# STEP 1: prepare parcel data for classification and image data extraction
#-------------------------------------------------------------

# Prepare the input data for optimal image data extraction:
#    TODO: 1) reproject to projection used in GEE: EPSG:4326
#    2) apply a negative buffer on the parcel to evade mixels
#    3) remove features that became null because of buffer
imagedata_input_parcel_filename_noext = f"{input_parcel_filename_noext}_bufm10"
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
input_parcel_filepath_gee = f"users/pieter_roggemans/{imagedata_input_parcel_filename_noext}"
ts_calc_gee.calc_timeseries_data(input_parcel_filepath=input_parcel_filepath_gee,
                                 input_country_code=country_code,
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
#             Remarks: - if the classname is 'UNKNOWN', the parcel won't be used for training
#                      - if the classname starts with 'IGNORE_', the parcel will be ignored
#           - pixcount (=global_settings.pixcount_s1s2_column): the number of S1/S2 pixels in the
#             parcel. Is -1 if the parcel doesn't have any S1/S2 data.
parcel_csv = os.path.join(class_dir, f"{input_parcel_filename_noext}_parcel.csv")
parcel_pixcount_csv = os.path.join(imagedata_dir, f"{base_filename}_pixcount.csv")
class_pre.prepare_input(input_parcel_filepath=input_parcel_filepath,
                        input_filetype=input_parcel_filetype,
                        input_parcel_pixcount_csv=parcel_pixcount_csv,
                        output_parcel_filepath=parcel_csv,
                        input_classtype_to_prepare=classtype_to_prepare)

# Combine all data needed to do the classification in one input file
parcel_classification_data_csv = os.path.join(class_dir, f"{base_filename}_parcel_classdata.csv")
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
parcel_train_csv = os.path.join(class_dir, f"{base_filename}_parcel_train.csv")
parcel_test_csv = os.path.join(class_dir, f"{base_filename}_parcel_test.csv")
class_pre.create_train_test_sample(input_parcel_csv=parcel_csv,
                                   output_parcel_train_csv=parcel_train_csv,
                                   output_parcel_test_csv=parcel_test_csv,
                                   balancing_strategy=balancing_strategy)

# Train the classifier and output test predictions
classifier_filepath = os.path.splitext(parcel_train_csv)[0] + "_classifier.pkl"
parcel_predictions_test_csv = os.path.join(class_dir, f"{base_filename}_predict_test.csv")
parcel_predictions_all_csv = os.path.join(class_dir, f"{base_filename}_predict_all.csv")
classification.train_test_predict(input_parcel_train_csv=parcel_train_csv,
                                  input_parcel_test_csv=parcel_test_csv,
                                  input_parcel_all_csv=parcel_csv,
                                  input_parcel_classification_data_csv=parcel_classification_data_csv,
                                  output_classifier_filepath=classifier_filepath,
                                  output_predictions_test_csv=parcel_predictions_test_csv,

                                  output_predictions_all_csv=parcel_predictions_all_csv)
'''
# STEP 5: in necessary, postprocess results
#-------------------------------------------------------------
if postprocess_to_groups is not None:
    ...ToDo
'''

# STEP 6: Report on the test accuracy, incl. ground truth
#-------------------------------------------------------------
# Preprocess the ground truth data
groundtruth_csv = None
if input_groundtruth_csv is not None:
    input_gt_noext, input_gt_ext = os.path.splitext(input_groundtruth_csv)
    groundtruth_csv = os.path.join(class_dir, f"{input_gt_noext}_classes{input_gt_ext}")
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
