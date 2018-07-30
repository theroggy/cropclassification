# -*- coding: utf-8 -*-
"""
Main script to do a classification.

@author: Pieter Roggemans

"""

import logging
import os
import datetime
import timeseries_s1s2_preprocessing as timeseries_pre
import timeseries_s1s2_gee as timeseries
import classification_preprocessing as class_pre
import classification
import classification_reporting as class_report

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
base_dir = 'C:\\temp\\CropClassification'                                               # Base dir
input_dir = os.path.join(base_dir, 'InputData')                                         # Input dir
input_preprocessed_dir = os.path.join(input_dir, 'Preprocessed')
input_parcel_filename_noext = 'Prc_flanders_2017_2018-01-09'                            # Input filename
input_parcel_filepath = os.path.join(input_dir, f"{input_parcel_filename_noext}.shp")   # Input filepath of the parcel
input_groundtruth_noext = 'Prc_flanders_2017_groundtruth'
input_groundtruth_csv = os.path.join(input_dir, f"{input_groundtruth_noext}.csv")       # The ground truth
input_parcel_filetype = 'BEFL'

imagedata_dir = os.path.join(base_dir, 'Timeseries_data')               # General data download dir
start_date_str = '2017-04-01'
end_date_str = '2017-09-15'                                             # End date is NOT inclusive for gee processing

# REMARK: the column names that are used/expected can be found in global_constants.py!
'''
# Settings for 7 main crops
output_classes_type = 'MOST_POPULAR_CROPS'
class_base_dir = os.path.join(base_dir, 'class_maincrops7')             # Specific dir for this classification
balancing_strategy = class_pre.BALANCING_STRATEGY_EQUAL
'''

# Settings for monitoring crop groups
output_classes_type = 'MONITORING_CROPGROUPS'
class_base_dir = os.path.join(base_dir, 'class_maincrops_mon')          # Specific dir for this classification
balancing_strategy = class_pre.BALANCING_STRATEGY_MEDIUM


class_dir = os.path.join(class_base_dir, '2018-07-30_Run1_bufm10_Standard')
log_dir = os.path.join(class_dir, 'log')
base_filename = 'BEVL2017_weekly_bufm10_stddev'

if not os.path.exists(input_parcel_filepath):
    print (f"CRITICAL: the parcel input file doesn't exist, so STOP: {input_parcel_filepath}")
    raise
if not os.path.exists(imagedata_dir):
    os.mkdir(imagedata_dir)
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

log_filename = os.path.join(log_dir, f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_class_maincrop.log')
fh = logging.FileHandler(filename=log_filename)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
logger.addHandler(fh)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------
# TODO: performance could be optimized by providing the option that the intermediate csv files aren't always written to disk and
#       read again by returning the result as DataSet... (or at least only written, not read again).

# STEP 1: prepare parcel data for classification and image data extraction
#-------------------------------------------------------------

# Prepare the input data for optimal image data extraction:
#   ( 1) reproject to projection used for Sentinel, for BEFL: epsg:32631)
#    2) apply a negative buffer on the parcel to evade mixels
#    3) remove features that became null because of buffer
imagedata_input_parcel_filename_noext = f"{input_parcel_filename_noext}_bufm10"
imagedata_input_parcel_filepath = os.path.join(input_preprocessed_dir, f"{imagedata_input_parcel_filename_noext}.shp")

timeseries_pre.prepare_input(input_parcel_filepath = input_parcel_filepath
                            ,output_imagedata_parcel_input_filepath = imagedata_input_parcel_filepath)

# STEP 2: Get the timeseries data needed for the classification
#-------------------------------------------------------------
# Get the time series data (S1 and S2) to be used for the classification later on...
# Result: data is put in csv files in imagedata_dir, in one csv file per date/period
# Remarks:
#    - the path to the imput data is specific for gee... so will need to be changed if another implementation is used
#    - the upload to gee as an asset is not implemented, because it need a google cloud account... so upload needs to be done manually
# TODO: probably the period of aggregation of the data should be a parameter to increase reusability?
input_parcel_filepath_gee = f'users/pieter_roggemans/{imagedata_input_parcel_filename_noext}'

timeseries.get_timeseries_data(input_parcel_filepath = input_parcel_filepath_gee
                              ,start_date_str = start_date_str
                              ,end_date_str = end_date_str
                              ,base_filename = base_filename
                              ,dest_data_dir = imagedata_dir)

# STEP 3: Preprocess all data needed for the classification
#-------------------------------------------------------------
# First prepare the basic input file with the classes that will be classified to.
# Remarks:
#    - this is typically specific for the input dataset and result wanted!!!
#    - the result is/should be a csv file with the following columns
#           - id        (=ID_COLUMN_NAME)   : unique ID for the parcel
#           - classname (=CLASS_COLUMN_NAME): the class that must be classified to.
#             Remarks: - if the classname is 'UNKNOWN', the parcel won't be used for training
#                      - if the classname starts with 'IGNORE_', the parcel will be ignored in general
parcel_classes_csv = os.path.join(class_dir, f"{input_parcel_filename_noext}_classes.csv")
class_pre.prepare_input(input_parcel_filepath = input_parcel_filepath
                       ,input_filetype = input_parcel_filetype
                       ,output_parcel_filepath = parcel_classes_csv
                       ,output_classes_type = output_classes_type)

# Combine all data needed to do the classification in one input file
parcel_classification_data_csv = os.path.join(class_dir, f"{base_filename}_classdata.csv")
class_pre.collect_and_prepare_timeseries_data(imagedata_dir = imagedata_dir
                     ,base_filename = base_filename
                     ,start_date_str = start_date_str
                     ,end_date_str = end_date_str
                     ,output_csv = parcel_classification_data_csv)

# STEP 4: Train and test the classification
#-------------------------------------------------------------
# Create the training sample...
# Remark: this creates a list of representative test parcel + a list of (candidate) training parcel.
parcel_classes_train_csv = os.path.join(class_dir, f"{base_filename}_parcel_classes_train.csv")
parcel_classes_test_csv = os.path.join(class_dir, f"{base_filename}_parcel_classes_test.csv")
parcel_pixcount_csv = os.path.join(imagedata_dir, f"{base_filename}_pixcount.csv")

class_pre.create_train_test_sample(input_parcel_classes_csv = parcel_classes_csv
                                  ,input_parcel_pixcount_csv = parcel_pixcount_csv
                                  ,output_parcel_classes_train_csv = parcel_classes_train_csv
                                  ,output_parcel_classes_test_csv = parcel_classes_test_csv
                                  ,balancing_strategy = balancing_strategy)

# Train the classifier and output test predictions
classifier_filepath = os.path.splitext(parcel_classes_train_csv)[0] + "_classifier.pkl"
parcel_test_predictions_csv = os.path.join(class_dir, f"{base_filename}_predict_test.csv")
classification.train_and_test(input_parcel_classes_train_csv = parcel_classes_train_csv
                    ,input_parcel_classes_test_csv = parcel_classes_test_csv
                    ,input_parcel_classification_data_csv = parcel_classification_data_csv
                    ,output_classifier_filepath = classifier_filepath
                    ,output_test_predictions_csv = parcel_test_predictions_csv)

# STEP 5: Report on the test accuracy
#-------------------------------------------------------------
# Preprocess the ground truth data
groundtruth_classes_csv = os.path.join(class_dir, f"{input_groundtruth_noext}_classes.csv")
class_pre.prepare_input(input_parcel_filepath = input_groundtruth_csv
                       ,input_filetype = input_parcel_filetype
                       ,output_parcel_filepath = groundtruth_classes_csv
                       ,output_classes_type = f"{output_classes_type}_GROUNDTRUTH")

# Print full reporting on the accuracy
report_txt = f"{parcel_test_predictions_csv}_accuracy_report.txt"
class_report.write_full_report(parcel_predictions_csv = parcel_test_predictions_csv
                              ,parcel_classes_test_csv = parcel_classes_test_csv
                              ,parcel_ground_truth_csv = groundtruth_classes_csv
                              ,output_report_txt = report_txt)

# Print a confusion matrix to asses the accuracy per pixcount, using consolidated prediction
report_txt = f"{parcel_test_predictions_csv}_accuracy_report_pixcount.txt"
class_report.write_OA_per_pixcount(parcel_predictions_csv = parcel_test_predictions_csv
                                  ,parcel_pixcount_csv = parcel_pixcount_csv
                                  ,output_report_txt = report_txt)

# STEP 6: Do the actual classification
#-------------------------------------------------------------
# Predict for all parcels
parcel_all_predictions_csv = os.path.join(class_dir, f"{base_filename}_predict_all.csv")
classification.predict(input_parcel_classes_csv = parcel_classes_csv
                    ,input_parcel_classification_data_csv = parcel_classification_data_csv
                    ,input_classifier_filepath = classifier_filepath
                    ,output_predictions_csv = parcel_all_predictions_csv)

# STEP 7: Report on the full accuracy
#-------------------------------------------------------------
# Print full reporting on the accuracy
report_txt = f"{parcel_all_predictions_csv}_accuracy_report.txt"
class_report.write_full_report(parcel_predictions_csv = parcel_all_predictions_csv
                              ,parcel_classes_test_csv = parcel_classes_csv
                              ,parcel_ground_truth_csv = groundtruth_classes_csv
                              ,output_report_txt = report_txt)

# Print a confusion matrix to asses the accuracy per pixcount, using consolidated prediction
report_txt = f"{parcel_all_predictions_csv}_accuracy_report_pixcount.txt"
class_report.write_OA_per_pixcount(parcel_predictions_csv = parcel_all_predictions_csv
                                  ,parcel_pixcount_csv = parcel_pixcount_csv
                                  ,output_report_txt = report_txt)


logging.shutdown()