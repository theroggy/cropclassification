# -*- coding: utf-8 -*-
"""
Main script to run detection of multiple crops in a parcel.

REMARK: is still experimental, not operational yet!

@author: Pieter Roggemans

"""

import logging
import os
import datetime
import timeseries_calc_preprocess as ts_pre
import timeseries_calc_gee as ts_calc
import classification_preprocess as class_pre
import timeseries as ts
import multicrop

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
year = 2018
country_code = 'BEFL'        # The region of the classification: typically country code

base_dir = 'X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-05-04_Monitoring_Classificatie'                                               # Base dir
input_dir = os.path.join(base_dir, 'InputData')                                         # Input dir
input_preprocessed_dir = os.path.join(input_dir, 'Preprocessed')

# Input file depends on the year
if year == 2017:
    input_parcel_filename_noext = 'Prc_flanders_2017_2018-01-09'                        # Input filename
elif year == 2018:
    input_parcel_filename_noext = 'Prc_BEFL_2018_2018-08-02'                            # Input filename
input_parcel_filepath = os.path.join(input_dir, f"{input_parcel_filename_noext}.shp")   # Input filepath of the parcel

input_parcel_filetype = country_code
imagedata_dir = os.path.join(base_dir, 'Timeseries_data')      # General data download dir
start_date_str = f"{year}-06-15"
end_date_str = f"{year}-07-15"                                 # End date is NOT inclusive for gee processing

# REMARK: the column names that are used/expected can be found/changed in global_constants.py!

# Settings for monitoring crop groups
classtype_to_prepare = 'MONITORING_CROPGROUPS'
class_base_dir = os.path.join(base_dir, f"{year}_multicrop") # Dir for the classification type

class_dir = os.path.join(class_base_dir, '2018-10-02_Run2_bufm10')
log_dir = os.path.join(class_dir, 'log')
base_filename = f"{country_code}{year}_bufm10_weekly"
sensordata_to_use = [ts.SENSORDATA_S1DB_ASCDESC]
parceldata_aggregations_to_use = [ts.PARCELDATA_AGGRAGATION_STDDEV]

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

log_filepath = os.path.join(log_dir, f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_detect_multicrop.log")
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
#imagedata_input_parcel_filename_noext = f"{input_parcel_filename_noext}"
'''
imagedata_input_parcel_filepath = os.path.join(input_preprocessed_dir, f"{imagedata_input_parcel_filename_noext}.shp")
timeseries_pre.prepare_input(input_parcel_filepath=input_parcel_filepath,
                             output_imagedata_parcel_input_filepath=imagedata_input_parcel_filepath)
'''
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
ts_calc.calc_timeseries_data(input_parcel_filepath=input_parcel_filepath_gee,
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

# TODO: this isn't a proper prepare function yet for this marker!!!
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
                                       min_fraction_data_in_column=0.0,
                                       sensordata_to_use=sensordata_to_use,
                                       parceldata_aggregations_to_use=parceldata_aggregations_to_use)

multicrop.detect_multicrop(input_parcel_csv=parcel_csv,
                           input_parcel_timeseries_data_csv=parcel_classification_data_csv)

# STEP 4: Train, test and classify
#-------------------------------------------------------------
# Create the training sample...

logging.shutdown()
