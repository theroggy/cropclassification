# -*- coding: utf-8 -*-
"""
Script to create timeseries data per parcel of
  - S1: the mean VV and VH backscatter data
  - S2: the 4 bands for periods when there is good coverage of cloudfree images of the area of
        interest
"""

from __future__ import print_function
from datetime import datetime
from datetime import timedelta
import glob
import logging
import os
import pathlib
import time
from typing import List

import numpy as np
import pandas as pd

# Imports for google earth engine
import ee

# Imports for google drive access
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import googleapiclient

# Import local stuff
import cropclassification.preprocess.timeseries as ts
import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
global_gee_tasks_cache = None

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def calc_timeseries_data(input_parcel_filepath: str,
                         input_country_code: str,
                         start_date_str: str,
                         end_date_str: str,
                         sensordata_to_get: List[str],
                         base_filename: str,
                         dest_data_dir: str):
    """ Calculate timeseries data for the input parcels

    args
    ------------
        data_to_get: an array with data you want to be calculated: check out the constants starting
                     with DATA_TO_GET... for the options.
    """
    # Check some variables...
    if sensordata_to_get is None:
        raise Exception("sensordata_to_get cannot be None")

    # Start calculation of the timeseries on gee
    logger.info("Start create_sentinel_timeseries_info")
    # On windows machines there seems to be an issue with gee. The following error is very common,
    # probably because there are too many sockets created in a short time... and the cleanup
    # procedure in windows can't follow:
    #     "OSError: [WinError 10048] Elk socketadres (protocol/netwerkadres/poort) kan normaal
    #      slechts één keer worden gebruikt"
    # So execute in a loop and retry every 10 seconds... this seems to be a working workaround.
    nb_retries = 0
    done_success = False
    while done_success is False and nb_retries < 10:
        try:
            calculate_sentinel_timeseries(input_parcel_filepath=input_parcel_filepath,
                                          input_country_code=input_country_code,
                                          start_date_str=start_date_str,
                                          end_date_str=end_date_str,
                                          sensordata_to_get=sensordata_to_get,
                                          base_filename=base_filename,
                                          dest_data_dir=dest_data_dir)
            done_success = True

        except OSError as ex:
            nb_retries += 1
            if ex.winerror == 10048:
                logger.warning(f"Exception [WinError {ex.winerror}] while trying calculate_sentinel_timeseries, retry! (Full exception message {ex})")
                time.sleep(10)
            else:
                raise

    # If it wasn't successful, log and stop.
    if done_success is False:
        message = "STOP: calculate_sentinel_timeseries couldn't be completed even after many retries..."
        logger.critical(message)
        raise Exception(message)

    # Download the data from GEE
    return_status = 'UNDEFINED'
    number_retries = 0
    while return_status == 'UNDEFINED' or return_status == 'RETRY_NEEDED':
        # Download the results
        try:
            logger.info('Now download needed timeseries files')
            return_status = download_sentinel_timeseries(dest_data_dir=dest_data_dir,
                                                         base_filename=base_filename)

            # Retry every 10 minutes
            if return_status == 'RETRY_NEEDED':
                logger.info('Not all data was available yet on google drive... try again in a few minutes...')

                # Retry only 36 times, or +- 6 hours
                if number_retries >= 70:
                    return_status = 'STOP'
                    message = "Retried a lot of times, but data still isn't available"
                    logger.error(message)
                    raise Exception(message)

                # Wait for 10 minutes before retrying again... but only sleep 10 seconds at
                # a time so it can be cancelled.
                nb_sleeps = 0
                while nb_sleeps < 30:
                    time.sleep(10)
                    nb_sleeps += 1

                number_retries += 1

        except:
            logger.error('ERROR downloading from google drive!')
            raise

def download_sentinel_timeseries(dest_data_dir: str,
                                 base_filename: str):
    """ Download the timeseries data from gee and clean it up. """

    logger.info("Start download_sentinel_timeseries")

    # Clear the tasks cache so it is loaded again on each run of the function...
    global global_gee_tasks_cache
    global_gee_tasks_cache = None

    # The directory containing the list of filenames of files that need to be downloaded...
    dest_data_dir_todownload = os.path.join(dest_data_dir, 'TODOWNLOAD')

    # Get the list of files to download...
    csv_files_todownload = glob.glob(os.path.join(dest_data_dir_todownload, f'{base_filename}_*.csv'))
    if len(csv_files_todownload) > 0:
        logger.info(f"Process files to download: {len(csv_files_todownload)}")
    else:
        logger.info("No files to download... stop.")
        return 'DOWNLOAD_READY'

    def connect_to_googledrive():
        """ Connect to google drive API """

        # Setup the Drive v3 API
        # We need to be able to read file metadata and file contents
        scopes = 'https://www.googleapis.com/auth/drive.readonly'

        # The client secret file needs to be located in the same dir as the script otherwise it
        # doesn't seem to work
        client_secret_file = 'client_secret_download-gee-data.json'
        store = file.Storage('credentials.json')
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(client_secret_file, scopes)
            creds = tools.run_flow(flow, store)
        return build('drive', 'v3', http=creds.authorize(Http()))

    # Check for each todownload file if it is available on google drive...
    return_status = 'ERROR'
    drive_service = None
    for curr_csv_to_download_path in sorted(csv_files_todownload):

        # Get the basename of the file to download -> this will be the file name in google drive
        curr_csv_to_download_basename = os.path.basename(curr_csv_to_download_path)
        dest_filepath = os.path.join(dest_data_dir, curr_csv_to_download_basename)
        if os.path.isfile(dest_filepath) is True:
            logger.error(f"SKIP, because file exists in destination dir, even though it is still in TODOWNLOAD dir as well???: {curr_csv_to_download_basename}")
            continue

        # If we aren't connected yet to google drive... go for it...
        if drive_service is None:
            drive_service = connect_to_googledrive()

        # Search the file on google drive...
        results = drive_service.files().list(q=f"name = '{curr_csv_to_download_basename}' and trashed != true",
                                             pageSize=100,
                                             fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        # Check the result of the search
        if not items:
            logger.warning(f"File to download not found on drive: {curr_csv_to_download_basename}")
            return_status = 'RETRY_NEEDED'
        elif len(items) > 1:
            logger.error(f"More than one file found for filename: {curr_csv_to_download_basename}")
        else:
            for item in items:

                # Download the file
                logger.info(f"Download file {curr_csv_to_download_basename} ({item['id']})")
                request = drive_service.files().get_media(fileId=item['id'])

                download_done = False
                with open(dest_filepath, "wb") as dest_file:
                    downloader = googleapiclient.http.MediaIoBaseDownload(dest_file, request)
                    while download_done is False:
                        status, download_done = downloader.next_chunk()
                        progress = status.progress()*100
                        logger.info(f"Download now at {progress:.2f} %")

                # Check if download was completed succesfully, and cleanup temp files...
                if download_done is True:
                    clean_gee_downloaded_csv(dest_filepath)
                    os.remove(curr_csv_to_download_path)
                elif os.path.isfile(dest_filepath):
                    logger.error("Download wasn't completed succesfully, remove partially downloaded csv: {dest_filepath}!")
                    os.remove(dest_filepath)
                else:
                    logger.error("Download wasn't completed succesfully and dest csv doesn't exist yet: {dest_filepath}!")

    # If there are no todownload files anymore... download is ready...
    csv_files_todownload = glob.glob(os.path.join(dest_data_dir_todownload, f'{base_filename}_*.csv'))
    if len(csv_files_todownload) == 0:
        return_status = 'DOWNLOAD_READY'

    # Return true if all files were downloaded
    return return_status

def clean_gee_downloaded_csvs_in_dir(dir_path: str):
    """ 
    Cleans csv's downloaded from gee by removing gee specific columns,... 
    """

    # Loop through all csv files in dir and remove the gee columns...
    csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
    for curr_csv in sorted(csv_files):
        clean_gee_downloaded_csv(curr_csv,
                                 remove_orig_csv=False)

def clean_gee_downloaded_csv(csv_file: str,
                             remove_orig_csv: bool = False):
    """ 
    Cleans a csv downloaded from gee by removing gee specific columns... 
    """

    try:
        # Prepare output filename
        file_noext, _ = os.path.splitext(csv_file)
        output_file = f"{file_noext}{conf.general['columndata_ext']}"

        # Check if output file exists already even though it is different from input file
        if output_file != csv_file and os.path.exists(output_file):
            logger.warning(f"Output file exists already, so don't create it again: {output_file}")
        elif os.path.getsize(csv_file) == 0:
            # If input file is empty...
            logger.info(f"File is empty, so just create new empty output file: {output_file}")
            open(output_file, 'w').close()
        else:
            # Read the file
            logger.debug(f"Read file and remove gee specifice columns from {csv_file}")

            # Sample 100 rows of data to determine dtypes, so floats can be read as float32 instead of the 
            # default float64. Writing those to eg. parquet is a lot more efficiënt.
            df_test = pd.read_csv(csv_file, nrows=100)
            float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
            float32_cols = {c: np.float32 for c in float_cols}

            # Now read entire file
            df_in = pd.read_csv(csv_file, engine='c', dtype=float32_cols)
            
            # Drop unnecessary gee specific columns...
            for column in df_in.columns:
                if column in ['system:index', '.geo']:
                    df_in.drop(column, axis=1, inplace=True)
                elif column == 'count':
                    logger.info(f"Rename count column to {conf.columns['pixcount_s1s2']}")
                    df_in.rename(columns={'count': conf.columns['pixcount_s1s2']}, inplace=True)

            # Set the id column as index
            #df_in.set_index('CODE_OBJ', inplace=True)
            df_in.set_index(conf.columns['id'], inplace=True)

            # If there are data columns, write to output file
            if len(df_in.columns) > 0:
                # Replace the original file by the cleaned one
                logger.info(f"Write the file with the gee specific columns removed to a new file: {output_file}")
                pdh.to_file(df_in, output_file, index=True)
            else:
                logger.warning(f"No data columns found in file {csv_file}, so return!!!")
                return            

        # If remove_orig_csv is True and the output filepath is different from orig filepath, 
        # remove orig file. 
        if remove_orig_csv and output_file != csv_file:
            logger.info(f"Remove orig csv file: {csv_file}")
            os.remove(csv_file)

    except Exception as ex:
        raise Exception(f"Error processing file {csv_file}") from ex

def calculate_sentinel_timeseries(input_parcel_filepath: str,
                                  input_country_code: str,
                                  start_date_str: str,
                                  end_date_str: str,
                                  sensordata_to_get: List[str],
                                  base_filename: str,
                                  dest_data_dir: str):
    '''
    Credits: partly based on a gee S1 extraction script written by Guido Lemoine.

    TODO: it would be cleaner if feature of interest didn't have to be passed explicitly... but could be calculated from input_parcels... but too slow??
    TODO: some code is still old code from the original js version that could be cleaner by just writing it in python.
          this would also result in using less sockets, and less error that sockets can't be reused on windows ;-).
    '''

    # Init some variables
    dest_data_dir_todownload = os.path.join(dest_data_dir, 'TODOWNLOAD')
    if not os.path.exists(dest_data_dir_todownload):
        os.mkdir(dest_data_dir_todownload)


    # Prepare filepath as it is available on gee
    input_parcel_filename = os.path.basename(input_parcel_filepath)
    input_parcel_filename_noext = os.path.splitext(input_parcel_filename)
    input_parcel_filepath_gee = f"{conf.dirs['gee']}{input_parcel_filename_noext}"

    # Initialize connection to server
    ee.Initialize()

    if input_country_code == 'BEFL':
        # Define the bounds of flanders as region of interest
        region_of_interest = ee.Geometry.Polygon(
            [[[3.219141559645095, 51.382839056214074],
              [2.395251749811223, 51.08622074707946],
              [2.6479083799461023, 50.75729132119862],
              [2.8071930702097916, 50.67382488209542],
              [4.05950498184086, 50.639001278789706],
              [5.460117113586875, 50.67382291700078],
              [5.927674232135132, 50.70514081680863],
              [5.8355122769162335, 50.95084747021692],
              [5.852391042458976, 51.06851086546833],
              [5.86840409339311, 51.13983411545178],
              [5.852835448995961, 51.18272203541686],
              [5.460115101564611, 51.327099242895095],
              [5.28459662023829, 51.413048026378355],
              [5.045432234403279, 51.494974825956845],
              [4.759572600621368, 51.50915727919978],
              [4.2627231304370525, 51.47616009334954],
              [4.17759696889857, 51.314226452568576]]])
    else:
        countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        region_of_interest = ee.Feature(countries
                                        .filterMetadata('country_co', 'equals', input_country_code)
                                        .union().first()).buffer(1000).geometry()

    # TODO: add check if the ID column exists in the parcel file, otherwise he processes everything without ID column in output :-(!!!

    # First adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
    logger.info('Adapt start_date and end_date so they are mondays')
    def get_monday(date_str):
        """ Get the first monday before the date provided. """
        parseddate = datetime.strptime(date_str, '%Y-%m-%d')
        year_week = parseddate.strftime('%Y_%W')
        year_week_monday = datetime.strptime(year_week + '_1', '%Y_%W_%w')
        return year_week_monday

    start_date = get_monday(start_date_str)
    end_date = get_monday(end_date_str)       # Remark: de end date is exclusive in gee filtering, so must be a monday as well...
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Keep the tasklist in a variable so it is loaded only once (if necessary)

#    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
#    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Prepare the input vector data we want to add timeseries info to...
    input_parcels = ee.FeatureCollection(input_parcel_filepath_gee)

    logging.info(f'Create sentinel timeseries from {start_date} till {end_date} for parcel in file {input_parcel_filepath_gee}')

    # Add some columns with feature info, to be able to eg. filter only polygons...
#    def add_feature_info(feature):
#        return feature.set('area', feature.area(), 'perimeter', feature.perimeter(), 'type', feature.geometry().type())
#    bevl2017 = bevl2017.map(add_feature_info)

    '''
    # Buffer
    # Now define a function to buffer the parcels inward
    def bufferFeature(ft):
      ft = ft.simplify(1).buffer(-20).simplify(1)
#     return ft.set({ncoords: ft.geometry().coordinates().length()})
      return ft
    input_parcels = input_parcels.map(bufferFeature)
    '''

    # Export non-polygons..
#    bevl2017_nopoly = bevl2017.filterMetadata('type', 'not_equals', 'Polygon')
#    ee.batch.Export.table.toDrive(collection = bevl2017_nopoly, folder = 'Monitoring', description = 'BEVL2017_no_polygon', fileFormat = 'KMZ')

    # Filter the S1 data we want to have (VV and VH pol.)
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH']))
          .filterBounds(region_of_interest)
          .filterDate(start_date_str, end_date_str).sort('system:time_start'))

    # Remove ugly edges from S1
    def mask_s1_edge(img):
        """ Masks the edges of S1 images where backscatter is close to zero... """
        mask = (img.select(0).unitScale(-25, 5).multiply(255).toByte()
                .connectedComponents(ee.Kernel.rectangle(1, 1), 100))
        return img.updateMask(mask.select(0))
    s1 = s1.map(mask_s1_edge)

    # Functions to convert from/to dB
    def to_natural(img):
        """ Converts an image to natural values from db values. """
        return (ee.Image(10.0).pow(img.select('..').divide(10.0))
                .copyProperties(img, ['system:time_start', 'orbitProperties_pass']))

    def to_db(img):
        """ Converts an image to db values from natural values. """
        return ee.Image(img).log10().multiply(10.0)
    s1 = s1.map(to_natural)

    # Load interesting S2 images
    s2 = (ee.ImageCollection('COPERNICUS/S2')
          .filterBounds(region_of_interest)
          .filterDate(start_date_str, end_date_str)
          .filter(ee.Filter.lessThan('CLOUDY_PIXEL_PERCENTAGE', 10)))

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bitmask = ee.Number(2).pow(10).int()
    cirrus_bitmask = ee.Number(2).pow(11).int()

    def mask_S2_clouds(image):
        """ Mask the clouds of a sentinel 2 image. """
        qa = image.select('QA60')
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloud_bitmask).eq(0).And(qa.bitwiseAnd(cirrus_bitmask).eq(0))
        return image.updateMask(mask)
    s2 = s2.map(mask_S2_clouds)

    # Olha's idea to create weekly mean images
    step = 7 # in days
    def nextday(date):
        """ Returns the next day. """
        return ee.Date(start_date_str).advance(date, "day")

    days = ee.List.sequence(0,
                            ee.Date(end_date_str).difference(ee.Date(start_date_str), 'day'),
                            step).map(nextday)
    periods = days.slice(0, -1).zip(days.slice(1))

    # Function to get a string representation for a period
    def get_period_str(period):
        """ Gets a string representation for the period. """
        return ee.Date(ee.List(period).get(0)).format('YYYYMMdd')

    # Get the S1's for a period...
    def get_s1s_forperiod(period):
        return s1.filterDate(ee.List(period).get(0), ee.List(period).get(1)).select(['VV', 'VH'])

    # Get the S1's for a period...
    def get_s1_forperiod(period):
        period_str = get_period_str(period)
        return (s1.filterDate(ee.List(period).get(0), ee.List(period).get(1))
                .mean()
                .select(['VV', 'VH'], [ee.String('VV_').cat(period_str),
                                       ee.String('VH_').cat(period_str)]))

    # Get the S1's for a period...
    def get_s1_asc_forperiod(period):
        period_str = get_period_str(period)
        s1_asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        return (s1_asc.filterDate(ee.List(period).get(0), ee.List(period).get(1))
                .mean()
                .select(['VV', 'VH'], [ee.String('VV_ASC_').cat(period_str),
                                       ee.String('VH_ASC_').cat(period_str)]))

    # Get the S1's for a period...
    def get_s1_desc_forperiod(period):
        period_str = get_period_str(period)
        s1_desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        return (s1_desc.filterDate(ee.List(period).get(0), ee.List(period).get(1))
                .mean()
                .select(['VV', 'VH'], [ee.String('VV_DESC_').cat(period_str),
                                       ee.String('VH_DESC_').cat(period_str)]))

    # Get the S2's for a period...
    # Remark: median on an imagecollection apparently has a result that
    # the geometry of the image is not correct anymore... so median only is done afterwards.
    def get_s2s_forperiod(period):
        return s2.filterDate(ee.List(period).get(0), ee.List(period).get(1))

    # Get a list of all tasks in gee
    # These are the different states that exist:
    #     - READY    : the task is submitted to the gee server, but not yet running
    #     - RUNNING  : the task is running on gee
    #     - COMPLETED: the task is completely completed
    #     - FAILED   : an error occured while processing the task and the tas stopped without result
    def get_gee_tasklist():
        tasklist = ee.batch.data.getTaskList()
        logger.info(f"Number of tasks: {len(tasklist)}")
        return tasklist

    # Remark: In some cases (eg. while debugging/testing) it would be easier that there would be an
    #         option to ignore completed ones. But there is an easy workaround: just change the
    #         basefilename in any way to ignore the completed ones.
    def check_if_task_exists(task_description: str, task_state_list) -> str:
        """ Checks if a task exists already on gee """

        # If the tasks aren't retrieved yet, do so...
        global global_gee_tasks_cache
        if global_gee_tasks_cache is None:
            global_gee_tasks_cache = get_gee_tasklist()

        # Check if there is a task with this name and this state (there can be multiple tasks with
        # this name!)
        for task in global_gee_tasks_cache:
            if(task['description'] == task_description
               and task['task_type'] == 'EXPORT_FEATURES'
               and task['state'] in task_state_list):
                logger.debug(f"<check_if_task_exists> Task {task_description} found with state {task['state']}")
                return True
        logger.debug(f"<check_if_task_exists> Task {task_description} doesn't exist with any of the states in {task_state_list}")
        return False

    def reduce_and_export(imagedata, reducer, export_descr: str):
        """ Reduces the imagedata over the features and export to drive. """

        # First check if the file exists already locally...
        # Format relevant local filename
        export_filename = export_descr + '.csv'
        dest_fullpath = os.path.join(dest_data_dir, export_filename)
        dest_fullpath_todownload = os.path.join(dest_data_dir_todownload, export_filename)

        # If the data is already available locally... go to next period
        if os.path.isfile(dest_fullpath):
            logger.info(f"For task {export_descr}, file already available locally: SKIP")
            return

        # If the data is already "ordered" in a previous run and is still busy processing, don't
        # start processing again
        if (os.path.isfile(dest_fullpath_todownload)
                and(check_if_task_exists(export_description, ['RUNNING', 'READY', 'COMPLETED']))):
            logger.info(f"For task {export_descr}, file still busy processing or is ready on gee: SKIP")
            return

        # Get the sentinel data for each parcel
        # Remark: from the input parcels, only keep the ID column...
        imagedata_perparcel = imagedata.reduceRegions(collection=input_parcels.select([conf.columns['id']]),
                                                      reducer=reducer,
                                                      scale=10)

        # Set the geometries to none, as we don't want to export them... and parameter
        # retainGeometry=False in select doesn't seem to work.
        def geom_to_none(feature):
            """ Set the geom of the feature to none. """
            return feature.setGeometry(None)
        imagedata_perparcel = imagedata_perparcel.map(geom_to_none)

        # Export to google drive
        export_task = ee.batch.Export.table.toDrive(collection=imagedata_perparcel,
                                                    folder='Monitoring',
                                                    description=export_descr,
                                                    fileFormat='CSV')
        ee.batch.Task.start(export_task)

        # Create file in todownload folder to indicate this file should be downloaded
        pathlib.Path(dest_fullpath_todownload).touch()

    # If the file doesn't exist yet... export the parcel with all interesting columns to csv...
    s1_for_count = (s1.filterDate(ee.List(periods.get(0)).get(0), ee.List(periods.get(0)).get(1))
                    .mean().select(['VV'], ['pixcount']))
    export_description = f"{base_filename}_pixcount"
    reduce_and_export(imagedata=s1_for_count,
                      reducer=ee.Reducer.count(),
                      export_descr=export_description)

    # Loop over all periods and export data per period to drive
    # Create the reducer we want to use...
    # Remark: always use both the mean and stdDev reducer, the stdDev is useful for detecting if
    #         the parcel isn't one crop in any case, and that way the name of the columns always
    #         end with the aggregation/reduce type used, otherwise it doesn't and other code will
    #         break
    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
#    reducer = ee.Reducer.mean()
    nb_periods = periods.length().getInfo()
    logger.info(f"Loop through all <{nb_periods}> periods")
    for i in range(0, nb_periods):

        # Calculate the start and end dates of this period...
        period_start_str = (start_date + timedelta(days=i*7)).strftime('%Y-%m-%d')
        period_end_str = (start_date + timedelta(days=(i+1)*7)).strftime('%Y-%m-%d')
        logger.debug(f"Process period: {period_start_str} till {period_end_str}")

        def merge_bands(image1, image2):
            """
            Merges the bands of the two images, without getting errors if one of the images is None.
            If both images are None, None will be returned.
            """

            if image1 is not None:
                if image2 is not None:
                    return image1.addBands(image2)
                else:
                    return image1
            else:
                return image2

        # Get mean s1 image of the s1 images that are available in this period
        SENSORDATA_S1 = conf.general['SENSORDATA_S1']
        if SENSORDATA_S1 in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
                return
            else:
                # Now the real work
                s1_forperiod = get_s1_forperiod([period_start_str, period_end_str])
                reduce_and_export(imagedata=s1_forperiod,
                                  reducer=reducer,
                                  export_descr=sensordata_descr)

        # Get mean s1 image of the s1 images that are available in this period
        SENSORDATA_S1DB = conf.general['SENSORDATA_S1DB']
        if SENSORDATA_S1DB in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1DB}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
                return
            else:
                # Now the real work
                s1_forperiod = get_s1_forperiod([period_start_str, period_end_str])
                reduce_and_export(imagedata=to_db(s1_forperiod),
                                  reducer=reducer,
                                  export_descr=sensordata_descr)

        # Get mean s1 asc and desc image of the s1 images that are available in this period
        SENSORDATA_S1_ASCDESC = conf.general['SENSORDATA_S1_ASCDESC']
        if SENSORDATA_S1_ASCDESC in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1_ASCDESC}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
            else:
                # Now the real work
                s1_asc_forperiod = get_s1_asc_forperiod([period_start_str, period_end_str])
                s1_desc_forperiod = get_s1_desc_forperiod([period_start_str, period_end_str])
                imagedata_forperiod = merge_bands(s1_asc_forperiod, s1_desc_forperiod)
                reduce_and_export(imagedata=imagedata_forperiod,
                                  reducer=reducer,
                                  export_descr=sensordata_descr)

        # Get mean s1 in DB, asc and desc image of the s1 images that are available in this period
        SENSORDATA_S1DB_ASCDESC = conf.general['SENSORDATA_S1DB_ASCDESC']
        if SENSORDATA_S1DB_ASCDESC in sensordata_to_get:
            # If the data is already available locally... skip
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S1DB_ASCDESC}"
            if os.path.isfile(os.path.join(dest_data_dir, f"{sensordata_descr}.csv")):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
            else:
                # Now the real work
                s1_asc_forperiod = get_s1_asc_forperiod([period_start_str, period_end_str])
                s1_desc_forperiod = get_s1_desc_forperiod([period_start_str, period_end_str])
                imagedata_forperiod = merge_bands(s1_asc_forperiod, s1_desc_forperiod)
                reduce_and_export(imagedata=to_db(imagedata_forperiod),
                                  reducer=reducer,
                                  export_descr=sensordata_descr)

        # Get mean s2 image of the s2 images that have (almost)cloud free images available in this
        # period
        SENSORDATA_S2gt95 = conf.general['SENSORDATA_S2gt95']
        if SENSORDATA_S2gt95 in sensordata_to_get:
            sensordata_descr = f"{base_filename}_{period_start_str}_{SENSORDATA_S2gt95}"

            # If the data is already available locally... skip
            # Remark: this logic is puth here additionaly to evade having to calculate the 95% rule
            #         even if data is available.
            dest_fullpath = os.path.join(dest_data_dir, f"{sensordata_descr}.csv")
            if os.path.isfile(dest_fullpath):
                logger.info(f"For task {sensordata_descr}, file already available locally: SKIP")
            else:
                # Get s2 images that are available in this period
                s2s_forperiod = get_s2s_forperiod([period_start_str, period_end_str])

                # If S2 available for entire flanders... export S2 as well
                s2_pct_bevl = (region_of_interest.intersection(s2s_forperiod.geometry())
                               .area().divide(region_of_interest.area()))
                if s2_pct_bevl.getInfo() > 0.95:
                    s2_forperiod = (s2s_forperiod.median()
                                    .select(['B2', 'B3', 'B4', 'B8'],
                                            [ee.String('S2B2_').cat(period_start_str),
                                             ee.String('S2B3_').cat(period_start_str),
                                             ee.String('S2B4_').cat(period_start_str),
                                             ee.String('S2B8_').cat(period_start_str)
                                            ]))
                else:
                    # Create an empty destination file so we'll know that we tested the 95% already
                    # and don't need to do it again...
                    pathlib.Path(dest_fullpath).touch()
                    continue

#                logger.debug(f"S2 Bands: {ee.Image(s2_forperiod).bandNames().getInfo()}")
                reduce_and_export(imagedata=s2_forperiod,
                                  reducer=reducer,
                                  export_descr=sensordata_descr)

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
