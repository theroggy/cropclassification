# -*- coding: utf-8 -*-
"""
Script to create timeseries data per parcel of
  - S1: the mean VV and VH backscatter data
  - S2: the 4 bands for periods when there is good coverage of cloudfree images of the area of interest

@author: Pieter Roggemans
"""

from __future__ import print_function
import logging
import os
import glob
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pathlib

# Imports for google earth engine
import ee

# Imports for google drive access
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import googleapiclient

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
import global_settings as gs

# Get a logger...
logger = logging.getLogger(__name__)
global_gee_tasks_cache = None

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def get_timeseries_data(input_parcel_filepath: str
                       ,start_date_str: str
                       ,end_date_str: str
                       ,base_filename: str
                       ,dest_data_dir: str):

    # Start calculation of the timeseries on gee
    logger.info("Start create_sentinel_timeseries_info")
    calculate_sentinel_timeseries(input_parcel_filepath = input_parcel_filepath
                                 ,start_date_str = start_date_str
                                 ,end_date_str = end_date_str
                                 ,base_filename = base_filename
                                 ,dest_data_dir = dest_data_dir)

    # Download the data from GEE
    return_status = 'UNDEFINED'
    number_retries = 0
    while return_status == 'UNDEFINED' or return_status == 'RETRY_NEEDED':
        # Download the results
        try:
            logger.info('Now download needed timeseries files')
            return_status = download_sentinel_timeseries(dest_data_dir = dest_data_dir
                                                        ,base_filename = base_filename)

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

def download_sentinel_timeseries(dest_data_dir: str
                                ,base_filename: str):

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
        # Setup the Drive v3 API
        # We need to be able to read file metadata and file contents
        SCOPES = 'https://www.googleapis.com/auth/drive.readonly'

        # The client secret file needs to be located in the same dir as the script otherwise it doesn't seem to work
        client_secret_file = 'client_secret.json'
        store = file.Storage('credentials.json')
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(client_secret_file, SCOPES)
            creds = tools.run_flow(flow, store)
        return build('drive', 'v3', http=creds.authorize(Http()))

    # Check for each todownload file if it is available on google drive...
    return_status = 'ERROR'
    drive_service = None
    for curr_csv_to_download_path in sorted(csv_files_todownload):

        # Get the basename of the file to download -> this will be the file name in google drive
        curr_csv_to_download_basename = os.path.basename(curr_csv_to_download_path)
        dest_filepath = os.path.join(dest_data_dir, curr_csv_to_download_basename)
        if os.path.isfile(dest_filepath):
            logger.error(f"SKIP, because file exists in destination dir, even though it is still in TODOWNLOAD dir as well???: {curr_csv_to_download_basename}")
            continue

        # If we aren't connected yet to google drive... go for it...
        if drive_service == None:
            drive_service = connect_to_googledrive()

        # Search the file on google drive...
        results = drive_service.files().list(q = f"name = '{curr_csv_to_download_basename}' and trashed != true"
                                            ,pageSize = 100
                                            ,fields = "nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        # Check the result of the search
        if not items:
            logger.warn(f"File to download not found on drive: {curr_csv_to_download_basename}")
            return_status = 'RETRY_NEEDED'
        elif (len(items) > 1):
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
                if download_done == True:
                    remove_gee_columns_from_csv(dest_filepath)
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

def remove_gee_columns_from_csv_in_dir(dir_path: str):
    # Loop through all csv files in dir and remove the gee columns...
    csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
    for curr_csv in sorted(csv_files):
        remove_gee_columns_from_csv(curr_csv)

def remove_gee_columns_from_csv(csv_file: str):
    logger.debug(f"Remove gee specifice columns from {csv_file}")
    # Read the file
    df_in = pd.read_csv(csv_file)

    # Drop unnecessary gee specific columns...
    column_dropped = False
    for column in df_in.columns :
        if column in ['system:index', '.geo']:
            df_in.drop(column, axis=1, inplace=True)
            column_dropped = True

    # If a column, was dropped... replace the original file by the cleaned one
    if column_dropped == True:
        logger.info(f"Replace the csv file with the gee specific columns removed: {csv_file}")
        csv_file_tmp = f"{csv_file}_cleaned.tmp"
        df_in.to_csv(csv_file_tmp, index=False)
        os.replace(csv_file_tmp, csv_file)

def calculate_sentinel_timeseries(input_parcel_filepath: str
                                 ,start_date_str: str
                                 ,end_date_str: str
                                 ,base_filename: str
                                 ,dest_data_dir: str):
    '''
    Credits: partly based on a gee S1 extraction script written by Guido Lemoine.
    '''
    dest_data_dir_todownload = os.path.join(dest_data_dir, 'TODOWNLOAD')
    if not os.path.exists(dest_data_dir_todownload):
        os.mkdir(dest_data_dir_todownload)

    # Initialize connection to server
    ee.Initialize()

    # Define the bounds of flanders
    bevl_geom = ee.Geometry.Polygon(
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
          [4.17759696889857, 51.314226452568576]]]);

    # Define the bounds of Belgium -> not needed, we use flanders above...
    #countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017");
    #be_geom = ee.Feature(countries.filterMetadata('country_na', 'equals', 'Belgium').union().first()).buffer(1000).geometry()

    # First adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
    logger.info('Adapt start_date and end_date so they are mondays')
    def get_monday(date_str):
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
    bevl2017 = ee.FeatureCollection(input_parcel_filepath)

    logging.info(f'Create sentinel timeseries from {start_date} till {end_date} for parcel in file {input_parcel_filepath}')

    # Add some columns with feature info, to be able to eg. filter only polygons...
#    def add_feature_info(feature):
#        return feature.set('area', feature.area(), 'perimeter', feature.perimeter(), 'type', feature.geometry().type())
#    bevl2017 = bevl2017.map(add_feature_info)

    # Export non-polygons..
    #bevl2017_nopoly = bevl2017.filterMetadata('type', 'not_equals', 'Polygon')
    #ee.batch.Export.table.toDrive(collection = bevl2017_nopoly, folder = 'Monitoring', description = 'BEVL2017_no_polygon', fileFormat = 'KMZ')

    # Filter the S1 data we want to have (VV and VH pol.)
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD'
                            ).filterMetadata('instrumentMode', 'equals', 'IW'
                            ).filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH'])
                            ).filterBounds(bevl_geom
                            ).filterDate(start_date_str, end_date_str).sort('system:time_start')

    # Remove ugly edges from S1
    def maskEdge(img):
        mask = img.select(0).unitScale(-25, 5).multiply(255).toByte().connectedComponents(ee.Kernel.rectangle(1,1), 100);
        return img.updateMask(mask.select(0))

    s1 = s1.map(maskEdge)

    # Functions to convert from/to dB
    def toNatural(img):
        return ee.Image(10.0).pow(img.select('..').divide(10.0)).copyProperties(img, ['system:time_start'])

    def toDB(img):
        return ee.Image(img).log10().multiply(10.0)

    s1 = s1.map(toNatural)

    # Load interesting S2 images
    s2 = ee.ImageCollection('COPERNICUS/S2'
        ).filterBounds(bevl_geom
        ).filterDate(start_date_str, end_date_str
        ).filter(ee.Filter.lessThan('CLOUDY_PIXEL_PERCENTAGE', 10));

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = ee.Number(2).pow(10).int();
    cirrusBitMask = ee.Number(2).pow(11).int();

    def maskS2clouds(image):
        qa = image.select('QA60');
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0));
        return image.updateMask(mask);

    s2 = s2.map(maskS2clouds);

    # Olha's idea to create weekly mean images
    step = 7 # in days
    def nextday(d):
        return ee.Date(start_date_str).advance(d, "day")

    days = ee.List.sequence(0, ee.Date(end_date_str).difference(ee.Date(start_date_str), 'day'), step
                           ).map(nextday)
    periods = days.slice(0,-1).zip(days.slice(1))

    # Function to get a string representation for a period
    def get_period_str(period):
        return ee.Date(ee.List(period).get(0)).format('YYYYMMdd')

    # Get the S1's for a period...
    def get_s1s_forperiod(period):
        return s1.filterDate(ee.List(period).get(0), ee.List(period).get(1)).select(['VV', 'VH'])

    # Get the S1's for a period...
    def get_s1_forperiod(period):
        period_str = get_period_str(period)
        return s1.filterDate(ee.List(period).get(0), ee.List(period).get(1)).mean().select(['VV', 'VH'], [ee.String('VV_').cat(period_str), ee.String('VH_').cat(period_str)])
    s1_perperiod = periods.map(get_s1_forperiod)

    # Get the S2's for a period...
    # Remark: median on an imagecollection apparently has a result that
    # the geometry of the image is not correct anymore... so median only is done afterwards.
    def get_s2s_forperiod(period):
        return s2.filterDate(ee.List(period).get(0), ee.List(period).get(1))

    # Stack all S1 bands for all periods in one image...
    def stack_bands(i1, i2):
        return ee.Image(i1).addBands(ee.Image(i2))
    s1_stack_allperiods = s1_perperiod.slice(1).iterate(stack_bands, s1_perperiod.get(0))

    # Get a list of all tasks in gee
    # These are the different states that exist:
    #     - READY    : the task is submitted to the gee server, but not yet running
    #     - RUNNING  : the task is running on gee
    #     - COMPLETED: the task is completely completed
    #     - FAILED   : an error occured while processing the task and the tas stopped without result
    def get_gee_tasklist():
        tasklist = ee.batch.data.getTaskList()
        logger.info('Number of tasks: ' + str(len(tasklist)))
        return tasklist

    # TODO: on locations where there is now a check if the task exists, it is not yet possible to ignore the completed
    # ones, which is probably needed as an option, definitely to make it easier to debug...
    def check_if_task_exists(task_description: str, task_state_list) -> str:
        """ Checks if a task exists already on gee """

        # If the tasks aren't retrieved yet, do so...
        global global_gee_tasks_cache
        if global_gee_tasks_cache is None:
            global_gee_tasks_cache = get_gee_tasklist()

        # Check if there is a task with this name and this state (there can be multiple tasks with this name!)
        for task in global_gee_tasks_cache:
            if (task['description'] == task_description
                    and task['task_type'] == 'EXPORT_FEATURES'
                    and task['state'] in task_state_list):
                logger.debug(f"<check_if_task_exists> Task {task_description} found with state {task['state']}")
                return True
        logger.debug(f"<check_if_task_exists> Task {task_description} doesn't exist with any of the states in {task_state_list}")
        return False

    # If the file doesn't exist yet... export the parcel with all interesting columns to csv...
    export_description = f'{base_filename}_pixcount'
    export_filename = export_description + '.csv'
    dest_fullpath = os.path.join(dest_data_dir, export_filename)
    dest_fullpath_todownload = os.path.join(dest_data_dir_todownload, export_filename)
    if ((not os.path.isfile(dest_fullpath))
            and (not os.path.isfile(dest_fullpath_todownload))
            and (not check_if_task_exists(export_description, ['RUNNING', 'READY' ''', 'COMPLETED' ''']))):
        logger.info(f"Prcinfo file doesn't exist yet... so create it: {dest_fullpath}")

        s1_for_count = s1.filterDate(ee.List(periods.get(0)).get(0), ee.List(periods.get(0)).get(1)).mean().select(['VV'], ['pixcount'])

        # Reduceregion uses the crs of the image to do the reduceregion, but
        # because there are images of two projections, it is WGS84
        bevl2017_pixcount = ee.Image(s1_for_count
            ).reduceRegions(collection = bevl2017
                           ,reducer = ee.Reducer.count()
                           ,scale=10)
        bevl2017_pixcount = bevl2017_pixcount.select([gs.id_column, 'count'], newProperties=[gs.id_column, 'pixcount'], retainGeometry=False)

        exportTask = ee.batch.Export.table.toDrive(
                collection = bevl2017_pixcount
                , folder = 'Monitoring'
                , description = export_description
                , fileFormat = 'CSV')
        ee.batch.Task.start(exportTask)
        pathlib.Path(dest_fullpath_todownload).touch()

    # Loop over all periods and export data per period to drive
    nb_periods = periods.length().getInfo()
    logger.info(f'Loop through all <{nb_periods}> periods')
    for i in range(0, nb_periods):

        # Calculate the start and end dates of this period...
        period_start_str = (start_date + timedelta(days=i*7)).strftime('%Y-%m-%d')
        period_end_str = (start_date + timedelta(days=(i+1)*7)).strftime('%Y-%m-%d')
        logger.debug(f"Process period: {period_start_str} till {period_end_str}")

        # Format relevant local filenames
        export_description = f'{base_filename}_{period_start_str}'
        export_filename = export_description + '.csv'
        dest_fullpath = os.path.join(dest_data_dir, export_filename)
        dest_fullpath_todownload = os.path.join(dest_data_dir_todownload, export_filename)

        # Get mean s1 image of the s1 images that are available in this period
        s1_forperiod = get_s1_forperiod([period_start_str, period_end_str])

        # If the data is already available locally... go to next period
        if (os.path.isfile(dest_fullpath)):
            logger.info(f"For period: {period_start_str}, file already available locally: SKIP: {export_filename}")
            continue

        # If the data is already "ordered" in a previous run and is still busy processing, don't start processing again
        if (os.path.isfile(dest_fullpath_todownload)
                and(check_if_task_exists(export_description, ['RUNNING', 'READY', 'COMPLETED']))):
            logger.info(f"For period: {period_start_str}, file still busy processing or is ready on gee: SKIP: {export_filename}")
            continue

        # Get s2 images that are available in this period
        s2s_forperiod = get_s2s_forperiod([period_start_str, period_end_str])

        # Determine export columns: the ID column of the parcel + the s1 bands for this period
        export_columns = ee.List([gs.id_column]).cat(ee.Image(s1_forperiod).bandNames())

        # If S2 available for entire flanders... add those bands as well
        s2_pct_bevl = bevl_geom.intersection(s2s_forperiod.geometry()).area().divide(bevl_geom.area())
        s2_forperiod = ee.Algorithms.If(s2_pct_bevl.gt(0.95), s2s_forperiod.median().select(['B2', 'B3', 'B4', 'B8'], [ee.String('S2B2_').cat(period_start_str), ee.String('S2B3_').cat(period_start_str), ee.String('S2B4_').cat(period_start_str), ee.String('S2B8_').cat(period_start_str)]), ee.Image())
        export_columns = ee.Algorithms.If(s2_pct_bevl.gt(0.95), export_columns.cat(ee.Image(s2_forperiod).bandNames()), export_columns);

        # Combine the mean and standard deviation reducers.
        reducers = ee.Reducer.mean().combine(reducer2 = ee.Reducer.stdDev(), sharedInputs = True)

        # Get the sentinel data for each parcel
        imagedata_forperiod_perparcel = ee.Image(s1_forperiod
                ).addBands(s2_forperiod
                ).reduceRegions(collection=bevl2017, reducer=reducers, scale=10)

        # Set the geometries to none, as we don't want to export them...
        def geom_to_none(f):
            return f.setGeometry(None)
        imagedata_forperiod_perparcel = imagedata_forperiod_perparcel.map(geom_to_none)

        # Export data to google drive
        logger.info(f"For period: {period_start_str}, start export of file: {export_filename}")
        exportTask = ee.batch.Export.table.toDrive(collection=imagedata_forperiod_perparcel
                                , folder='Monitoring'
                                , description=export_description
                                , fileFormat='CSV')
        ee.batch.Task.start(exportTask)

        # Create file in todownload folder to indicate this file should be downloaded
        pathlib.Path(dest_fullpath_todownload).touch()

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')