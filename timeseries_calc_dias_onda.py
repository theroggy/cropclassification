# -*- coding: utf-8 -*-
"""
@author: Pieter Roggemans
"""

from concurrent import futures
from datetime import datetime
import glob
import io
import logging
import math
import multiprocessing
import os
import shutil
import signal    # To catch CTRL-C
import sys
import time

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import windows
from rasterstats import zonal_stats

from helpers import log as log_helper

# General init
logger = logging.getLogger(__name__)

# Add handler for CTRL-C so it also works when parallel processes are running.
def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
#signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGINT, signal_handler)

def calc_stats(features_filepath: str,
               image_paths: str,
               output_dir: str,
               temp_dir: str,
               log_dir: str):
    """
    Calculate the statistics.
    """
    # Some checks on the input parameters
    if len(image_paths) == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    start_time = datetime.now()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create process pool for parallelisation...
    nb_parallel_images_max = 2
    nb_parallel = min(nb_parallel_images_max, len(image_paths))
    with futures.ProcessPoolExecutor(nb_parallel) as pool:
        
        # Loop over all images and process them in parallel...
        future_dict = {}
        start_time = datetime.now()
        nb_todo = len(image_paths)
        nb_done_total = 0
        for image_path in image_paths:

            # Create output filepath
            # Filename of the features
            features_filename = os.path.splitext(os.path.basename(features_filepath))[0] 
            # Filename of the image -> remove .zip if it is a zip file
            image_filename = os.path.basename(image_path)
            image_filename_noext, image_ext = os.path.splitext(image_filename)
            if image_ext.lower() == '.zip':
                image_filename = image_filename_noext
            output_filepath = os.path.join(output_dir, f"{features_filename}__{image_filename}.csv")

            # If the output files already exist, skip
            # TODO: temporary code, needs to be cleaned up!
            # TODO: probably needs to work with busy flags as well, as an output file can be partly ready...!
            output_VV_filepath = os.path.join(output_dir, f"{features_filename}__{image_filename}_VV_.csv")
            output_VH_filepath = os.path.join(output_dir, f"{features_filename}__{image_filename}_VH_.csv")

            if os.path.exists(output_VV_filepath) and os.path.exists(output_VH_filepath):
                logger.info(f"Output files exist already, so skip for {output_filepath}")
                nb_todo -= 1
                continue

            # Loop through all files to calc them... (Parallel via submit version)
            future = pool.submit(calc_stats_image, 
                                 features_filepath,
                                 image_path,
                                 output_filepath,
                                 temp_dir,
                                 log_dir)
            future_dict[future] = {"features_filepath": features_filepath,
                                   "image_path": image_path, 
                                   "future_start_time": datetime.now()}

        # Loop through the completed batches
        for future in futures.as_completed(future_dict):            
            # Get some info about the future that has now completed
            future_info = future_dict[future]
            start_time_latestbatch = future_info['future_start_time']

            # Try to extract the result
            try:
                data = future.result()           

                # Log the progress and prediction speed
                logger.info(f"Ready processing image: {future_info['image_path']}")
                nb_done_latestbatch = 1
                nb_done_total += nb_done_latestbatch
                progress_msg = get_progress_message(nb_todo, nb_done_total, nb_done_latestbatch, start_time, start_time_latestbatch)
                logger.info(progress_msg)

            except Exception as ex:
                logger.error(f"Exception getting result for {future_info['image_path']}: {ex}")
                raise
    
    logger.info(f"Time required for calculating data for {len(image_paths)} images, all bands: {(datetime.now()-start_time).total_seconds()} sec")

def get_progress_message(nb_todo: int, 
                         nb_done_total: int, 
                         nb_done_latestbatch: int, 
                         start_time: datetime, 
                         start_time_latestbatch: datetime) -> str:   
    """
    Returns a progress message based on the input.

    Args
        nb_todo: total number of items that need(ed) to be processed
        nb_done_total: total number of items that have been processed already
        nb_done_latestbatch: number of items that were processed in the latest batch
        start_time: datetime the processing started
        start_time_latestbatch: datetime the latest batch started
    """
    time_passed_s = (datetime.now()-start_time).total_seconds()
    time_passed_latestbatch_s = (datetime.now()-start_time_latestbatch).total_seconds()

    # Calculate the overall progress
    large_number = 9999999999
    if time_passed_s > 0:
        nb_per_hour = (nb_done_total/time_passed_s) * 3600
    else:
        nb_per_hour = large_number
    hours_to_go = (int)((nb_todo-nb_done_total)/nb_per_hour)
    min_to_go = (int)((((nb_todo-nb_done_total)/nb_per_hour)%1)*60)

    # Calculate the speed of the latest batch
    if time_passed_latestbatch_s > 0: 
        nb_per_hour_latestbatch = (nb_done_latestbatch/time_passed_latestbatch_s) * 3600
    else:
        nb_per_hour_latestbatch = large_number
    
    # Return formatted message
    message = f"{hours_to_go}:{min_to_go} left for {nb_todo-nb_done_total} todo at {nb_per_hour:0.0f}/h ({nb_per_hour_latestbatch:0.0f}/h last batch)"
    return message
    
def calc_stats_image(features_filepath,
                     image_path: str,
                     output_filepath: str,
                     temp_dir: str,
                     log_dir: str) -> bool:
    """
    Calculate the statistics for one image.

    Returns True if succesfully completed.
    """
    # When running in parallel processes, the logging needs to be initialised again
    '''
    global logger
    if logger is None:
        if log_dir is None:
            log_dir = 'log'
        logger = log_helper.main_log_init(log_dir, f"{__name__}_{os.getpid()}", print=False)
    '''

    # set up logging to file - see previous section for more details
    logger = logging.getLogger('calc_stats_image')
    logger.propagate = False
    log_filepath = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calc_stats_image_{os.getpid()}.log")
    fh = logging.FileHandler(filename=log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s'))
    logger.addHandler(fh)

    start_time = datetime.now()

    # If output file(s) already exists, remove it
    # TODO: doesn't take bands into acount yet...
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # Preprocess image + get basic info about it
    image_prepr_path = preprocess_image(image_path, temp_dir)
    logger.info(f"Preprocessing ready, result: {image_prepr_path}")
    image_bounds, image_affine, image_epsg = get_image_info(image_prepr_path)

    features_gdf = load_features_file(features_filepath=features_filepath, 
                                      target_epsg=image_epsg,
                                      columns_to_retain=["CODE_OBJ", "geometry"],
                                      bbox=image_bounds)

    # Check if features were found, otherwise no use to proceed
    nb_todo = len(features_gdf)
    if nb_todo == 0:
        logger.info(f"No features were found in the bounding box of the image, so return: {image_path}")
        return True

    # Prepare some variables for the further processing
    nb_parallel_max = multiprocessing.cpu_count()
    
    # Calculate the number per batch, but keep the number between 100 and 50000...
    nb_per_batch = min(max(math.ceil(nb_todo/nb_parallel_max), 100), 50000)

    # Clustering the batches geographically would result in having to load smaller pieces of the image later on...
    # maybe sort them on X coordinate eg.?
    features_gdf_batches = [features_gdf.loc[i:i+nb_per_batch-1,:] for i in range(0, nb_todo, nb_per_batch)]
    nb_batches = len(features_gdf_batches)
    nb_parallel = min(nb_parallel_max, nb_batches)

    # Create process pool for parallelisation...
    with futures.ProcessPoolExecutor(nb_parallel) as pool:

        # Now loop through all batches to execute them... (Parallel via submit version)
        start_time_loop = datetime.now()
        future_dict = {}
        nb_processed = 0        
        for features_gdf_batch in features_gdf_batches:

            start_time_batch = datetime.now()
            future = pool.submit(calc_stats_image_gdf, 
                                 features_gdf_batch,
                                 image_prepr_path,
                                 output_filepath,
                                 log_dir,
                                 start_time_batch)
            future_dict[future] = {"image_path": image_path,
                                   "image_prepr_path": image_prepr_path, 
                                   "start_time_batch": start_time_batch,
                                   "nb_items_batch": len(features_gdf_batch)}

        # Loop through the completed batches
        for future in futures.as_completed(future_dict):            
            # Get some info about the future that has now completed
            future_info = future_dict[future]
            start_time_currbatch = future_info['start_time_batch']
            nb_processed_currbatch = future_info['nb_items_batch']

            # Try to extract the result
            try:
                data = future.result()
                logger.debug(f"Ready processing batch of {nb_processed_currbatch} for features image: {future_info['image_path']}")
            except Exception as ex:
                logger.error(f"Exception getting result for {future_info['image_path']}: {ex}")
                raise

    # TODO: write something so you know the image has been treated?
    logger.info(f"Time required for calculating data for {nb_todo} features, all bands: {datetime.now()-start_time} s of {image_path}")
    
    # If the preprocessing created a temp new file, clean it up again...
    if image_prepr_path != image_path:
        logger.info(f"Remove local temp image copy: {image_prepr_path}")
        if os.path.isdir(image_prepr_path):           
            shutil.rmtree(image_prepr_path, ignore_errors=True)
        else:
            os.remove(image_prepr_path)

    return True

def load_features_file(features_filepath: str,
                       columns_to_retain: [],
                       target_epsg: int,
                       bbox = None) -> gpd.GeoDataFrame:
    """
    Load the features and reproject to the target crs.

    Remarks:
        * Reprojected version is "cached" on disk so on a next call, it can be directly read.
        * Locking and waiting is used to ensure correct results even if used in a parallel way.

    Args
        features_filepath: 
        columns_to_retain: 
        target_srs: 
        bbox: bounds of the area to be loaded, in the target_epsg
    """
    # Load parcel input file and preprocess it: remove excess columns + reproject if needed.
    # By convention, the features filename should end on the projection... so extract epsg from filename
    features_filepath_noext, ext = os.path.splitext(features_filepath)
    if features_filepath_noext.find('_') != -1:
        splitted = features_filepath_noext.split('_')
        features_epsg = splitted[len(splitted)-1]

    # Determine the correct filename for the input features in the correct projection.
    if features_epsg != target_epsg:
        features_prepr_filepath = f"{features_filepath_noext}_{target_epsg}{ext}"
    else:
        features_prepr_filepath = features_filepath

    # Prepare filename for a "busy file" to ensure proper behaviour in a parallel processing context
    features_prepr_filepath_busy = f"{features_prepr_filepath}_busy"

    # If the file doesn't exist yet in right projection, read original input file to reproject/write to new file with correct epsg
    features_gdf = None
    if not (os.path.exists(features_prepr_filepath_busy) 
        or os.path.exists(features_prepr_filepath)):
        
        # Create lock file in an atomic way, so we are sure we are the only process working on it. 
        # If function returns true, there isn't any other thread/process already working on it
        if create_lock_file(features_prepr_filepath_busy):

            try:

                # Read (all) original features + remove unnecessary columns...
                logger.info(f"Read original file {features_filepath}")
                start_time = datetime.now()
                features_gdf = gpd.read_file(features_filepath)
                logger.info(f"Read ready, found {len(features_gdf)} features, crs: {features_gdf.crs}, took {(datetime.now()-start_time).total_seconds()} s")
                for column in features_gdf.columns:
                    if column not in columns_to_retain and column not in ['geometry', 'x_ref']:
                        features_gdf.drop(columns=column, inplace=True)

                # Reproject them
                logger.info(f"Reproject features from {features_gdf.crs} to epsg {target_epsg}")
                features_gdf = features_gdf.to_crs(epsg=target_epsg)
                logger.info("Reprojected, now sort on x_ref")
                
                # Order features on x coordinate
                if 'x_ref' not in features_gdf.columns:
                    features_gdf['x_ref'] = features_gdf.geometry.bounds.minx
                features_gdf.sort_values(by=['x_ref'], inplace=True)
                features_gdf.reset_index(inplace=True)

                # Cache the file for future use
                logger.info(f"Write {len(features_gdf)} reprojected features to {features_prepr_filepath}")
                features_gdf.to_file(features_prepr_filepath)
                logger.info(f"Reprojected features written")

                """ 
                # TODO: redo performance test of using pkl as cache file
                features_gdf.to_pickle('file.pkl')
                logger.info(f"Pickled")
                df2 = pd.read_pickle('file.pkl')
                logger.info(f"Pickle read")
                """

            finally:    
                # Remove lock file as everything is ready for other processes to use it...
                os.remove(features_prepr_filepath_busy)

            # Now filter the parcels that are in bbox provided
            if bbox is not None:
                logger.info(f"bbox provided, so filter features in the bbox of {bbox}")
                xmin, ymin, xmax, ymax = bbox
                features_gdf = features_gdf.cx[xmin:xmax, ymin:ymax]
                logger.info(f"Found {len(features_gdf)} features in bbox")

                """ Slower + crashes?
                logger.info(f"Filter only features in the bbox of image with spatial index {image_bounds}")
                spatial_index = features_gdf.sindex
                possible_matches_index = list(spatial_index.intersection(image_bounds))
                features_gdf = features_gdf.iloc[possible_matches_index]
                #features_gdf = features_gdf[features_gdf.intersects(image_shape)]
                logger.info(f"Found {len(features_gdf)} features in the bbox of image {image_bounds}")                    
                """

    # If there exists already a file with the features in the right projection, we can just read the data
    if features_gdf is None:

        # If a "busy file" still exists, the file isn't ready yet, but another process is working on it, so wait till it disappears
        while os.path.exists(features_prepr_filepath_busy):
            time.sleep(1)

        logger.info(f"Read {features_prepr_filepath}")
        start_time = datetime.now()
        features_gdf = gpd.read_file(features_prepr_filepath, bbox=bbox)
        logger.info(f"Read ready, found {len(features_gdf)} features, crs: {features_gdf.crs}, took {(datetime.now()-start_time).total_seconds()} s")

        # Order features on x_ref to (probably) have more clustering of features in further action... 
        if 'x_ref' not in features_gdf.columns:
            features_gdf['x_ref'] = features_gdf.geometry.bounds.minx
        features_gdf.sort_values(by=['x_ref'], inplace=True)
        features_gdf.reset_index(inplace=True)

        # To be sure, remove the columns anyway...
        for column in features_gdf.columns:
            if column not in columns_to_retain and column not in ['geometry']:
                features_gdf.drop(columns=column, inplace=True)

    # Ready, so return result...
    return features_gdf

def calc_stats_image_gdf(features_gdf,
                         image_path: str,
                         output_filepath: str,
                         log_dir: str,
                         future_start_time = None) -> bool:
    """

    Returns True if succesfully completed.
    """

    # TODO: the differenct bands should be possible to process in parallel as well... so this function should process only one band!
    # When running in parallel processes, the logging needs to be initialised again
    '''
    global logger
    if logger is None:
        logger = log_helper.main_log_init(log_dir, f"{__name__}_{os.getpid()}", False)
        #logger.setLevel(logging.ERROR)
    '''
    logger = logging.getLogger('calc_stats_image')
    logger.propagate = False
    log_filepath = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calc_stats_image_{os.getpid()}.log")
    fh = logging.FileHandler(filename=log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s'))
    logger.addHandler(fh)

    # Log the time betwoon scheduling the future and acually run...
    if future_start_time is not None:
        logger.info(f"Start, {(datetime.now()-future_start_time).total_seconds()} after future was scheduled")
 
    # TODO: probably reading the image should be moved to a function...
    # First determine image format
    image_path_noext, image_ext = os.path.splitext(image_path)
    if image_ext.upper() == '.CARD':
        # First compose image band datafile name, than read the metadata
        # TODO: probably cleaner/easier to read from metadata file...
        image_data = {}             # Dict for the transforms and the data per band
        image_dir, image_basename_noext = os.path.split(image_path_noext)
        image_card_datadir = os.path.join(image_path, f"{image_basename_noext}.data")
        image_band_filepath = os.path.join(image_card_datadir, "Gamma0_VV.img")
        logger.info(f"Read image data from {image_band_filepath}")
        with rasterio.open(image_band_filepath) as src:
            # Determine the window we need to read from the image:
            window_to_read = projected_bounds_to_window(features_gdf.total_bounds, 
                                                        src.transform, src.width, src.height) 
            window_to_read_transform = rasterio.windows.transform(window_to_read, src.transform)
            # Read!
            image_data['VV'] = src.read(1, window=window_to_read) 
        image_band_filepath = os.path.join(image_card_datadir, "Gamma0_VH.img")
        logger.info(f"Read image data from {image_band_filepath}")
        with rasterio.open(image_band_filepath) as src:
            image_data['VH'] = src.read(1, window=window_to_read) 
        logger.info(f"Image data read")
    else:
        message = f"Only .CARD image format currently supported!, not {image_path}"
        logger.error(message)
        raise NotImplementedError(message)

    # Loop over image bands
    image_id = os.path.basename(image_path)
    for band in image_data:

        # Calc zonal stats
        logger.info(f"Calculate statistics for band {band}")
        # TODO: probably some kind of masking is still necessary at the edges of the images!!!
        features_stats = zonal_stats(features_gdf, image_data[band], 
                                     affine=window_to_read_transform, prefix="", nodata=0, geojson_out=True,
                                     stats=["count", "mean", "std", "min", "max", "percentile_25", "percentile_50", "percentile_75"])
        logger.info(f"Statistics calculated... ")
        #features_stats_df = pd.DataFrame(features_stats)
        features_stats_gdf = gpd.GeoDataFrame.from_features(features_stats)
        
        logger.info(f"Dataframe created from stats... ")

        #df = pd.DataFrame.from_dict(df.properties.to_dict(), orient='index')
        features_stats_gdf['image_id'] = image_id
        features_stats_gdf['band'] = band
        
        logger.info(f"Rename columns... ")
        features_stats_gdf.rename(index=str, columns={"percentile_25": "p25", "percentile_50": "p50","percentile_75": "p75"}, inplace=True)
        
        # df is the dataframe
        if len(features_stats_gdf) > 0:

            # Remove rows with empty data
            features_stats_gdf.dropna(inplace=True)
            if len(features_stats_gdf.values) > 0:
                output_filepath_noext, output_filepath_ext = os.path.splitext(output_filepath)
                output_band_filepath = f"{output_filepath_noext}_{band}_{output_filepath_ext}"
                logger.info(f"Write data for {len(features_stats_gdf.values)} parcels found to {output_band_filepath}")
                features_stats_df = features_stats_gdf.drop(columns='geometry')

                # If file exists already... append
                # TODO: This isn't full-proof in a parallel context!!!
                if not os.path.exists(output_band_filepath):
                    features_stats_df.to_csv(output_band_filepath, index=False, sep=',')
                else:
                    features_stats_df.to_csv(output_band_filepath, index=False, mode='a', header=False, sep=',')
            else:
                logger.info(f"No data found")
    
    return True

def projected_bounds_to_window(projected_bounds,
                               image_transform, 
                               image_pixel_width: int,
                               image_pixel_height: int):
    # Take bounds of the features + convert to image pixels  
    xmin, ymin, xmax, ymax = projected_bounds
    window_to_read_raw = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, image_transform)
    
    # Round so it only increases window size
    window_to_read = window_to_read_raw.round_offsets('floor').round_lengths('ceil')
    
    # Now some general math on window properties, but as they are readonly properties, work on copy
    col_off, row_off, width, height = window_to_read.flatten()
    # Add buffer of 1 pixel extra around
    col_off -= 1
    row_off -= 1
    width += 2
    height += 2
    
    # Make sure the offsets aren't negative, as the pixels that are 'read' there acually  
    # get some value instead of eg. nan...!
    if col_off < 0:
        width -= abs(col_off)
        col_off = 0
    if row_off < 0:
        height -= abs(row_off)
        row_off = 0
    
    # Make sure there won't be extra pixels to the top and right that will be read 
    if (col_off + width) > image_pixel_width:
        width = image_pixel_width - col_off
    if (row_off + height) > image_pixel_height:
        height = image_pixel_height - row_off
    
    # Ready... prepare to return...
    window_to_read = rasterio.windows.Window(col_off, row_off, width, height)
    logger.debug(f"window_to_read_raw: {window_to_read_raw}, window_to_read: {window_to_read}, image_pixel_width: {image_pixel_width}, image_pixel_height: {image_pixel_height}, file transform: {image_transform}")
    return window_to_read

def create_lock_file(filename):
    """
    Create a lock file in an atomic way, so it is threadsafe.

    Returns True if the file was created by this thread, False if the file existed already.
    """
    try:
        os.open(filename,  os.O_CREAT | os.O_EXCL)
        return True
    except FileExistsError:
        return False

def preprocess_image(image_path: str,
                     temp_dir: str) -> str:
    """
    Preprocesses the input image. 

    The preprocessing consist of unzipping the file to the temp dir specified.

    Returns the path to the proprocessed file/directory.
    """

    # If the input path is not a zip file, don't make local copy and just return image path
    image_path_ext = os.path.splitext(image_path)[1]
    if image_path_ext.lower() != ".zip":
        return image_path
    else:    
        # It is a zip file, so it needs to be unzipped first...
        # Create destination file path
        image_basename_withzipext = os.path.basename(image_path)
        image_basename = os.path.splitext(image_basename_withzipext)[0]
        image_unzipped_path = os.path.join(temp_dir, image_basename)

        image_unzipped_path_busy = f"{image_unzipped_path}_busy"
        # If the input is a zip file, unzip file to temp local location if it doesn't exist yet
        # If the file doesn't exist yet in right projection, read original input file to reproject/write to new file with correct epsg
        if not (os.path.exists(image_unzipped_path_busy) 
            or os.path.exists(image_unzipped_path)):

            # Create temp dir if it doesn't exist yet
            os.makedirs(temp_dir, exist_ok=True)

            # Create lock file in an atomic way, so we are sure we are the only process working on it. 
            # If function returns true, there isn't any other thread/process already working on it
            if create_lock_file(image_unzipped_path_busy):

                try:
                    logger.info(f"Unzip image {image_path} to local location {temp_dir}")

                    # Create the dest dir where the file will be unzipped to + unzip!
                    if not os.path.exists(image_unzipped_path):
                        import zipfile
                        with zipfile.ZipFile(image_path,"r") as zippedfile:
                            zippedfile.extractall(temp_dir)
                finally:    
                    # Remove lock file when we are ready
                    os.remove(image_unzipped_path_busy)

        # If a "busy file" still exists, the file isn't ready yet, but another process is working on it, so wait till it disappears
        while os.path.exists(image_unzipped_path_busy):
            time.sleep(1)
        
        # Now we are ready to return the path...
        return image_unzipped_path

def get_image_info(image_path):

    # Get image projection
    # First determine image format
    image_path_noext, image_ext = os.path.splitext(image_path)
    if image_ext.upper() == '.CARD':
        # First compose image band datafile name, than read the metadata
        # TODO: probably cleaner/easier to read from metadata file...
        image_dir, image_basename_noext = os.path.split(image_path_noext)
        image_card_datadir = os.path.join(image_path, f"{image_basename_noext}.data")
        image_band_filepath = os.path.join(image_card_datadir, "Gamma0_VV.img")
        logger.debug(f"Read image metadata from {image_band_filepath}")
        with rasterio.open(image_band_filepath) as src:
            image_bounds = src.bounds
            image_affine = src.transform
            image_crs = str(src.crs)
            image_epsg = image_crs.upper().replace('EPSG:', '')
        logger.debug(f"Image metadata read: crs: {image_crs}")
    else:
        message = f"Only .CARD image format currently supported!, not {image_path}"
        logger.error(message)
        raise NotImplementedError(message)
    
    return (image_bounds, image_affine, image_epsg)

def main():

    # First define/init some general variables/constants
    year = 2018
    country_code = 'BEFL'        # The region of the classification: typically country code

    #base_dir = "X:\\Monitoring\\Markers\\PlayGround\\PIEROG"                                # Base dir
    base_dir = f"{os.path.expanduser('~')}{os.sep}markers{os.sep}cropclassification"
    input_dir = os.path.join(base_dir, 'inputdata')                                         # Input dir
    input_preprocessed_dir = os.path.join(input_dir, 'preprocessed')

    global logger
    base_log_dir = os.path.join(base_dir, 'log')
    log_dir = os.path.join(base_dir, f"{datetime.now():%Y-%m-%d_%H-%M-%S}")
    logger = log_helper.main_log_init(log_dir, __name__)

    # Input features file depends on the year
    if year == 2017:
        input_features_filename_noext = 'Prc_flanders_2017_2018-01-09'                        # Input filename
    elif year == 2018:
        input_features_filename_noext = 'Prc_BEFL_2018_2018-08-02'                            # Input filename
    input_features_filepath = os.path.join(input_dir, f"{input_features_filename_noext}.shp")   # Input filepath of the parcel

    # Input image files
    imagedata_dir = os.path.join(base_dir, 'Timeseries_data')      # General data  dir
    start_date_str = f"{year}-03-27"
    end_date_str = f"{year}-08-10"                                 # End date is NOT inclusive for gee processing

    input_image_dir = "X:\\GIS\\GIS DATA\\Ortho_Belgie\\Sentinel2\\S2_NDVI_Periodiek"
    #input_image_searchstr = f"{input_image_dir}{os.sep}*.tif"
    input_image_searchstr = "/mnt/NAS3/CARD/FLANDERS/S1*/L1TC/2017/*/*/*.CARD"
    input_image_filepaths = glob.glob(input_image_searchstr)
    logger.info(f"Found {len(input_image_filepaths)} images to process")

    # Output dir
    output_dir = os.path.join(base_dir, "output")

    # Prepare temp dir path and clean contents from it.
    temp_dir = os.path.join(base_dir, "tmp")
    logger.info(f"Clean the temp dir")
    if os.path.exists(temp_dir):
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(temp_dir + os.sep)
    
    """
    # TEST CODE !!!!!!!!!!!!!!!!
    # For testing purposes, so extra cleanup...
    # Take only the x first images found while testing
    #input_image_filepaths = input_image_filepaths[:1]
    logger.info(f"As we are only testing, process only {len(input_image_filepaths)} first images")

    # Clean log dir 
    logger.info(f"As we are only testing, delete the contents of the log dir")
    if os.path.exists(base_log_dir):
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(base_log_dir + os.sep)

    # Delete output dir 
    logger.info(f"As we are only testing, delete the output dir")
    if os.path.exists(output_dir):
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(output_dir + os.sep)
    """    
    # END TEST CODE !!!!!!!!!!!!!!!!
    
    # Start calculation
    logger.info(f"Start processing")
    calc_stats(features_filepath=input_features_filepath,
               image_paths=input_image_filepaths,
               output_dir=output_dir,
               temp_dir=temp_dir,
               log_dir=log_dir)
    
if __name__ == '__main__':
    main()
