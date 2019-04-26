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
import signal    # To catch CTRL-C explicitly
import sys
import time

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import windows
from rasterstats import zonal_stats
from shapely.geometry import polygon as sh_polygon
import xml.etree.ElementTree as ET

from helpers import log as log_helper

# General init
logger = logging.getLogger(__name__)

# Add handler for CTRL-C so it also works when parallel processes are running.
def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
#signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGINT, signal_handler)

'''
def calc_stats_old(features_filepath: str,
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
    nb_parallel_max = multiprocessing.cpu_count()
    nb_parallel = min(nb_parallel_max, len(image_paths))

    # Loop over all images and process them in parallel...
    future_dict = {}
    start_time = datetime.now()
    nb_todo = len(image_paths)
    nb_done_total = 0
    with futures.ProcessPoolExecutor(nb_parallel) as pool:
        
        for i, image_path in enumerate(image_paths):

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
            # Calculate how much paralellisation to do in next level 
            nb_parallel_calc_stats_image = max(nb_parallel_max-(nb_todo-(i+1)), 1)
            future = pool.submit(calc_stats_image, 
                                 features_filepath,
                                 image_path,
                                 output_filepath,
                                 temp_dir,
                                 log_dir,
                                 nb_parallel_calc_stats_image)
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
    
    logger.info(f"Time taken to calculate data for {nb_todo} images: {(datetime.now()-start_time).total_seconds()} sec")
'''
def calc_stats(features_filepath: str,
               image_paths: str,
               output_dir: str,
               temp_dir: str,
               log_dir: str,
               force: bool = False):
    """
    Calculate the statistics.
    """
    # Some checks on the input parameters
    if len(image_paths) == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    image_paths.sort()
    start_time = datetime.now()
    nb_todo = len(image_paths)

    # Create process pool for parallelisation...
    nb_parallel_max = multiprocessing.cpu_count()
    nb_parallel = nb_parallel_max
    with futures.ProcessPoolExecutor(nb_parallel) as pool:

        # Loop over all images to start the data preparation for each of them in parallel...
        image_dict = {}
        calc_stats_batch_dict = {}
        nb_done_total = 0
        curr_image_path_index = 0
        while True:

            # If not all images aren't processed/processing yet
            if curr_image_path_index < len(image_paths):    

                # Create output filepath
                image_path = image_paths[curr_image_path_index]
                curr_image_path_index += 1
                image_info = get_image_info(image_path)
                output_filepath = get_output_filepath(features_filepath, image_path, output_dir,
                                                      image_info['orbit_properties_pass'])

                # Check if the features-image combination has been processed already
                # TODO: if force is true, remove the output files!
                output_done_filepath = f"{output_filepath}_DONE"
                if(force == False
                   and os.path.exists(output_done_filepath)):
                    logger.info(f"DONE file exist already, so skip {output_done_filepath}")
                    nb_todo -= 1
                    continue

                # Always make sure there are nb_parallel_max prepare_calc's active. 
                nb_busy = 0
                for image_path_tmp in image_dict:
                    if image_dict[image_path_tmp]['status'] == 'IMAGE_PREPARE_CALC_BUSY':
                        nb_busy += 1
                if nb_busy < nb_parallel_max:
                    logger.debug(f"nb_busy: {nb_busy}, nb_parallel_max: {nb_parallel_max}, so nb_busy < nb_parallel_max")          
                    # Start the prepare processing assync
                    future = pool.submit(prepare_calc, 
                                         features_filepath,
                                         image_path,
                                         output_filepath,
                                         temp_dir,
                                         log_dir,
                                         nb_parallel_max)
                    image_dict[image_path] = {'prepare_calc_future': future, 
                                              'prepare_calc_starttime': datetime.now(),
                                              'output_filepath': output_filepath,
                                              'output_done_filepath': output_done_filepath,
                                              'status': 'IMAGE_PREPARE_CALC_BUSY'}
                    
                    # Jump to next image to start the prepare_calc for it...
                    continue

            # Loop through the images to find which are ready... to start the real calculations...
            for image_path in image_dict:
                
                # Get the info that is available in the dict with futures for the one that has now completed
                image_info = image_dict[image_path]
                    
                # If the status isn't PREPARE_CALC_BUSY or if the prepare_calc is still running, go to next...
                if(image_info['status'] != 'IMAGE_PREPARE_CALC_BUSY' 
                   or image_info['prepare_calc_future'].running()):
                    continue

                # Extract the result from the preparation
                try:
                    # Get the result from the completed  prepare_inputs
                    prepare_calc_result = image_info['prepare_calc_future'].result()

                    # If nb_features to be treated is 0... set done file and continue with next...
                    if prepare_calc_result['nb_features_to_calc_total'] == 0:
                        logger.info(f"No features found overlapping image, just create done file: {image_info['output_done_filepath']}")
                        create_file_atomic(image_info['output_done_filepath'])
                        image_info['status'] = 'IMAGE_CALC_DONE'
                        continue

                    # Add info about the result of the prepare_calc to the image info...
                    image_info['image_prepared_path'] = prepare_calc_result['image_prepared_path']
                    image_info['feature_batches'] = prepare_calc_result['feature_batches']
                    image_info['nb_features_to_calc_total'] = prepare_calc_result['nb_features_to_calc_total']
                    image_info['temp_features_dir'] = prepare_calc_result['temp_features_dir']

                    # Set status to calc_is_busy so we know calculation is busy...
                    image_info['status'] = 'IMAGE_CALC_BUSY'
                    image_info['calc_starttime'] = datetime.now()

                    # Now loop through all prepared feature batches to start the statistics calculation for each  
                    for features_batch in image_info['feature_batches']:
                        start_time_batch = datetime.now()
                        future = pool.submit(calc_stats_image_gdf, 
                                             features_batch['filepath'],
                                             image_info['image_prepared_path'],
                                             image_info['output_filepath'],
                                             log_dir,
                                             start_time_batch)
                        calc_stats_batch_dict[features_batch['filepath']] = {'calc_stats_future': future,
                                                                       'image_path': image_path,
                                                                       'image_prepared_path': image_info['image_prepared_path'], 
                                                                       'start_time_batch': start_time_batch,
                                                                       'nb_items_batch': features_batch['nb_items'],
                                                                       'status': 'BATCH_CALC_BUSY'}

                except Exception as ex:
                    message = f"Exception getting result for {image_path}: {ex}"
                    logger.error(message)
                    raise Exception(message)

            # Loop through the completed calculations
            for calc_stats_batch_id in calc_stats_batch_dict:
                # Get some info about the future that has now completed
                calc_stats_batch_info = calc_stats_batch_dict[calc_stats_batch_id]
                
                # If not processed yet, but it is done, get the results
                if(calc_stats_batch_info['status'] == 'BATCH_CALC_BUSY' 
                   and calc_stats_batch_info['calc_stats_future'].done() is True):

                    try:
                        # Get the result
                        result = calc_stats_batch_info['calc_stats_future'].result()
                        if not result:
                            raise Exception("Returned False?")
                        
                        # Set the processed flag to True
                        calc_stats_batch_info['status'] = 'BATCH_CALC_DONE'
                        logger.debug(f"Ready processing batch of {calc_stats_batch_info['nb_items_batch']} for features image: {calc_stats_batch_info['image_path']}")

                    except Exception as ex:
                        logger.error(f"Exception getting result for {calc_stats_batch_info['image_path']}: {ex}")
                        raise

            # Loop over all image_paths that are busy being calculated to check if there are still calc stats batches busy
            for image_path in image_dict:

                # If the image is busy being calculated...
                image_info = image_dict[image_path]
                if image_info['status'] == 'IMAGE_CALC_BUSY':

                    # Check if there are still batches busy for this image...
                    batches_busy = False
                    for calc_stats_batch_id in calc_stats_batch_dict:
                        if(calc_stats_batch_dict[calc_stats_batch_id]['image_path'] == image_path 
                           and calc_stats_batch_dict[calc_stats_batch_id]['status'] == 'BATCH_CALC_BUSY'):
                            batches_busy = True
                            break
                    
                    # If no batches are busy anymore for the file_path... cleanup.
                    if batches_busy == False:
                        # If all batches are done, the image is done...
                        image_info['status'] = 'IMAGE_CALC_DONE'

                        # Create the DONE file to indicate this features-image combination has been processed
                        # TODO: these paths (output_filepath, image_prepared_path,...) should be searched for again, or saved in a dict or???
                        create_file_atomic(image_info['output_done_filepath'])
                        logger.info(f"Time required for calculating data for {image_info['nb_features_to_calc_total']} features, all bands: {datetime.now()-start_time} s of {image_path}")
                
                        # If the preprocessing created a temp image file, clean it up...
                        image_prepared_path = image_info['image_prepared_path']
                        if image_prepared_path != image_path:
                            logger.info(f"Remove local temp image copy: {image_prepared_path}")
                            if os.path.isdir(image_prepared_path):           
                                shutil.rmtree(image_prepared_path, ignore_errors=True)
                            else:
                                os.remove(image_prepared_path)

                        # If the preprocessing created temp pickle files with features, clean them up...
                        shutil.rmtree(image_info['temp_features_dir'], ignore_errors=True)
                            
                        # Log the progress and prediction speed
                        logger.info(f"Ready processing image: {image_path}")
                        nb_done_latestbatch = 1
                        nb_done_total += nb_done_latestbatch
                        progress_msg = get_progress_message(nb_todo, nb_done_total, nb_done_latestbatch, start_time, image_info['calc_starttime'])
                        logger.info(progress_msg)

            # Loop over all images to check if they are all done
            all_done = True
            for image_path in image_paths:
                if(image_path not in image_dict
                   or image_dict[image_path]['status'] != 'IMAGE_CALC_DONE'):
                    all_done = False
                    break                   

            # If no processing is needed, or if all processing is ready, stop the never ending loop...
            if len(image_dict) == 0 or all_done is True:
                break 
            else:
                # Sleep before starting next iteration...
                time.sleep(1)   

        logger.info(f"Time taken to calculate data for {nb_todo} images: {(datetime.now()-start_time).total_seconds()} sec")

def get_output_filepath(features_filepath: str, 
                        image_path: str,
                        output_dir: str,
                        orbit_properties_pass: str):
    """
    Prepare the output filepath.
    """
    # Filename of the features
    features_filename = os.path.splitext(os.path.basename(features_filepath))[0] 
    # Filename of the image -> remove .zip if it is a zip file
    image_filename = os.path.basename(image_path)
    image_filename_noext, image_ext = os.path.splitext(image_filename)
    if image_ext.lower() == '.zip':
        image_filename = image_filename_noext

    # Interprete the orbit...
    if orbit_properties_pass == 'ASCENDING':
        orbit = 'ASC'
    elif orbit_properties_pass == 'DESCENDING':
        orbit = 'DESC'
    else:
        message = f"Unknown orbit_properties_pass: {orbit_properties_pass}"
        logger.error(message)
        raise Exception(message)

    output_filepath = os.path.join(output_dir, f"{features_filename}__{image_filename}_{orbit}.csv")
    return output_filepath

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
    
def prepare_calc(features_filepath,
                 image_path: str,
                 output_filepath: str,
                 temp_dir: str,
                 log_dir: str,
                 nb_parallel_max: int = 16) -> bool:
    """
    Prepare the inputs for a calculation.

    Returns True if succesfully completed.
    Remark: easiest it returns something, when used in a parallel way: concurrent.futures likes it better if something is returned
    """
    # When running in parallel processes, the logging needs to be write to seperate files + no console logging
    global logger
    logger = logging.getLogger('prepare_calc')
    logger.propagate = False
    log_filepath = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d_%H-%M-%S}_prepare_calc_{os.getpid()}.log")
    fh = logging.FileHandler(filename=log_filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s'))
    logger.addHandler(fh)

    ret_val = {}

    # Prepare the image
    image_prepared_path = prepare_image(image_path, temp_dir)
    logger.debug(f"Preparing ready, result: {image_prepared_path}")
    ret_val['image_prepared_path'] = image_prepared_path

    # Get info about the image
    image_info = get_image_info(image_prepared_path)
    logger.info(f"image_info: {image_info}")

    # Load the features that overlap with the image.
    # TODO: passing both bbox and poly is double, or not? 
    # footprint epsg should be passed as well, or reproject here first?
    logger.info(f"poly: {image_info['footprint']['shape']}")
    features_gdf = load_features_file(features_filepath=features_filepath, 
                                      target_epsg=image_info['image_epsg'],
                                      columns_to_retain=["CODE_OBJ", "geometry"],
                                      bbox=image_info['image_bounds'],
                                      polygon=image_info["footprint"]["shape"])

    # Check if overlapping features were found, otherwise no use to proceed
    nb_todo = len(features_gdf)
    ret_val['nb_features_to_calc_total'] = nb_todo
    if nb_todo == 0:
        logger.info(f"No features were found in the bounding box of the image, so return: {image_path}")
        return ret_val
    
    # Calculate the number per batch, but keep the number between 100 and 50000...
    nb_per_batch = min(max(math.ceil(nb_todo/nb_parallel_max), 100), 50000)

    # The features were already sorted on x coordinate, so the features in the batches are 
    # already clustered geographically 
    features_gdf_batches = [features_gdf.loc[i:i+nb_per_batch-1,:] for i in range(0, nb_todo, nb_per_batch)]

    # Pickle the batches to temporary files
    # Create temp dir to put the pickles in... and clean or create it.
    # TODO: change dir so it is always unique
    temp_features_dirname = os.path.splitext(os.path.basename(image_path))[0]
    temp_features_dir = os.path.join(temp_dir, temp_features_dirname)
    ret_val['temp_features_dir'] = temp_features_dir
    if os.path.exists(temp_features_dir):
        logger.info(f"Remove dir {temp_features_dir + os.sep}")
        shutil.rmtree(temp_features_dir + os.sep)
    else:
        os.makedirs(temp_features_dir, exist_ok=True)

    '''
    if os.path.exists(temp_features_dir):
        logger.info(f"Temp dir exists: {temp_features_dir}")
    '''

    # Loop over the batches, pickle them and add the filepaths to the result...
    ret_val['feature_batches'] = []
    for i, features_gdf_batch in enumerate(features_gdf_batches):
        batch_info = {}
        pickle_filepath = os.path.join(temp_features_dir, f"{i}.pkl")
        logger.info(f"Write pkl of {len(features_gdf_batch)} features: {pickle_filepath}")
        try:
            features_gdf_batch.to_pickle(pickle_filepath)
        except Exception as ex:
            logger.error(f"Exception writing pickle: {pickle_filepath}: {ex}")
            raise 
        batch_info["filepath"] = pickle_filepath
        batch_info["nb_items"] = len(features_gdf_batch)
        ret_val["feature_batches"].append(batch_info)

    # Ready... return
    return ret_val
'''
def calc_stats_image(features_filepath,
                     image_path: str,
                     output_filepath: str,
                     temp_dir: str,
                     log_dir: str,
                     nb_parallel_max: int = 16) -> bool:
    """
    Calculate the statistics for one image.

    Returns True if succesfully completed.
    Remark: easiest it returns something, when used in a parallel way: concurrent.futures likes it better if something is returned
    """
    # When running in parallel processes, the logging needs to be write to seperate files + no console logging
    global logger
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

    # Prepare image + get basic info about it
    image_prepr_path = prepare_image(image_path, temp_dir)
    logger.info(f"Preparing ready, result: {image_prepr_path}")
    image_info = get_image_info(image_prepr_path)
    logger.info(f"image_info: {image_info}")

    features_gdf = load_features_file(features_filepath=features_filepath, 
                                      target_epsg=image_info['image_epsg'],
                                      columns_to_retain=["CODE_OBJ", "geometry"],
                                      bbox=image_info['image_bounds'])

    # Check if features were found, otherwise no use to proceed
    nb_todo = len(features_gdf)
    if nb_todo == 0:
        logger.info(f"No features were found in the bounding box of the image, so return: {image_path}")
        return True
    
    # Calculate the number per batch, but keep the number between 100 and 50000...
    nb_per_batch = min(max(math.ceil(nb_todo/nb_parallel_max), 100), 50000)

    # The features were already sorted on x coordinate, so the features in the batches are 
    # already clustered geographically 
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
'''
def load_features_file(features_filepath: str,
                       columns_to_retain: [],
                       target_epsg: int,
                       bbox = None,
                       polygon = None) -> gpd.GeoDataFrame:
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
        if create_file_atomic(features_prepr_filepath_busy):

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

    # If there is a polygon provided, filter on the polygon (as well)
    if polygon is not None:
        logger.info("Filter polygon provided, start filter")
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs={'init' :'epsg:4326'}, index=[0])
        logger.debug(f"polygon_gdf: {polygon_gdf}")
        logger.debug(f"polygon_gdf.crs: {polygon_gdf.crs}, features_gdf.crs: {features_gdf.crs}")
        polygon_gdf = polygon_gdf.to_crs(features_gdf.crs)
        logger.debug(f"polygon_gdf, after reproj: {polygon_gdf}")
        logger.debug(f"polygon_gdf.crs: {polygon_gdf.crs}, features_gdf.crs: {features_gdf.crs}")
        features_gdf = gpd.sjoin(features_gdf, polygon_gdf, how='inner', op='within')
        
        # Drop column added by sjoin
        features_gdf.drop(columns='index_right', inplace=True)
        '''
        spatial_index = gdf.sindex
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(polygon)]
        '''
        logger.info(f"Filter ready, found {len(features_gdf)}")
            
    # Ready, so return result...
    return features_gdf

def calc_stats_image_gdf(features_gdf,
                         image_path: str,
                         output_filepath: str,
                         log_dir: str,
                         future_start_time = None) -> bool:
    """

    Returns True if succesfully completed.
    Remark: easiest it returns something, when used in a parallel way: concurrent.futures likes it better if something is returned
    """

    # TODO: the different bands should be possible to process in parallel as well... so this function should process only one band!
    # When running in parallel processes, the logging needs to be write to seperate files + no console logging
    global logger
    logger = logging.getLogger('calc_stats_image_gdf')
    logger.propagate = False
    log_filepath = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d_%H-%M-%S}_calc_stats_image_gdf_{os.getpid()}.log")
    fh = logging.FileHandler(filename=log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(funcName)s|%(message)s'))
    logger.addHandler(fh)

    # Log the time between scheduling the future and acually run...
    if future_start_time is not None:
        logger.info(f"Start, {(datetime.now()-future_start_time).total_seconds()} after future was scheduled")

    # If the features_gdf is a string, use it as file path to unpickle geodataframe...
    if isinstance(features_gdf, str):
        features_gdf_pkl_filepath = features_gdf
        logger.info(f"Read pickle: {features_gdf_pkl_filepath}")
        features_gdf = pd.read_pickle(features_gdf_pkl_filepath)
        logger.info(f"Read pickle with {len(features_gdf)} features ready")

    # Get the image data   
    image_data = get_image_data(image_path, bounds=features_gdf.total_bounds, bands=['VV', 'VH'], pixel_buffer=1)
    
    # Loop over image bands
    #image_id = os.path.basename(image_path)
    for band in image_data:

        # Calc zonal stats
        logger.info(f"Calculate statistics for band {band} on {len(features_gdf)} features")
        # TODO: probably some kind of masking is still necessary at the edges of the images!!!
        features_stats = zonal_stats(features_gdf, image_data[band]['data'], 
                                     affine=image_data[band]['transform'], prefix="", nodata=0, geojson_out=True,
                                     stats=["count", "mean", "std", "min", "max", "percentile_25", "percentile_50", "percentile_75"])
        logger.info(f"Statistics calculated... ")
        #features_stats_df = pd.DataFrame(features_stats)
        features_stats_gdf = gpd.GeoDataFrame.from_features(features_stats)
        
        logger.info(f"Dataframe created from stats... ")

        #df = pd.DataFrame.from_dict(df.properties.to_dict(), orient='index')
        #features_stats_gdf['image_id'] = image_id
        features_stats_gdf['band'] = band
        
        logger.info(f"Rename columns... ")
        features_stats_gdf.rename(index=str, columns={"percentile_25": "p25", "percentile_50": "p50","percentile_75": "p75"}, inplace=True)
        
        # df is the dataframe
        if len(features_stats_gdf) > 0:

            # Remove rows with empty data
            features_stats_gdf.dropna(inplace=True)
            if len(features_stats_gdf.values) > 0:
                output_filepath_noext, output_filepath_ext = os.path.splitext(output_filepath)
                output_band_filepath = f"{output_filepath_noext}_{band}{output_filepath_ext}"
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

def get_image_data(image_path,
                   bounds,
                   bands: [],
                   pixel_buffer: int = 0) -> dict:
    """
    Reads the data from and image.

    Adds a small buffer around the bounds asked to evade possible rounding issues.

    Returns a dict of the following structure:
    imagedata[band]['data']: the data read as numpy array
                   ['transform']: the Affine transform of the band read

    Args
        image_path: the path the image
        bounds: the bounds to be read, in coordinates in the projection of the image
        bands: list of bands to be read, eg. "VV", "VH",... 
        pixel_buffer: number to pixels to take as buffer around the bounds provided in pixels
    """
    # Get info about the image
    image_info = get_image_info(image_path)

    # Now get the data
    image_data = {}              # Dict for the transforms and the data per band
    if image_info["image_type"] == 'CARD':
        # Loop over bands and get data
        for band in bands:
            band_relative_filepath = image_info["bands"][band]['relative_filepath']
            image_band_filepath = os.path.join(image_path, band_relative_filepath)
            logger.info(f"Read image data from {image_band_filepath}")
            image_data[band] = {}
            with rasterio.open(image_band_filepath) as src:
                # Determine the window we need to read from the image:
                window_to_read = projected_bounds_to_window(
                        bounds, src.transform, src.width, src.height, pixel_buffer) 
                image_data[band]['transform'] = rasterio.windows.transform(window_to_read, src.transform) 
                # Read!
                logger.debug(f"Read image data from {image_band_filepath}")
                image_data[band]['data'] = src.read(1, window=window_to_read)
                logger.info(f"Image data read")
    else:
        message = f"Only .CARD image format currently supported!, not {image_path}"
        logger.error(message)
        raise NotImplementedError(message)

    return image_data

def get_image_info(image_path) -> dict:

    image_info = {}

    # First determine image format
    image_path_noext, image_ext = os.path.splitext(image_path)
    if image_ext.upper() == ".CARD":
        
        # First extract and fill out some basic info
        image_info["image_type"] = "CARD"
        _, image_basename_noext = os.path.split(image_path_noext)
        image_info["image_id"] = image_basename_noext
        
        # Read info from the metadata file
        metadata_xml_filepath = os.path.join(image_path, 'metadata.xml')
        metadata = ET.parse(metadata_xml_filepath)
        metadata_root = metadata.getroot()

        logger.debug(f"Parse metadata info from {metadata_xml_filepath}")

        # Get the filename
        image_info["filename"] = metadata_root.find('filename').text

        # Get the footprint
        image_info["footprint"] = {}
        #footprint = metadata_root.find('footprint')
        footprint_gml = metadata_root
        poly = footprint_gml.find("{http://www.opengis.net/gml}Polygon")
        image_info["footprint"]["srsname"] = poly.attrib.get("srsName")
        linear_ring = []
        #for coord in poly.findall("{http://www.opengis.net/gml}outerBoundaryIs/{http://www.opengis.net/gml}LinearRing/{http://www.opengis.net/gml}coord"):
        #    linear_ring.append((float(coord.findtext("{http://www.opengis.net/gml}X")),float(coord.findtext("{http://www.opengis.net/gml}Y"))))
        coord_str = poly.find("{http://www.opengis.net/gml}outerBoundaryIs/{http://www.opengis.net/gml}LinearRing/{http://www.opengis.net/gml}coordinates")
        logger.debug(f"coord_str: {coord_str}, coord_str.text: {coord_str.text}")
        coord_list = coord_str.text.split(' ')
        for coord in coord_list:
            # Watch out, latitude (~y) is first, than longitude (~x)
            # TODO: add check if the projection is in degrees (latlon) or coordinates (xy) instead of hardcoded latlon
            y, x = coord.split(',') 
            linear_ring.append((float(x), float(y)))
        image_info["footprint"]["shape"] = sh_polygon.Polygon(linear_ring)

        # Read info from the manifest.safe file
        manifest_xml_searchstring = os.path.join(image_path, f"*_manifest.safe")
        manifest_xml_filepaths = glob.glob(manifest_xml_searchstring)

        # We should only find one .safe file (at the moment)
        if len(manifest_xml_filepaths) != 1:
            message = f"Error: didn't find exactly one .safe file doing glob with {manifest_xml_searchstring}"
            logger.error(message)
            raise Exception(message)

        # Now parse the .safe file
        manifest_xml_filepath = manifest_xml_filepaths[0]
        manifest = ET.parse(manifest_xml_filepath)
        manifest_root = manifest.getroot()

        # Define namespaces...
        ns = {"safe": "http://www.esa.int/safe/sentinel-1.0",
              "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
              "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"}

        logger.debug(f"Parse manifest info from {metadata_xml_filepath}")
        image_info["transmitter_receiver_polarisation"] = []
        for polarisation in manifest_root.findall("metadataSection/metadataObject/metadataWrap/xmlData/s1sarl1:standAloneProductInformation/s1sarl1:transmitterReceiverPolarisation", ns):
            image_info["transmitter_receiver_polarisation"].append(polarisation.text)
        image_info["instrument_mode"] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:platform/safe:instrument/safe:extension/s1sarl1:instrumentMode/s1sarl1:mode", ns).text
        image_info["orbit_properties_pass"] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:orbitReference/safe:extension/s1:orbitProperties/s1:pass", ns).text

        # Now have a look in the files themselves to get band info,...
        # TODO: probably cleaner/easier to read from metadata files...
        image_datadirname = f"{image_basename_noext}.data"
        image_datadir = os.path.join(image_path, image_datadirname)
        band_filepaths = glob.glob(f"{image_datadir}{os.sep}*.img")
        image_info["bands"] = {}
        for i, band_filepath in enumerate(band_filepaths):
            # Extract bound,... info from the first file only (they are all the same)
            # TODO: in S2 they have a different resolution, so then I need to open each image?
            if i == 0:
                logger.debug(f"Read image metadata from {band_filepath}")
                with rasterio.open(band_filepath) as src:
                    image_info['image_bounds'] = src.bounds
                    image_info['image_affine'] = src.transform
                    image_info['image_crs'] = str(src.crs)
                    image_info['image_epsg'] = image_info['image_crs'].upper().replace('EPSG:', '')
                logger.debug(f"Image metadata read: {image_info}")
            band_filename = os.path.basename(band_filepath)
            if band_filename == "Gamma0_VH.img":
                band = 'VH'
            elif band_filename == "Gamma0_VV.img":
                band = 'VV'
            else:
                raise NotImplementedError(f"Filename not supported: {band_filepath}")

            # Add specific info about the band
            image_info['bands'][band] = {}
            image_info['bands'][band]['filepath'] = band_filepath
            image_info['bands'][band]['relative_filepath'] = os.path.join(image_datadirname, band_filename) 
            image_info['bands'][band]['filename'] = band_filename
    else:
        message = f"Only .CARD image format currently supported!, not {image_path}"
        logger.error(message)
        raise NotImplementedError(message)
    
    return image_info

def projected_bounds_to_window(projected_bounds,
                               image_transform, 
                               image_pixel_width: int,
                               image_pixel_height: int,
                               pixel_buffer: int = 0):
    """
    Returns a rasterio.windows.Window to be used in rasterio to read the part of the image specified.

    Args
        projected_bounds: bounds to created the window from, in projected coordinates
        image_transform: Affine transform of the image you want to create the pixel window for
        image_pixel_width: total width of the image you want to create the pixel window for, in pixels
        image_pixel_height: total height of the image you want to create the pixel window for, in pixels
        pixel_buffer: number to pixels to take as buffer around the bounds provided in pixels
    """
    # Take bounds of the features + convert to image pixels  
    xmin, ymin, xmax, ymax = projected_bounds
    window_to_read_raw = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, image_transform)
    
    # Round so it only increases window size
    window_to_read = window_to_read_raw.round_offsets('floor').round_lengths('ceil')
    
    # Now some general math on window properties, but as they are readonly properties, work on copy
    col_off, row_off, width, height = window_to_read.flatten()
    # Add buffer of 1 pixel extra around
    col_off -= pixel_buffer
    row_off -= pixel_buffer
    width += 2*pixel_buffer
    height += 2*pixel_buffer
    
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

def create_file_atomic(filename):
    """
    Create a lock file in an atomic way, so it is threadsafe.

    Returns True if the file was created by this thread, False if the file existed already.
    """
    try:
        os.open(filename,  os.O_CREAT | os.O_EXCL)
        return True
    except FileExistsError:
        return False

def prepare_image(image_path: str,
                  temp_dir: str) -> str:
    """
    Prepares the input image for usages.

    In case of a zip file, the file is unzipped to the temp dir specified.

    Returns the path to the prepared file/directory.
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
            if create_file_atomic(image_unzipped_path_busy):

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

def main():

    # First define/init some general variables/constants
    #base_dir = "X:\\Monitoring\\Markers\\PlayGround\\PIEROG"                                # Base dir
    base_dir = f"{os.path.expanduser('~')}{os.sep}markers{os.sep}cropclassification"
    input_dir = os.path.join(base_dir, 'inputdata')                                         # Input dir

    global logger
    base_log_dir = os.path.join(base_dir, 'log')
    log_dir = os.path.join(base_log_dir, f"{datetime.now():%Y-%m-%d_%H-%M-%S}")
    logger = log_helper.main_log_init(log_dir, __name__)

    # Input features file depends on the year
    input_features_filename = "Prc_BEFL_2018_2018-08-02.shp"
    input_features_filepath = os.path.join(input_dir, input_features_filename)   # Input filepath of the parcel

    # Input image files
    input_image_dir = "X:\\GIS\\GIS DATA\\Ortho_Belgie\\Sentinel2\\S2_NDVI_Periodiek"
    #input_image_searchstr = f"{input_image_dir}{os.sep}*.tif"
    input_image_searchstr = "/mnt/NAS3/CARD/FLANDERS/S1*/L1TC/2018/*/*/*.CARD"
    input_image_filepaths = glob.glob(input_image_searchstr)
    logger.info(f"Found {len(input_image_filepaths)} images to process")

  
    # Temp dir + clean contents from it.
    temp_dir = os.path.join(base_dir, "tmp")
    logger.info(f"Clean the temp dir")
    if os.path.exists(temp_dir):
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(temp_dir + os.sep)
    
    # Output dir 
    test = False
    if not test:
        output_basedir = os.path.join(base_dir, "output")
    else:
        output_basedir = os.path.join(base_dir, "output_test")
        logger.info(f"As we are testing, use test output basedir: {output_basedir}")
    input_features_filename_noext = os.path.splitext(input_features_filename)[0]
    output_dir = os.path.join(output_basedir, input_features_filename_noext)
    if test:
        if os.path.exists(output_dir):
            logger.info(f"As we are only testing, clean the output dir: {output_dir}")
            # By adding a / at the end, only the contents are recursively deleted
            shutil.rmtree(output_dir + os.sep)
    
    if test:
        # For testing purposes: different output dirs + extra cleanup...
        # Take only the x first images found while testing
        input_image_filepaths = input_image_filepaths[:10]
        logger.info(f"As we are only testing, process only {len(input_image_filepaths)} first images")

        # Clean log dir 
        logger.info(f"As we are only testing, delete the contents of the log dir")
        if os.path.exists(base_log_dir):
            # By adding a / at the end, only the contents are recursively deleted
            shutil.rmtree(base_log_dir + os.sep)
        
    """
    # TEST to extract exact footprint from S1 image...
    filepath = "/mnt/NAS3/CARD/FLANDERS/S1A/L1TC/2017/01/01/S1A_IW_GRDH_1SDV_20170101T055005_20170101T055030_014634_017CB9_Orb_RBN_RTN_Cal_TC.CARD/S1A_IW_GRDH_1SDV_20170101T055005_20170101T055030_014634_017CB9_Orb_RBN_RTN_Cal_TC.data/Gamma0_VH.img"
    image = rasterio.open(filepath)
    geoms = list(rasterio.features.dataset_features(src=image, as_mask=True, precision=5))
    footprint = gpd.GeoDataFrame.from_features(geoms)        
    logger.info(footprint)
    footprint = footprint.simplify(0.00001)        
    logger.info(footprint)
    logger.info("Ready")
    # Start calculation
    """

    logger.info(f"Start processing")
    calc_stats(features_filepath=input_features_filepath,
               image_paths=input_image_filepaths,
               output_dir=output_dir,
               temp_dir=temp_dir,
               log_dir=log_dir)

if __name__ == '__main__':
    main()
