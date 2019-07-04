# -*- coding: utf-8 -*-
"""
Calculate timeseries data per image.
"""

from concurrent import futures
from datetime import datetime
import glob
import io
import logging
import math
import multiprocessing
import numpy as np
import os
import shutil
import signal    # To catch CTRL-C explicitly and kill children
import sys
import time

from affine import Affine
import geopandas as gpd
import pandas as pd
import psutil    # To catch CTRL-C explicitly and kill children
import rasterio
from rasterio import windows
from rasterstats import zonal_stats
from shapely.geometry import polygon as sh_polygon
import xml.etree.ElementTree as ET
import zipfile

from cropclassification.helpers import pandas_helper as pdh
from cropclassification.helpers import geofile

# General init
logger = logging.getLogger(__name__)

#signal.signal(signal.SIGINT, signal.default_int_handler)

def calc_stats(features_filepath: str,
               id_column: str,
               image_paths: str,
               bands: [],
               output_dir: str,
               temp_dir: str,
               log_dir: str,
               force: bool = False):
    """
    Calculate the statistics.
    """

    # TODO: probably need to apply some object oriented approach here for "image", because there are to many properties,... to be clean/clear this way.
    # TODO: maybe passing the executor pool to a calc_stats_for_image function can have both the advantage of not creating loads of processes + 
    # keep the cleanup logic after calculation together with the processing logic

    # Some checks on the input parameters
    if len(image_paths) == 0:
        logger.info("No image paths... so nothing to do, so return")
        return

    # General init
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create process pool for parallelisation...
    nb_parallel_max = multiprocessing.cpu_count()
    nb_parallel = nb_parallel_max
    with futures.ProcessPoolExecutor(nb_parallel) as pool:

        # Loop over all images to start the data preparation for each of them in parallel...
        image_paths.sort()
        start_time = datetime.now()
        nb_todo = len(image_paths)
        nb_errors_max = 10
        nb_errors = 0

        image_dict = {}
        calc_stats_batch_dict = {}
        nb_done_total = 0
        i = 0

        try:
            while True:

                # If not all images aren't processed/processing yet
                if i < len(image_paths):    

                    # Get more detailed info about the image
                    image_path = image_paths[i]
                    i += 1
                    try:
                        image_info = get_image_info(image_path)
                    except:
                        # If not possible to get info for image, log and skip it
                        nb_errors += 1 
                        logger.exception(f"SKIP image, because error getting info for {image_path}")
                        nb_todo -= 1
                        continue

                    # If sentinel2 and cloud coverage too high... skip
                    max_cloudcover_pct = 10
                    if image_info['satellite'].startswith('S2') and image_info['Cloud_Coverage_Assessment'] > max_cloudcover_pct:
                        logger.info(f"SKIP image, Cloud_Coverage_Assessment: {image_info['Cloud_Coverage_Assessment']:0.2f} > {max_cloudcover_pct} for {image_path}")
                        nb_todo -= 1
                        continue

                    # If sentinel1 and wrong productTimelinessCategory, skip: we only want 1 type to evade images used twice
                    if image_info['satellite'].startswith('S1') and image_info['productTimelinessCategory'] != 'Fast-24h':
                        logger.info(f"SKIP image, productTimelinessCategory should be 'Fast-24h', but is: {image_info['productTimelinessCategory']} for {image_path}")
                        nb_todo -= 1
                        continue

                    # Create base output filename
                    orbit = None
                    if image_info['satellite'].startswith('S1'):
                        orbit = image_info['orbit_properties_pass']    
                    output_base_filepath = get_output_filepath(
                            features_filepath, image_path, output_dir, orbit, band=None)
                    output_base_dir, output_base_filename = os.path.split(output_base_filepath)
                    output_base_busy_filepath = f"{output_base_dir}{os.sep}BUSY_{output_base_filename}"

                    # Check for which bands there is a valid output file already
                    bands_done = 0
                    for band in bands:
                        # Prepare the output filepaths...
                        output_band_filepath = get_output_filepath(
                                features_filepath, image_path, output_dir, orbit, band)
                        output_band_dir, output_band_filename = os.path.split(output_band_filepath)
                        output_band_busy_filepath = f"{output_band_dir}{os.sep}BUSY_{output_band_filename}"

                        # If a busy output file exists, remove it, otherwise we can get double data in it...
                        if os.path.exists(output_band_busy_filepath):
                            os.remove(output_band_busy_filepath)

                        # Check if the output file exists already
                        if os.path.exists(output_band_filepath):
                            if force == False:
                                logger.debug(f"Output file for band exists already {output_band_filepath}")
                                bands_done += 1
                            else:
                                os.remove(output_band_filepath)

                    # If all bands already processed, skip image...
                    if len(bands) == bands_done:
                        logger.info(f"SKIP image: output files for all bands exist already for {output_base_filepath}")
                        nb_todo -= 1
                        continue

                    # Always make sure there are maximum nb_parallel_max prepare_calc's active
                    nb_busy = 0
                    for image_path_tmp in image_dict:
                        if image_dict[image_path_tmp]['status'] == 'IMAGE_PREPARE_CALC_BUSY':
                            nb_busy += 1
                    if nb_busy < nb_parallel_max:
                        logger.debug(f"nb_busy: {nb_busy}, nb_parallel_max: {nb_parallel_max}, so nb_busy < nb_parallel_max")          
                        # Start the prepare processing assync
                        # TODO: possibly it is cleaner to do this per band...
                        future = pool.submit(prepare_calc, 
                                            features_filepath,
                                            id_column,
                                            image_path,
                                            output_base_filepath,
                                            temp_dir,
                                            log_dir,
                                            nb_parallel_max)
                        image_dict[image_path] = {'features_filepath': features_filepath,
                                                'prepare_calc_future': future, 
                                                'image_info': image_info,
                                                'prepare_calc_starttime': datetime.now(),
                                                'output_base_filepath': output_base_filepath,
                                                'output_base_busy_filepath': output_base_busy_filepath,
                                                'status': 'IMAGE_PREPARE_CALC_BUSY'}
                        
                        # Jump to next image to start the prepare_calc for it...
                        continue
                    else:
                        # There are already enough prepare's busy, reset i so this image is
                        # tried again in the next loop...
                        i -= 1

                # Loop through the images to find which are ready... to start the real calculations...
                for image_path in image_dict:
                    
                    # Get all info that is available in the dict with futures for the one that has now completed
                    image = image_dict[image_path]
                        
                    # If the status isn't PREPARE_CALC_BUSY or if the prepare_calc is still running, go to next...
                    if(image['status'] != 'IMAGE_PREPARE_CALC_BUSY' 
                    or image['prepare_calc_future'].running()):
                        continue

                    # Extract the result from the preparation
                    try:
                        # Get the result from the completed  prepare_inputs
                        prepare_calc_result = image['prepare_calc_future'].result()

                        # If nb_features to be treated is 0... create (empty) output files and continue with next...
                        if prepare_calc_result['nb_features_to_calc_total'] == 0:
                            for band in bands:
                                # Prepare the output filepath...
                                orbit = None
                                if image['image_info']['satellite'].startswith('S1'):
                                    orbit = image['image_info']['orbit_properties_pass']
                                output_band_filepath = get_output_filepath(
                                        features_filepath, image_path, output_dir, orbit, band)

                                # Create output file
                                logger.info(f"No features found overlapping image, just create done file: {output_band_filepath}")                                
                                create_file_atomic(output_band_filepath)
                                image['status'] = 'IMAGE_CALC_DONE'

                            # Jump to next image
                            continue

                        # Add info about the result of the prepare_calc to the image info...
                        image['image_prepared_path'] = prepare_calc_result['image_prepared_path']
                        image['feature_batches'] = prepare_calc_result['feature_batches']
                        image['nb_features_to_calc_total'] = prepare_calc_result['nb_features_to_calc_total']
                        image['temp_features_dir'] = prepare_calc_result['temp_features_dir']

                        # Set status to calc_is_busy so we know calculation is busy...
                        image['status'] = 'IMAGE_CALC_BUSY'
                        image['calc_starttime'] = datetime.now()

                        # Now loop through all prepared feature batches to start the statistics calculation for each
                        logger.info(f"Start statistics calculation for {image_path}")  
                        for features_batch in image['feature_batches']:
                            start_time_batch = datetime.now()
                            future = pool.submit(calc_stats_image_gdf, 
                                                features_batch['filepath'],
                                                id_column,
                                                image['image_prepared_path'],
                                                bands,
                                                image['output_base_busy_filepath'],
                                                log_dir,
                                                start_time_batch)
                            calc_stats_batch_dict[features_batch['filepath']] = {
                                    'calc_stats_future': future,
                                    'image_path': image_path,
                                    'image_prepared_path': image['image_prepared_path'], 
                                    'start_time_batch': start_time_batch,
                                    'nb_items_batch': features_batch['nb_items'],
                                    'status': 'BATCH_CALC_BUSY'}

                    except Exception as ex:
                        message = f"Exception getting result of prepare_calc for {image}"
                        logger.exception(message)
                        nb_errors += 1
                        if nb_errors > nb_errors_max:
                            raise Exception(message) from ex

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
                            message = f"Exception getting result of calc_stats_image_gdf for {calc_stats_batch_info}"
                            logger.exception(message)
                            nb_errors += 1
                            if nb_errors > nb_errors_max:
                                raise Exception(message) from ex 

                # Loop over all image_paths that are busy being calculated to check if there are still calc stats batches busy
                for image_path in image_dict:

                    # If the image is busy being calculated...
                    image = image_dict[image_path]
                    if image['status'] == 'IMAGE_CALC_BUSY':

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
                            image['status'] = 'IMAGE_CALC_DONE'
                    
                            # If the preprocessing created a temp image file, clean it up...
                            image_prepared_path = image['image_prepared_path']
                            if image_prepared_path != image_path:
                                logger.info(f"Remove local temp image copy: {image_prepared_path}")
                                if os.path.isdir(image_prepared_path):           
                                    shutil.rmtree(image_prepared_path, ignore_errors=True)
                                else:
                                    os.remove(image_prepared_path)

                            # If the preprocessing created temp pickle files with features, clean them up...
                            shutil.rmtree(image['temp_features_dir'], ignore_errors=True)

                            # Rename the (completed) output files
                            output_base_filepath_noext, output_base_ext = os.path.splitext(image['output_base_filepath'])
                            output_base_busy_filepath_noext, output_base_busy_ext = os.path.splitext(image['output_base_busy_filepath'])
                            for band in bands:
                                # TODO: creating the output filepaths should probably be cleaner centralised...
                                output_band_busy_filepath = f"{output_base_busy_filepath_noext}_{band}{output_base_busy_ext}"
                                output_band_filepath = f"{output_base_filepath_noext}_{band}{output_base_ext}"

                                # If BUSY output file exists, rename it
                                if os.path.exists(output_band_busy_filepath):
                                    os.rename(output_band_busy_filepath, output_band_filepath)
                                else:
                                    # If BUSY output file doesn't exist, create empty file
                                    logger.info(f"No features found overlapping image after processing, create done file: {output_band_filepath}")
                                    create_file_atomic(output_band_filepath)

                            # Log the progress and prediction speed
                            logger.info(f"Ready processing image: {image_path}")
                            nb_done_latestbatch = 1
                            nb_done_total += nb_done_latestbatch
                            progress_msg = get_progress_message(nb_todo, nb_done_total, nb_done_latestbatch, start_time, image['calc_starttime'])
                            logger.info(progress_msg)

                # If all images have been started, check if there are still images busy in dict 
                all_done = False
                if i == len(image_paths):
                    all_done = True
                    for image_path in image_dict:
                        if image_dict[image_path]['status'] != 'IMAGE_CALC_DONE':
                            all_done = False
                            break   

                # If no processing is needed, or if all processing is ready, stop the never ending loop...
                if len(image_dict) == 0 or all_done is True:
                    if nb_errors > 0:
                        raise Exception(f"Ready processing, but there were {nb_errors} errors!")
                    break 
                else:
                    # Sleep before starting next iteration...
                    time.sleep(0.1)

        except KeyboardInterrupt:
            # If CTRL+C is used, shut down pool and kill children
            print('You pressed Ctrl+C')
            print('Worker processes are being stopped, followed by exit!')

            # Stop process pool + kill children + exit
            try:
                pool.shutdown(wait=False)
                parent = psutil.Process(os.getpid())
                children = parent.children(recursive=True)
                for process_pid in children:
                    print(f"Kill child with pid {process_pid}")
                    process_pid.send_signal(signal.SIGTERM)
            finally:
                sys.exit(1)

        logger.info(f"Time taken to calculate data for {nb_todo} images: {(datetime.now()-start_time).total_seconds()} sec")

def get_output_filepath(features_filepath: str, 
                        image_path: str,
                        output_dir: str,
                        orbit_properties_pass: str,
                        band: str):
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
    if orbit_properties_pass is not None:
        if orbit_properties_pass == 'ASCENDING':
            orbit = '_ASC'
        elif orbit_properties_pass == 'DESCENDING':
            orbit = '_DESC'
        else:
            message = f"Unknown orbit_properties_pass: {orbit_properties_pass}"
            logger.error(message)
            raise Exception(message)
    else:
        orbit = ''

    # Prepare basic filepath
    output_filepath_noext = os.path.join(
            output_dir, f"{features_filename}__{image_filename}{orbit}")

    # If a band was specified
    if band is not None:
        output_filepath_noext = f"{output_filepath_noext}_{band}"
    
    # Add extension
    output_filepath = f"{output_filepath_noext}.sqlite"
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
                 id_column,
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
    footprint_shape = None
    if 'footprint' in image_info:
        logger.info(f"poly: {image_info['footprint']['shape']}")
        footprint_shape = image_info['footprint']['shape']
    features_gdf = load_features_file(features_filepath=features_filepath, 
                                      target_epsg=image_info['image_epsg'],
                                      columns_to_retain=[id_column, 'geometry'],
                                      bbox=image_info['image_bounds'],
                                      polygon=footprint_shape)

    # Check if overlapping features were found, otherwise no use to proceed
    nb_todo = len(features_gdf.index)
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
        logger.info(f"Write pkl of {len(features_gdf_batch.index)} features: {pickle_filepath}")
        try:
            features_gdf_batch.to_pickle(pickle_filepath)
        except Exception as ex:
            logger.exception(f"Exception writing pickle: {pickle_filepath}")
            raise 
        batch_info["filepath"] = pickle_filepath
        batch_info["nb_items"] = len(features_gdf_batch.index)
        ret_val["feature_batches"].append(batch_info)

    # Ready... return
    return ret_val

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
        features_prepr_filepath = f"{features_filepath_noext}_{target_epsg}.gpkg"
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
                features_gdf = geofile.read_file(features_filepath)
                logger.info(f"Read ready, found {len(features_gdf.index)} features, crs: {features_gdf.crs}, took {(datetime.now()-start_time).total_seconds()} s")
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
                logger.info(f"Write {len(features_gdf.index)} reprojected features to {features_prepr_filepath}")
                geofile.to_file(features_gdf, features_prepr_filepath, index=False)
                logger.info(f"Reprojected features written")

                """ 
                # TODO: redo performance test of using pkl as cache file
                features_gdf.to_pickle('file.pkl')
                logger.info(f"Pickled")
                df2 = pd.read_pickle('file.pkl')
                logger.info(f"Pickle read")
                """
            except Exception as ex:
                # If an exception occurs...
                message = f"Exception, so delete possibly not completed file: {features_prepr_filepath}"
                logger.exception(message)
                os.remove(features_prepr_filepath)
                raise Exception(message) from ex
            finally:    
                # Remove lock file as everything is ready for other processes to use it...
                os.remove(features_prepr_filepath_busy)

            # Now filter the parcels that are in bbox provided
            if bbox is not None:
                logger.info(f"bbox provided, so filter features in the bbox of {bbox}")
                xmin, ymin, xmax, ymax = bbox
                features_gdf = features_gdf.cx[xmin:xmax, ymin:ymax]
                logger.info(f"Found {len(features_gdf.index)} features in bbox")

                """ Slower + crashes?
                logger.info(f"Filter only features in the bbox of image with spatial index {image_bounds}")
                spatial_index = features_gdf.sindex
                possible_matches_index = list(spatial_index.intersection(image_bounds))
                features_gdf = features_gdf.iloc[possible_matches_index]
                #features_gdf = features_gdf[features_gdf.intersects(image_shape)]
                logger.info(f"Found {len(features_gdf.index)} features in the bbox of image {image_bounds}")                    
                """

    # If there exists already a file with the features in the right projection, we can just read the data
    if features_gdf is None:

        # If a "busy file" still exists, the file isn't ready yet, but another process is working on it, so wait till it disappears
        while os.path.exists(features_prepr_filepath_busy):
            time.sleep(1)

        logger.info(f"Read {features_prepr_filepath}")
        start_time = datetime.now()
        features_gdf = geofile.read_file(features_prepr_filepath, bbox=bbox)
        logger.info(f"Read ready, found {len(features_gdf.index)} features, crs: {features_gdf.crs}, took {(datetime.now()-start_time).total_seconds()} s")

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
        logger.info(f"Filter ready, found {len(features_gdf.index)}")
            
    # Ready, so return result...
    return features_gdf

def calc_stats_image_gdf(features_gdf,
                         id_column: str,
                         image_path: str,
                         bands: [],
                         output_base_filepath: str,
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
        logger.info(f"Read pickle with {len(features_gdf.index)} features ready")

    # Reset index, otherwise the concat later one gives wrong results
    features_gdf.reset_index(inplace=True)
    output_base_filepath_noext, output_ext = os.path.splitext(output_base_filepath)

    # If the image has a quality band, check that one first so parcels with
    # bad pixels can be removed as the data can't be trusted anyway
    quality_band = 'SCL-20m'
    if quality_band in bands:
        # Specific for Scene Classification (SCL) band, interprete already
        # Folowing values are considered "bad":
        #   -> 0 (no_data), 1 (saturated or defective), 3 (cloud shadows),
        #      8: (cloud, medium proba), 9 (cloud, high proba), 11 (snow)
        # These are considered "OK":
        #   -> 2 (dark area pixels), 4 (vegetation), 5 (not_vegetated),
        #      6 (water), 7 (unclassified), 10 (thin cirrus)
        # List of classes: https://usermanual.readthedocs.io/en/latest/pages/ProductGuide.html#quality-indicator-bands
        
        # Get the image data and calculate
        logger.info(f"Calculate categorical counts for band {quality_band} on {len(features_gdf.index)} features")
        image_data = get_image_data(image_path, bounds=features_gdf.total_bounds, bands=[quality_band], pixel_buffer=1)
        features_stats = zonal_stats(features_gdf, image_data[quality_band]['data'], 
                affine=image_data[quality_band]['transform'], prefix="", nodata=0, categorical=True)
        features_stats_df = pd.DataFrame(features_stats)
        features_stats_df.fillna(value=0, inplace=True)

        # Define which columns contain good pixels and which don't
        bad_pixels_cols = ['0.0', '1.0', '3.0', '8.0', '9.0', '11.0']
        all_cols = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', 
                    '6.0', '7.0', '8.0', '9.0', '10.0', '11.0']

        # Make sure the dataframe contains columns for all possible values 
        for i, col in enumerate(all_cols):
            if col in features_stats_df.columns:
                # Cast to int, otherwise is float
                features_stats_df[col] = features_stats_df[col].astype('int32')
            else:
                features_stats_df.insert(loc=i, column=col, value=0)

        # Add bad pixels column 
        nb_bad_pixels_column = "nb_bad_pixels"
        features_stats_df[nb_bad_pixels_column] = features_stats_df[bad_pixels_cols].sum(axis=1)
        
        # Add index and write to file
        features_stats_df.insert(loc=0, column=id_column, value=features_gdf[id_column])
        output_band_filepath = f"{output_base_filepath_noext}_{quality_band}{output_ext}"
        logger.info(f"Write data for {len(features_stats_df.index)} parcels found to {output_band_filepath}")
        pdh.to_file(features_stats_df, output_band_filepath, index=False, append=True)

        # Use the nb_bad_pixels column to filter only parcels without bad pixels
        features_gdf.insert(loc=0, column=nb_bad_pixels_column, value=features_stats_df[nb_bad_pixels_column])
        features_gdf = features_gdf.loc[features_gdf[nb_bad_pixels_column] == 0]
        features_gdf.drop(columns=[nb_bad_pixels_column], inplace=True)
        
        # Check if there are still features to be calculated
        if len(features_gdf.index) == 0:
            logger.info(f"After checking quality band, no more features to be calculated, so stop")
            return True

    # Loop over image bands
    for band in bands:
        # The quality band is already treated, so skip it here
        if band == 'SCL-20m':
            continue

        # Get the image data and calculate statistics
        logger.info(f"Read band {band} for bounds {features_total_bounds}")
        image_data = get_image_data(
                image_path, bounds=features_total_bounds, bands=[band], pixel_buffer=1)
        
        # Upsample the image to double resolution, so we can use all_touched=True without 
        # introducing big errors due to mixels
        upsample_factor = 2
        image_data_upsampled = (image_data[band]['data'].repeat(upsample_factor, axis=0)
                                                        .repeat(upsample_factor, axis=1))
        affine_upsampled = image_data[band]['transform'] * Affine.scale(1/upsample_factor)

        logger.info(f"Calculate zonal statistics for band {band} on {len(features_gdf.index)} features")
        features_stats = zonal_stats(features_gdf, image_data_upsampled, 
                affine=affine_upsampled, prefix="", nodata=0, all_touched=True,
                stats=['count', 'mean', 'std', 'min', 'max'])
        features_stats_df = pd.DataFrame(features_stats)
        features_stats_df['count'] = features_stats_df['count'].divide(upsample_factor*2)

        # Add original id column to statistics dataframe
        features_stats_df.insert(loc=0, column=id_column, value=features_gdf[id_column])

        # Remove rows with empty data
        features_stats_df.dropna(inplace=True)
        if len(features_stats_df.index) == 0:
            logger.info(f"No data found for band {band}, so no use to process other bands")
            return True
        features_stats_df.set_index(id_column, inplace=True)
        output_band_filepath = f"{output_base_filepath_noext}_{band}{output_ext}"
        logger.info(f"Write data for {len(features_stats_df.index)} parcels found to {output_band_filepath}")
        pdh.to_file(features_stats_df, output_band_filepath, append=True)

    return True

def get_image_data(image_path,
                   bounds,
                   bands: [],
                   pixel_buffer: int = 0) -> dict:
    """
    Reads the data from the image.

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
    if image_info['image_type'] in ('CARD', 'SAFE'):
        # Loop over bands and get data
        for band in bands:
            band_relative_filepath = image_info['bands'][band]['relative_filepath']
            image_band_filepath = os.path.join(image_path, band_relative_filepath)
            band_index = image_info['bands'][band]['bandindex']
            logger.info(f"Read image data from {image_band_filepath}, with band_index: {band_index}")
            image_data[band] = {}
            with rasterio.open(image_band_filepath) as src:
                # Determine the window we need to read from the image:
                window_to_read = projected_bounds_to_window(
                        bounds, src.transform, src.width, src.height, pixel_buffer) 
                image_data[band]['transform'] = rasterio.windows.transform(window_to_read, 
                                                                           src.transform) 
                # Read!
                # Remark: bandindex in rasterio is 1-based instead of 0-based -> +1
                logger.debug(f"Read image data from {image_band_filepath}")
                image_data[band]['data'] = src.read(band_index+1, window=window_to_read)

                # If dB data should be converted to natural + the image band data is in dB
                #if db_to_natural is True and image_info['bands'][band]['is_db'] is True:
                #    image_data[band]['data'] = np.power(10, np.divide(image_data[band]['data'], 10))
                logger.info(f"Image data read")
    else:
        message = f"Format currently not supported: {image_path}"
        logger.error(message)
        raise NotImplementedError(message)

    return image_data

def get_image_info(image_path) -> dict:

    image_info = {}

    # First determine general info of image
    image_path_noext, image_ext = os.path.splitext(image_path)
    _, image_basename_noext = os.path.split(image_path_noext)
    image_basename_noext_split = image_basename_noext.split('_')
    image_info['satellite'] = image_basename_noext_split[0]
    
    # Specific code per image type
    if image_ext.upper() == '.CARD':
        
        # This is a sentinel 1 image (GRD or coherence)
        # First extract and fill out some basic info
        image_info['image_type'] = 'CARD'
        image_info['image_id'] = image_basename_noext
        
        # Read info from the metadata file
        metadata_xml_filepath = os.path.join(image_path, 'metadata.xml')
        metadata = ET.parse(metadata_xml_filepath)
        metadata_root = metadata.getroot()

        logger.debug(f"Parse metadata info from {metadata_xml_filepath}")

        try:            
            # Get the filename
            image_info['filename'] = metadata_root.find('productFilename').text

            # Get the footprint
            image_info['footprint'] = {}
            footprint = metadata_root.find('footprint')
            poly = footprint.find("{http://www.opengis.net/gml}Polygon")
            image_info['footprint']['srsname'] = poly.attrib.get('srsName')
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
            image_info['footprint']['shape'] = sh_polygon.Polygon(linear_ring)
        except Exception as ex:
            raise Exception(f"Exception extracting info from {metadata_xml_filepath}") from ex

        # Read info from the manifest.safe file
        manifest_xml_searchstring = os.path.join(image_path, f"*_manifest.safe")
        manifest_xml_filepaths = glob.glob(manifest_xml_searchstring)

        # The number of .safe indicates whether it is a GRD or a Coherence image
        nb_safefiles = len(manifest_xml_filepaths)
        if nb_safefiles == 1:

            # Now parse the .safe file
            manifest_xml_filepath = manifest_xml_filepaths[0]

            try:
                manifest = ET.parse(manifest_xml_filepath)
                manifest_root = manifest.getroot()

                # Define namespaces...
                ns = {'safe': "http://www.esa.int/safe/sentinel-1.0",
                      's1': "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
                      's1sarl1': "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"}

                logger.debug(f"Parse manifest info from {metadata_xml_filepath}")
                image_info['transmitter_receiver_polarisation'] = []
                for polarisation in manifest_root.findall("metadataSection/metadataObject/metadataWrap/xmlData/s1sarl1:standAloneProductInformation/s1sarl1:transmitterReceiverPolarisation", ns):
                    image_info['transmitter_receiver_polarisation'].append(polarisation.text)
                image_info['productTimelinessCategory'] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/s1sarl1:standAloneProductInformation/s1sarl1:productTimelinessCategory", ns).text

                image_info['instrument_mode'] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:platform/safe:instrument/safe:extension/s1sarl1:instrumentMode/s1sarl1:mode", ns).text
                image_info['orbit_properties_pass'] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:orbitReference/safe:extension/s1:orbitProperties/s1:pass", ns).text

                # Now have a look in the files themselves to get band info,...
                # TODO: probably cleaner/easier to read from metadata files...
                image_basename_noext_nodate = image_basename_noext[:image_basename_noext.rfind('_', 0)]                               
                image_datadirname = f"{image_basename_noext_nodate}.data"
                image_datadir = os.path.join(image_path, image_datadirname)
                band_filepaths = glob.glob(f"{image_datadir}{os.sep}*.img")

                # If no files were found, error!
                if len(band_filepaths) == 0:
                    message = f"No image files found in {image_datadir}!"
                    logger.error(message)
                    raise Exception(message)

                image_info['bands'] = {}
                for i, band_filepath in enumerate(band_filepaths):
                    # Extract bound,... info from the first file only (they are all the same)
                    if i == 0:
                        #logger.debug(f"Read image metadata from {band_filepath}")
                        with rasterio.open(band_filepath) as src:
                            image_info['image_bounds'] = src.bounds
                            image_info['image_affine'] = src.transform
                            image_info['image_crs'] = str(src.crs)
                            image_info['image_epsg'] = image_info['image_crs'].upper().replace('EPSG:', '')
                        #logger.debug(f"Image metadata read: {image_info}")
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
                    image_info['bands'][band]['bandindex'] = 0

            except Exception as ex:
                raise Exception(f"Exception extracting info from {manifest_xml_filepath}") from ex

        elif nb_safefiles == 2 or nb_safefiles == 3:
            # 2 safe files -> coherence
            # Now parse the first .safe file
            # TODO: maybe check if the info in all safe files  are the same or?.?
            manifest_xml_filepath = manifest_xml_filepaths[0]
            
            try:
                manifest = ET.parse(manifest_xml_filepath)
                manifest_root = manifest.getroot()

                # Define namespaces...
                ns = {"safe": "http://www.esa.int/safe/sentinel-1.0",
                      "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
                      "s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"}

                #logger.debug(f"Parse manifest info from {metadata_xml_filepath}")
                image_info["transmitter_receiver_polarisation"] = []
                for polarisation in manifest_root.findall("metadataSection/metadataObject/metadataWrap/xmlData/s1sarl1:standAloneProductInformation/s1sarl1:transmitterReceiverPolarisation", ns):
                    image_info["transmitter_receiver_polarisation"].append(polarisation.text)
                image_info['productTimelinessCategory'] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/s1sarl1:standAloneProductInformation/s1sarl1:productTimelinessCategory", ns).text
                image_info["instrument_mode"] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:platform/safe:instrument/safe:extension/s1sarl1:instrumentMode/s1sarl1:mode", ns).text
                image_info["orbit_properties_pass"] = manifest_root.find("metadataSection/metadataObject/metadataWrap/xmlData/safe:orbitReference/safe:extension/s1:orbitProperties/s1:pass", ns).text

                # Now have a look in the files themselves to get band info,...
                # TODO: probably cleaner/easier to read from metadata files...
                # For coherence filename, remove extra .LCO1 extension + date
                #image_datafilebasename, _ = os.path.splitext(image_basename_noext)
                #image_datafilebasename = image_datafilebasename[:image_datafilebasename.rindex('_')]
                image_basename_noext_nodate = image_basename_noext[:image_basename_noext.rfind('_', 0)]
                image_datafilename = f"{image_basename_noext_nodate}_byte.tif"
                image_datafilepath = os.path.join(image_path, image_datafilename)
                
                #logger.debug(f"Read image metadata from {image_datafilepath}")            
                with rasterio.open(image_datafilepath) as src:
                    image_info['image_bounds'] = src.bounds
                    image_info['image_affine'] = src.transform
                    image_info['image_crs'] = str(src.crs)
                    image_info['image_epsg'] = image_info['image_crs'].upper().replace('EPSG:', '')
                #logger.debug(f"Image metadata read: {image_info}")

                # Add specific info about the bands
                image_info["bands"] = {}
                band = 'VH'
                image_info['bands'][band] = {}
                image_info['bands'][band]['filepath'] = image_datafilepath
                image_info['bands'][band]['relative_filepath'] = image_datafilename
                image_info['bands'][band]['filename'] = image_datafilename
                image_info['bands'][band]['bandindex'] = 0

                band = 'VV'
                image_info['bands'][band] = {}
                image_info['bands'][band]['filepath'] = image_datafilepath
                image_info['bands'][band]['relative_filepath'] = image_datafilename 
                image_info['bands'][band]['filename'] = image_datafilename
                image_info['bands'][band]['bandindex'] = 1

            except Exception as ex:
                raise Exception(f"Exception extracting info from {manifest_xml_filepath}") from ex
        else:
            message = f"Error: found {nb_safefiles} .safe files doing glob with {manifest_xml_searchstring}"
            logger.error(message)
            raise Exception(message)

    elif image_ext.upper() == ".SAFE":
        # This is a level 2 sentinel 2 file

        # First extract and fill out some basic info
        image_info["image_type"] = "SAFE"
        _, image_basename_noext = os.path.split(image_path_noext)
        image_info["image_id"] = image_basename_noext

        # Read info from the manifest.safe file
        '''
        manifest_xml_searchstring = os.path.join(image_path, f"*_manifest.safe")
        manifest_xml_filepaths = glob.glob(manifest_xml_searchstring)

        # We should only find one .safe file (at the moment)
        nb_safefiles = len(manifest_xml_filepaths)
        if nb_safefiles == 1:

        # Now parse the metadata xml file
        manifest_xml_filepath = manifest_xml_filepaths[0]
        '''
        metadata_xml_filepath = os.path.join(image_path, "MTD_MSIL2A.xml")
        try:
            metadata = ET.parse(metadata_xml_filepath)
            metadata_root = metadata.getroot()

            # Define namespaces...
            #xsi:schemaLocation="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
            ns = {'n1': "https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
                  'xsi': "http://www.w3.org/2001/XMLSchema-instance"}

            #logger.debug(f"Parse metadata info from {metadata_xml_filepath}")
            image_info['Cloud_Coverage_Assessment'] = float(metadata_root.find("n1:Quality_Indicators_Info/Cloud_Coverage_Assessment", ns).text)
            
        except Exception as ex:
            raise Exception(f"Exception extracting info from {metadata_xml_filepath}") from ex
            
        # Now have a look in the files themselves to get band info,...
        # TODO: probably cleaner/easier to read from metadata files?
        image_datadir = os.path.join(image_path, "GRANULE")
        band_filepaths = glob.glob(f"{image_datadir}{os.sep}**{os.sep}*.jp2", recursive=True)

        # If no files were found, error!
        if len(band_filepaths) == 0:
            message = f"No image files found in {image_datadir}!"
            logger.error(message)
            raise Exception(message)

        image_info["bands"] = {}
        for i, band_filepath in enumerate(band_filepaths):
            band_filename = os.path.basename(band_filepath)
            band_filename_noext, _ = os.path.splitext(band_filename)
            band_filename_noext_split = band_filename_noext.split('_')

            if len(band_filename_noext_split) == 4:
                # IMG_DATA files
                band = f"{band_filename_noext_split[2]}-{band_filename_noext_split[3]}"
            elif len(band_filename_noext_split) == 5:
                # IMG_DATA files
                band = f"{band_filename_noext_split[3]}-{band_filename_noext_split[4]}"
            elif len(band_filename_noext_split) == 3:
                # QI_DATA files
                band = band_filename_noext
            else:
                message = f"Filename of band doesn't have supported format: {band_filepath}"
                logger.error(message)
                raise Exception(message)

            # Add specific info about the band
            image_info['bands'][band] = {}
            image_info['bands'][band]['filepath'] = band_filepath
            image_info['bands'][band]['relative_filepath'] = band_filepath.replace(image_path, '')[1:]
            image_info['bands'][band]['filename'] = band_filename
            image_info['bands'][band]['bandindex'] = 0

            # Extract bound,... info 
            logger.debug(f"Read image metadata from {band_filepath}")
            with rasterio.open(band_filepath) as src:
                image_info['bands'][band]['bounds'] = src.bounds
                image_info['bands'][band]['affine'] = src.transform
                image_info['bands'][band]['crs'] = str(src.crs)
                image_info['bands'][band]['epsg'] = str(src.crs).upper().replace('EPSG:', '')
                
                # Store the crs also on image level, and check if all bands have the same crs 
                if i == 0:
                    image_info['image_bounds'] = image_info['bands'][band]['bounds']
                    image_info['image_crs'] = image_info['bands'][band]['crs']
                    image_info['image_epsg'] = image_info['bands'][band]['epsg']
                else:
                    if(image_info['image_crs'] != image_info['bands'][band]['crs'] 
                       or image_info['image_epsg'] != image_info['bands'][band]['epsg']):
                        message = f"Not all bands have the same crs for {image_info}"
                        logger.error(message)
                        raise Exception(message)
                
        logger.debug(f"Image metadata read: {image_info}")

    else:
        message = f"Not a supported image format: {image_path}"
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

    # Debug info
    """
    bounds_to_read = rasterio.windows.bounds(window_to_read, image_transform)
    logger.debug(f"projected_bounds: {projected_bounds}, "
                + f"window_to_read_raw: {window_to_read_raw}, window_to_read: {window_to_read}, " 
                + f"image_pixel_width: {image_pixel_width}, image_pixel_height: {image_pixel_height}, "
                + f"file transform: {image_transform}, bounds_to_read: {bounds_to_read}")
    """

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

if __name__ == '__main__':
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
