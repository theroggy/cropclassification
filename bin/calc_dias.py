# -*- coding: utf-8 -*-
"""
Calaculate the timeseries data per image on DIAS.
"""

from datetime import datetime
import glob
import os
import shutil
import sys
[sys.path.append(i) for i in ['.', '..']]

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.preprocess import timeseries_calc_per_image_dias_onda as calc_ts

def main():

    test = False

    # Determine the config files to load depending on the marker_type
    config_filepaths = ["../config/general.ini",
                        "../config/local_overrule_linux.ini"]

    # Specify the date range:
    year = 2018
    month_start = 3
    month_stop = 8

    # Read the configuration files
    conf.read_config(config_filepaths, year=year)

    # Get the general output dir
    input_preprocessed_dir = conf.dirs['input_preprocessed_dir']
    timeseries_per_image_dir = conf.dirs['timeseries_per_image_dir']

    # Init logging
    if not test:
        base_log_dir = conf.dirs['log_dir']
    else:
        base_log_dir = conf.dirs['log_dir'] + '_test'
    log_dir = f"{base_log_dir}{os.sep}calc_dias_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Clean test log dir if it exist
    if test and os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)

    global logger
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    if test:         
        logger.info(f"As we are testing, clean all test logging and use new log_dir: {log_dir}")

    # Write the consolidated config as ini file again to the run dir
    config_used_filepath = os.path.join(log_dir, 'config_used.ini')
    with open(config_used_filepath, 'w') as config_used_file:
        conf.config.write(config_used_file)

    # Get some general config
    columndata_ext = conf.general['columndata_ext']
    rowdata_ext = conf.general['rowdata_ext']
    output_ext = conf.general['output_ext']
    geofile_ext = conf.general['geofile_ext']

    # Input features file depends on the year
    if year == 2017:
        input_features_filename = "Prc_BEFL_2017_2019-06-14_bufm5.shp"
    elif year == 2018:
        input_features_filename = "Prc_BEFL_2018_2019-06-14_bufm5.shp"
    elif year == 2019:
        input_features_filename = "Prc_BEFL_2019_2019-06-25_bufm5.shp"
    else:
        raise Exception(f"Not a valid year: {year}")
    input_features_filepath = os.path.join(input_preprocessed_dir, input_features_filename)
    
    # Init output dir 
    if not test:
        output_basedir = timeseries_per_image_dir
    else:
        output_basedir = timeseries_per_image_dir + '_test'
        logger.info(f"As we are testing, use test output basedir: {output_basedir}")
    input_features_filename_noext = os.path.splitext(input_features_filename)[0]
    output_dir = os.path.join(output_basedir, input_features_filename_noext)
    if test:
        if os.path.exists(output_dir):
            logger.info(f"As we are only testing, clean the output dir: {output_dir}")
            # By adding a / at the end, only the contents are recursively deleted
            shutil.rmtree(output_dir + os.sep)
  
    # Temp dir + clean contents from it.
    temp_dir = conf.dirs['temp_dir'] + os.sep + 'calc_dias'
    logger.info(f"Clean the temp dir {temp_dir}")
    if os.path.exists(temp_dir):
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(temp_dir + os.sep)
        
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

    # Process S1 GRD images
    # -------------------------------   
    input_image_filepaths = []
    for i in range(month_start, month_stop):
        input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S1*/L1TC/{year}/{i:02d}/*/*.CARD"
        input_image_filepaths.extend(glob.glob(input_image_searchstr))
    logger.info(f"Found {len(input_image_filepaths)} S1 GRD images to process")

    if test:
        # Take only the x first images found while testing
        
        #input_image_filepaths = input_image_filepaths[:10]
        input_image_filepaths = []
        input_image_filepaths.append("/mnt/NAS3/CARD/FLANDERS/S1A/L1TC/2018/04/09/S1A_IW_GRDH_1SDV_20180409T054153_20180409T054218_021386_024D13_D824_Orb_RBN_RTN_Cal_TC_20190612T171437.L1TC.CARD")
        input_image_filepaths.append("/mnt/NAS3/CARD/FLANDERS/S1A/L1TC/2018/04/22/S1A_IW_GRDH_1SDV_20180422T173236_20180422T173301_021583_025328_99D1_Orb_RBN_RTN_Cal_TC_20190612T171441.L1TC.CARD")        

        logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

    calc_ts.calc_stats(features_filepath=input_features_filepath,
                       id_column=conf.columns['id'],
                       image_paths=input_image_filepaths,
                       bands=['VV', 'VH'],
                       output_dir=output_dir,
                       temp_dir=temp_dir,
                       log_dir=log_dir)

    # Process S2 images
    # -------------------------------
    input_image_filepaths = []
    for i in range(month_start, month_stop):
        input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S2*/L2A/{year}/{i:02d}/*/*.SAFE"
        input_image_filepaths.extend(glob.glob(input_image_searchstr))    
    logger.info(f"Found {len(input_image_filepaths)} S2 images to process")

    if test:
        # Take only the x first images found while testing
        input_image_filepaths = input_image_filepaths[:10]
        logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

    calc_ts.calc_stats(features_filepath=input_features_filepath,
                       id_column=conf.columns['id'],
                       image_paths=input_image_filepaths,
                       bands=['B02_10m', 'B03_10m', 'B04_10m', 'B08_10m', 'SCL_20m'],
                       output_dir=output_dir,
                       temp_dir=temp_dir,
                       log_dir=log_dir)

    # Process S1 Coherence images
    # -------------------------------   
    input_image_filepaths = []
    for i in range(month_start, month_stop):
        input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S1*/L1CO/{year}/{i:02d}/*/*.CARD"
        input_image_filepaths.extend(glob.glob(input_image_searchstr))  
    logger.info(f"Found {len(input_image_filepaths)} S1 Coherence images to process")

    if test:
        # Take only the x first images found while testing
        input_image_filepaths = input_image_filepaths[:10]
        logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

    calc_ts.calc_stats(features_filepath=input_features_filepath,
                       id_column=conf.columns['id'],
                       image_paths=input_image_filepaths,
                       bands=['VV', 'VH'],
                       output_dir=output_dir,
                       temp_dir=temp_dir,
                       log_dir=log_dir)

if __name__ == '__main__':
    main()
