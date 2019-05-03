# -*- coding: utf-8 -*-
"""
@author: Pieter Roggemans
"""

from datetime import datetime
import glob
import os
import shutil
import sys
[sys.path.append(i) for i in ['.', '..']]

from cropclassification.helpers import log_helper
from cropclassification.preprocess import timeseries_calc_dias_onda as calc_ts

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
    input_image_searchstr = "/mnt/NAS3/CARD/FLANDERS/S1*/L1CO/2018/*/*/*.CARD"
    input_image_filepaths = glob.glob(input_image_searchstr)
    logger.info(f"Found {len(input_image_filepaths)} images to process")
  
    # Temp dir + clean contents from it.
    temp_dir = os.path.join(base_dir, "tmp")
    logger.info(f"Clean the temp dir")
    if os.path.exists(temp_dir):
        # By adding a / at the end, only the contents are recursively deleted
        shutil.rmtree(temp_dir + os.sep)
    
    test = True

    # Output dir 
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
    calc_ts.calc_stats(features_filepath=input_features_filepath,
                       image_paths=input_image_filepaths,
                       bands=['VV', 'VH'],
                       output_dir=output_dir,
                       temp_dir=temp_dir,
                       log_dir=log_dir,
                       stop_on_error=False)

if __name__ == '__main__':
    main()
