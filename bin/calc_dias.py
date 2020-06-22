# -*- coding: utf-8 -*-
"""
Calaculate the timeseries data per image on DIAS.
"""

import datetime
import glob
import os
import shutil
import sys
[sys.path.append(i) for i in ['.', '..']]

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"
    os.environ['PROJ_LIB'] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\proj"
    
from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.preprocess import timeseries_calc_dias_onda_per_image as calc_ts

def main():

    # Determine the config files to load depending on the marker_type
    config_filepaths = ["../config/general.ini",
                        "../config/local_overrule_linux.ini"]

    test = False

# Specify the calculations
    calculations = []
    calculation = {}
    #calculation['parcel_year'] = 2017
    #calculation['calc_date_start'] = datetime.date(2017, 5, 1)
    #calculation['calc_date_stop'] = datetime.date(2018, 3, 1)
    #calculations.append(calculation)

    calculation = {}
    calculation['parcel_year'] = 2020
    calculation['calc_date_start'] = datetime.date(2020, 3, 15)
    calculation['calc_date_stop'] = datetime.date(2020, 8, 18)
    calculations.append(calculation)

    for calculation in calculations:

        calc_year_start = calculation['calc_date_start'].year
        calc_year_stop = calculation['calc_date_stop'].year
        calc_month_start = calculation['calc_date_start'].month
        calc_month_stop = calculation['calc_date_stop'].month

        parcel_year = calculation['parcel_year']

        # Read the configuration files
        conf.read_config(config_filepaths, year=calc_year_start)

        # Get the general output dir
        input_preprocessed_dir = conf.dirs['input_preprocessed_dir']
        timeseries_per_image_dir = conf.dirs['timeseries_per_image_dir']

        # Init logging
        if not test:
            base_log_dir = conf.dirs['log_dir']
        else:
            base_log_dir = conf.dirs['log_dir'] + '_test'
        log_dir = f"{base_log_dir}{os.sep}calc_dias_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

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

        # Input features file depends on the year
        if parcel_year == 2017:
            input_features_filename = "Prc_BEFL_2017_2019-06-14_bufm5.shp"
        elif parcel_year == 2018:
            input_features_filename = "Prc_BEFL_2018_2019-06-14_bufm5.shp"
        elif parcel_year == 2019:
            #input_features_filename = "Prc_BEFL_2019_2019-06-25_bufm5.shp"
            input_features_filename = "Prc_BEFL_2019_2019-08-14_bufm5.shp"
        elif parcel_year == 2020:
            input_features_filename = "Prc_BEFL_2020_2020-06-09_bufm5.shp"
        else:
            raise Exception(f"Not a valid parcel_year: {parcel_year}")
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

        ##### Process S1 GRD images #####
        input_image_filepaths = []
        for year in range(calc_year_start, calc_year_stop+1):
            # TODO: works, but doesn't seem to be the most elegant code...
            if year < calc_year_stop:
                month_stop = 12
            else:
                month_stop = calc_month_stop
            if year > calc_year_start:
                month_start = 1
            else:
                month_start = calc_month_start
            for month in range(calc_month_start, calc_month_stop+1):
                input_image_searchstr = f"/mnt/NAS*/CARD/FLANDERS/S1*/L1TC/{year}/{month:02d}/*/*.CARD"
                input_image_filepaths.extend(glob.glob(input_image_searchstr))
        logger.info(f"Found {len(input_image_filepaths)} S1 GRD images to process")

        if test:
            # Take only the x first images found while testing
            input_image_filepaths = input_image_filepaths[:10]
            logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

        calc_ts.calc_stats_per_image(
                features_filepath=input_features_filepath,
                id_column=conf.columns['id'],
                image_paths=input_image_filepaths,
                bands=['VV', 'VH'],
                output_dir=output_dir,
                temp_dir=temp_dir,
                log_dir=log_dir)

        ##### Process S2 images #####
        input_image_filepaths = []
        for year in range(calc_year_start, calc_year_stop+1):
            # TODO: works, but doesn't seem to be the most elegant code...
            if year < calc_year_stop:
                month_stop = 12
            else:
                month_stop = calc_month_stop
            if year > calc_year_start:
                month_start = 1
            else:
                month_start = calc_month_start
            for month in range(month_start, month_stop+1):
                input_image_searchstr = f"/mnt/NAS*/CARD/FLANDERS/S2*/L2A/{year}/{month:02d}/*/*.SAFE"
                logger.info(f"Search for {input_image_searchstr}")
                input_image_filepaths.extend(glob.glob(input_image_searchstr))    
        logger.info(f"Found {len(input_image_filepaths)} S2 images to process")

        if test:
            # Take only the x first images found while testing
            input_image_filepaths = input_image_filepaths[:10]
            logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

        # TODO: refactor underlying code so the SCL band is used regardless of it being passed here
        max_cloudcover_pct = conf.timeseries.getfloat('max_cloudcover_pct')
        calc_ts.calc_stats_per_image(
                features_filepath=input_features_filepath,
                id_column=conf.columns['id'],
                image_paths=input_image_filepaths,
                bands=['B02-10m', 'B03-10m', 'B04-10m', 'B08-10m', 'SCL-20m'],
                output_dir=output_dir,
                temp_dir=temp_dir,
                log_dir=log_dir,
                max_cloudcover_pct=max_cloudcover_pct)

        ##### Process S1 Coherence images #####   
        input_image_filepaths = []
        for year in range(calc_year_start, calc_year_stop+1):
            # TODO: works, but doesn't seem to be the most elegant code...
            if year < calc_year_stop:
                month_stop = 12
            else:
                month_stop = calc_month_stop
            if year > calc_year_start:
                month_start = 1
            else:
                month_start = calc_month_start
            for month in range(calc_month_start, calc_month_stop+1):
                input_image_searchstr = f"/mnt/NAS*/CARD/FLANDERS/S1*/L1CO/{year}/{month:02d}/*/*.CARD"
                input_image_filepaths.extend(glob.glob(input_image_searchstr))  
        logger.info(f"Found {len(input_image_filepaths)} S1 Coherence images to process")

        if test:
            # Take only the x first images found while testing
            input_image_filepaths = input_image_filepaths[:10]
            logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

        calc_ts.calc_stats_per_image(features_filepath=input_features_filepath,
                id_column=conf.columns['id'],
                image_paths=input_image_filepaths,
                bands=['VV', 'VH'],
                output_dir=output_dir,
                temp_dir=temp_dir,
                log_dir=log_dir)

if __name__ == '__main__':
    main()
