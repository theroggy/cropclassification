# -*- coding: utf-8 -*-
"""
Calculates periodic timeseries for input parcels.
"""

from datetime import datetime
import logging
import gc
import os, shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd

# Import local stuff
import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.geofile as geofile_util
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

IMAGETYPE_S1_GRD = 'S1_GRD'
IMAGETYPE_S1_COHERENCE = 'S1_COH'
IMAGETYPE_S2_L2A = 'S2_L2A'

def prepare_input(input_parcel_filepath: Path,
                  output_imagedata_parcel_input_filepath: Path,
                  output_parcel_nogeo_filepath: Path = None,
                  force: bool = False):
    """
    This function creates a file that is preprocessed to be a good input file for
    timeseries extraction of sentinel images.

    Args
        input_parcel_filepath (Path): input file
        output_imagedata_parcel_input_filepath (Path): prepared output file
        output_parcel_nogeo_filepath (Path): output file with a copy of the non-geo data
        force: force creation, even if output file(s) exist already

    """
    ##### Check if parameters are OK and init some extra params #####
    if not input_parcel_filepath.exists():
        raise Exception(f"Input file doesn't exist: {input_parcel_filepath}")
    
    # Check if the input file has a projection specified
    if geofile_util.get_crs(input_parcel_filepath) is None:
        message = f"The parcel input file doesn't have a projection/crs specified, so STOP: {input_parcel_filepath}"
        logger.critical(message)
        raise Exception(message)

    # If force == False Check and the output file exists already, stop.
    if(force is False 
       and output_imagedata_parcel_input_filepath.exists()
       and (output_parcel_nogeo_filepath is None or output_parcel_nogeo_filepath.exists())):
        logger.warning("prepare_input: force == False and output files exist, so stop: " 
                       + f"{output_imagedata_parcel_input_filepath}, "
                       + f"{output_parcel_nogeo_filepath}")
        return

    logger.info(f"Process input file {input_parcel_filepath}")

    # Create temp dir to store temporary data for tracebility
    temp_output_dir = output_imagedata_parcel_input_filepath.parent / 'temp'
    if not temp_output_dir.exists():
        os.mkdir(temp_output_dir)

    ##### Read the parcel data and write nogeo version #####
    parceldata_gdf = geofile_util.read_file(input_parcel_filepath)
    logger.info(f'Parceldata read, shape: {parceldata_gdf.shape}')

    # Check if the id column is present and set as index
    if conf.columns['id'] in parceldata_gdf.columns:
        parceldata_gdf.set_index(conf.columns['id'], inplace=True)
    else:
        message = f"STOP: Column {conf.columns['id']} not found in input parcel file: {input_parcel_filepath}. Make sure the column is present or change the column name in global_constants.py"
        logger.critical(message)
        raise Exception(message)
        
    if(output_parcel_nogeo_filepath is not None
       and (force is True or not output_parcel_nogeo_filepath.exists())):
        logger.info(f"Save non-geo data to {output_parcel_nogeo_filepath}")
        parceldata_nogeo_df = parceldata_gdf.drop(['geometry'], axis = 1)
        pdh.to_file(parceldata_nogeo_df, output_parcel_nogeo_filepath)

    ##### Do the necessary conversions and write buffered file #####
    
    # If force == False Check and the output file exists already, stop.
    if(force is False 
       and output_imagedata_parcel_input_filepath.exists()):
        logger.warning("prepare_input: force == False and output files exist, so stop: " 
                       + f"{output_imagedata_parcel_input_filepath}")
        return

    logger.info('Apply buffer on parcel')
    parceldata_buf_gdf = parceldata_gdf.copy()

    # resolution = number of segments per circle
    buffer_size = -conf.marker.getint('buffer')
    parceldata_buf_gdf[conf.columns['geom']] = (parceldata_buf_gdf[conf.columns['geom']]
                                                .buffer(buffer_size, resolution=5))

    # Export buffered geometries that result in empty geometries
    logger.info('Export parcels that are empty after buffer')
    parceldata_buf_empty_df = parceldata_buf_gdf.loc[
            parceldata_buf_gdf[conf.columns['geom']].is_empty == True]
    if len(parceldata_buf_empty_df.index) > 0:
        parceldata_buf_empty_df.drop(conf.columns['geom'], axis=1, inplace=True)
        temp_empty_filepath = temp_output_dir / f"{output_imagedata_parcel_input_filepath.stem}_empty.sqlite"
        pdh.to_file(parceldata_buf_empty_df, temp_empty_filepath)

    # Export parcels that don't result in a (multi)polygon
    parceldata_buf_notempty_gdf = parceldata_buf_gdf.loc[
            parceldata_buf_gdf[conf.columns['geom']].is_empty == False]
    parceldata_buf_nopoly_gdf = parceldata_buf_notempty_gdf.loc[
            ~parceldata_buf_notempty_gdf[conf.columns['geom']].geom_type.isin(['Polygon', 'MultiPolygon'])]
    if len(parceldata_buf_nopoly_gdf.index) > 0:
        logger.info('Export parcels that are no (multi)polygons after buffer')
        parceldata_buf_nopoly_gdf.drop(conf.columns['geom'], axis=1, inplace=True)      
        temp_nopoly_filepath = temp_output_dir / f"{output_imagedata_parcel_input_filepath.stem}_nopoly.sqlite"
        geofile_util.to_file(parceldata_buf_nopoly_gdf, temp_nopoly_filepath)

    # Export parcels that are (multi)polygons after buffering
    parceldata_buf_poly_gdf = parceldata_buf_notempty_gdf.loc[
            parceldata_buf_notempty_gdf[conf.columns['geom']].geom_type.isin(['Polygon', 'MultiPolygon'])]
    for column in parceldata_buf_poly_gdf.columns:
        if column not in [conf.columns['id'], conf.columns['geom']]:
            parceldata_buf_poly_gdf.drop(column, axis=1, inplace=True)
    logger.info(f"Export parcels that are (multi)polygons after buffer to {output_imagedata_parcel_input_filepath}")
    geofile_util.to_file(parceldata_buf_poly_gdf, output_imagedata_parcel_input_filepath)
    logger.info(parceldata_buf_poly_gdf)

    message = ("The buffered file just has been prepared, so probably you now you probably need " 
    + "to sync it to the DIAS and start the timeseries data extraction before proceding!")
    logger.warning(message)
    raise Exception(message)

def calculate_periodic_data(
            input_parcel_filepath: Path,
            input_base_dir: Path,
            start_date_str: str,   
            end_date_str: str,    
            sensordata_to_get: List[str],       
            dest_data_dir: Path,
            force: bool = False):
    """
    This function creates a file that is a weekly summarize of timeseries images from DIAS.

    TODO: add possibility to choose which values to extract (mean, min, max,...)?
        
    Args:
        input_parcel_filepath (Path): [description]
        input_base_dir (Path): [description]
        start_date_str (str): Start date in format %Y-%m-%d. Needs to be aligned already on the 
                periods wanted + data on this date is included.
        end_date_str (str): End date in format %Y-%m-%d. Needs to be aligned already on the 
                periods wanted + data on this date is excluded.
        sensordata_to_get ([]): 
        dest_data_dir (Path): [description]
        force (bool, optional): [description]. Defaults to False.
    """
    logger.info('calculate_periodic_data')

    # Init
    input_dir = input_base_dir / input_parcel_filepath.stem

    input_ext = conf.general['data_ext']
    output_ext = conf.general['data_ext']

    id_column = conf.columns['id']
    gdf_input_parcel = geofile_util.read_file(input_parcel_filepath, columns=[id_column])
    nb_input_parcels = len(gdf_input_parcel.index)
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d') 
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d') 
    year = start_date_str.split("-")[0] 

    pixcount_filename = f"{input_parcel_filepath.stem}_weekly_pixcount{output_ext}"
    pixcount_filepath = dest_data_dir / pixcount_filename

    # Prepare output dir
    test = False
    if test is True:
        dest_data_dir = Path(f"{str(dest_data_dir)}_test")
    if not dest_data_dir.exists():
        os.mkdir(dest_data_dir)
    
    # Create Dataframe with all files with their info
    logger.debug('Create Dataframe with all files and their properties')
    file_info_list = []
    for filename in os.listdir(input_dir): 
        if filename.endswith(input_ext):
            # Get seperate filename parts
            file_info = get_file_info(input_dir / filename)
            file_info_list.append(file_info)
    
    all_input_files_df = pd.DataFrame(file_info_list)

    # Loop over the data we need to get
    for sensordata_type in sensordata_to_get:
        start_week = int(datetime.strftime(start_date , '%W'))
        end_week = int(datetime.strftime(end_date , '%W'))

        logger.debug('Get files we need based on start- & stopdates, sensordata_to_get,...')
        if sensordata_type == conf.general['SENSORDATA_S1_ASCDESC']:
            # Filter files to the ones we need
            satellitetype = 'S1'
            imagetype = IMAGETYPE_S1_GRD
            bands = ['VV', 'VH']
            orbits = ['ASC', 'DESC']
            input_files_df = all_input_files_df.loc[(all_input_files_df.date >= start_date) 
                                      & (all_input_files_df.date < end_date)
                                      & (all_input_files_df.imagetype == imagetype)
                                      & (all_input_files_df.band.isin(bands))
                                      & (all_input_files_df.orbit.isin(orbits))]
            calculate_weekly(
                start_week=start_week,
                end_week=end_week,
                year=year,
                sensordata_type=sensordata_type,
                bands=bands,
                orbits=orbits,
                imagetype=imagetype,
                input_files_df=input_files_df,
                input_parcel_filepath=input_parcel_filepath,
                output_pixcount_filepath=pixcount_filepath,
                output_dest_data_dir=dest_data_dir,
                output_ext=output_ext,
                force=force)
        elif sensordata_type == conf.general['SENSORDATA_S2gt95']:
            satellitetype = 'S2'
            imagetype = IMAGETYPE_S2_L2A
            bands = ['B02-10m', 'B03-10m', 'B04-10m', 'B08-10m', 'B11-20m', 'B12-20m'] #conf.timeseries.getlist('s2bands')
            input_files_df = all_input_files_df.loc[(all_input_files_df.date >= start_date) 
                                                         & (all_input_files_df.date < end_date) 
                                                         & (all_input_files_df.imagetype == imagetype)                                                         
                                                         & (all_input_files_df.band.isin(bands))]

            calculate_variable(
                start_week=start_week,
                end_week=end_week,
                year=year,
                sensordata_type=sensordata_type,
                bands=bands,
                orbits=[None],
                imagetype=imagetype,
                nb_input_parcels=nb_input_parcels,
                input_files_df=input_files_df,
                input_parcel_filepath=input_parcel_filepath,
                output_pixcount_filepath=pixcount_filepath,
                output_dest_data_dir=dest_data_dir,
                output_ext=output_ext,
                force=force)

        elif sensordata_type == conf.general['SENSORDATA_S1_COHERENCE']:
            satellitetype = 'S1'
            imagetype = IMAGETYPE_S1_COHERENCE
            bands = ['VV', 'VH']
            orbits = ['ASC', 'DESC']
            input_files_df = all_input_files_df.loc[(all_input_files_df.date >= start_date) 
                                                         & (all_input_files_df.date < end_date) 
                                                         & (all_input_files_df.imagetype == imagetype)                                                         
                                                         & (all_input_files_df.band.isin(bands))]
            calculate_weekly(
                start_week=start_week,
                end_week=end_week,
                year=year,
                sensordata_type=sensordata_type,
                bands=bands,
                orbits=orbits,
                imagetype=imagetype,
                input_files_df=input_files_df,
                input_parcel_filepath=input_parcel_filepath,
                output_pixcount_filepath=pixcount_filepath,
                output_dest_data_dir=dest_data_dir,
                output_ext=output_ext,
                force=force)
        else:
            raise Exception(f"Unsupported sensordata_type: {sensordata_type}")

def calculate_weekly(
    start_week: int,
    end_week: int,
    year: str,
    sensordata_type: str,
    bands: List[str],
    orbits: List[str],
    imagetype: str,
    input_files_df: pd.DataFrame,
    input_parcel_filepath: Path,
    output_pixcount_filepath: Path,
    output_dest_data_dir: Path,
    output_ext: str,
    force: bool = False):
    for period_index in range(start_week, end_week):
        # Get the date of the first day of period period_index (eg. monday for a week)
        period_date = get_monday_from_week(year, period_index)

        # New file name
        period_date_str_long = period_date.strftime('%Y-%m-%d')
        period_data_filename = f"{input_parcel_filepath.stem}_weekly_{period_date_str_long}_{sensordata_type}{output_ext}"
        period_data_filepath = output_dest_data_dir / period_data_filename

        period_data_df = calculate_period(
            period_date=period_date,
            period_index=period_index,
            period_data_filepath=period_data_filepath,
            imagetype=imagetype,
            bands=bands, 
            orbits=orbits,
            input_files_df=input_files_df,
            output_pixcount_filepath=output_pixcount_filepath,
            force=force)

        if period_data_df is not None:
            logger.info(f"Write new file: {period_data_filepath.stem}")
            pdh.to_file(period_data_df, period_data_filepath)

            # Create pixcount file if it doesn't exist yet...
            if not output_pixcount_filepath.exists():
                pixcount_s1s2_column = conf.columns['pixcount_s1s2']
                
                # Get max count of all count columns available 
                columns_to_use = [column for column in period_data_df.columns if column.endswith('_count')]
                period_data_df[pixcount_s1s2_column] = np.nanmax(period_data_df[columns_to_use], axis=1)
                
                pixcount_df = period_data_df[pixcount_s1s2_column]
                pixcount_df.fillna(value=0, inplace=True)

                pdh.to_file(pixcount_df, output_pixcount_filepath)

def calculate_variable(
    start_week: int,
    end_week: int,
    year: str,
    sensordata_type: str,
    bands: List[str],
    orbits: List[str],
    imagetype: str,
    nb_input_parcels: int,
    input_files_df: pd.DataFrame,
    input_parcel_filepath: Path,
    output_pixcount_filepath: Path,
    output_dest_data_dir: Path,
    output_ext: str,
    force: bool = False):

    max_sliding_window = 3
    id_column = conf.columns['id']
    min_parcels_with_data_pct = conf.timeseries.getfloat('min_parcels_with_data_pct')
    temp_dir = output_dest_data_dir / f"{input_parcel_filepath.stem}"
    to_process_indices: List[int] = list(range(start_week, end_week))
    sliding_indices: List[int] = []
    
    def getTempFileName(index: int):
        return temp_dir / f"{input_parcel_filepath.stem}_{index}_{sensordata_type}{output_ext}"

    if (not temp_dir.exists()):
        os.mkdir(temp_dir)
    
    while len(to_process_indices) > 0:
        current_index = to_process_indices.pop(0)
        
        if (len(sliding_indices) > 0 and sliding_indices[0] + max_sliding_window <= current_index):
            logger.info(f"Sliding the window..")
            sliding_indices.pop(0)

        logger.info(f"Starting calculation for period: {current_index}")
        # Get the date of the first day of period period_index (eg. monday for a week)
        
        # NOTE: cant change file name to period because we *compare* with it later
        period_start = get_monday_from_week(year, current_index)
        period_date_str_long = period_start.strftime('%Y-%m-%d') 
        period_data_filename = f"{input_parcel_filepath.stem}_weekly_{period_date_str_long}_{sensordata_type}{output_ext}" 
        period_data_filepath = output_dest_data_dir / period_data_filename

        if (period_data_filepath.exists() and not force):
            logger.info(f"SKIP: force is False and file exists: {period_data_filepath}")
            sliding_indices = [] # reset
            continue

        period_data_temp_filename = getTempFileName(current_index) 
        if (period_data_temp_filename.exists() and not force): 
            period_data_df = pdh.read_file(period_data_temp_filename)
        else:
            period_data_df = calculate_period(
                    period_date=period_start,
                    period_index=current_index,
                    period_data_filepath=period_data_temp_filename,
                    imagetype=imagetype,
                    bands=bands, 
                    orbits=orbits,
                    input_files_df=input_files_df,
                    output_pixcount_filepath=output_pixcount_filepath,
                    prefix_columns=False,
                    force=force)

        if period_data_df is not None:
            data_available_pct = len(period_data_df.index)*100/nb_input_parcels
            if data_available_pct < min_parcels_with_data_pct:
                logger.info(f"Not enough data found: {data_available_pct} < {min_parcels_with_data_pct}")

                if (not period_data_temp_filename.exists()):
                    pdh.to_file(period_data_df, period_data_temp_filename)

                # try to bundle previous incomplete data
                if len(sliding_indices) > 0:
                    logger.info(f"Trying with a larger window now {current_index} + {sliding_indices[::-1]}")

                    for index in sliding_indices[::-1]:
                        previous_filename = getTempFileName(index) 
                        previous_df = pdh.read_file(previous_filename)
                        previous_df.set_index(id_column, inplace=True)
                        period_data_df = period_data_df.combine_first(previous_df) # keep the latest data, only add missing data/rows

                    data_available_pct = len(period_data_df.index)*100/nb_input_parcels
                    sliding_indices.append(current_index)

                    if data_available_pct < min_parcels_with_data_pct:
                        logger.info(f"Merge didnt give enough data: {data_available_pct} < {min_parcels_with_data_pct}.")
                        logger.info("Trying again later.")
                        continue
                    
                    logger.info(f"Enough data found for {sliding_indices}: {data_available_pct} < {min_parcels_with_data_pct}")
                else: 
                    logger.info(f"Trying again later.")
                    sliding_indices.append(current_index)
                    continue

            sliding_indices = [] # clean up the previous incomplete data.. it's now useless.
            period_data_df.columns = period_data_df.columns.str.replace('TS_', f'TS_{period_date_str_long}_')

            # Write result to file
            logger.info(f"Write new file: {period_data_filepath.stem}")
            pdh.to_file(period_data_df, period_data_filepath)

            # Create pixcount file if it doesn't exist yet...
            if not output_pixcount_filepath.exists():
                pixcount_s1s2_column = conf.columns['pixcount_s1s2']
                
                # Get max count of all count columns available 
                columns_to_use = [column for column in period_data_df.columns if column.endswith('_count')]
                period_data_df[pixcount_s1s2_column] = np.nanmax(period_data_df[columns_to_use], axis=1)
                
                pixcount_df = period_data_df[pixcount_s1s2_column]
                pixcount_df.fillna(value=0, inplace=True)

                pdh.to_file(pixcount_df, output_pixcount_filepath)
        else:
            # If there is no output, there is nothing to merge/append, so just continue
            logger.info(f"No data found for period {current_index}, skipping..")
            
    # clean up temp files
    shutil.rmtree(temp_dir)
        
def calculate_period(
        period_date: datetime,
        period_index: int,
        output_pixcount_filepath: Path,
        period_data_filepath: Path,
        imagetype: str,
        bands: List[str],
        orbits: List[str],
        input_files_df: pd.DataFrame,
        prefix_columns: bool = True,
        force: bool = False): 
    # Check if output file exists already
    if period_data_filepath.exists() and output_pixcount_filepath.exists():
        if force is False:
            logger.info(f"SKIP: force is False and file exists: {period_data_filepath}")
            return
        else:
            os.remove(period_data_filepath)

    # Loop over bands and orbits (all combinations of bands and orbits!)
    logger.info(f"Calculate file: {period_data_filepath.stem}")
    period_data_df = None
    id_column = conf.columns['id']
    gc.collect()                  # Try to evade memory errors
    for band, orbit in [(band, orbit) for band in bands for orbit in orbits]:

        # Get list of files needed for this period, band
        period_files_df = input_files_df.loc[(input_files_df.week == period_index)
                                                    & (input_files_df.band == band)]
        
        # If an orbit to be filtered was specified, filter
        if orbit is not None:
            period_files_df = period_files_df.loc[(period_files_df.orbit == orbit)]

        if len(period_files_df) == 0:
            logger.warn("No input files found!")

        # Loop all period_files
        period_band_data_df = None
        statistic_columns_dict = {'count': [], 'max': [], 'mean': [], 'median': [], 'min': [], 'std': []}
        for j, imagedata_filepath in enumerate(period_files_df.filepath.tolist()):
            
            # If file has filesize == 0, skip
            imagedata_filepath = Path(imagedata_filepath)
            if imagedata_filepath.stat().st_size == 0:
                continue 

            # Read the file (but only the columns we need)
            columns = [column for column in statistic_columns_dict]
            columns.append(id_column)

            image_data_df = pdh.read_file(imagedata_filepath, columns=columns)
            image_data_df.set_index(id_column, inplace=True)
            image_data_df.index.name = id_column

            # Remove rows with nan values
            nb_before_dropna = len(image_data_df.index)
            image_data_df.dropna(inplace=True)
            nb_after_dropna = len(image_data_df.index)
            if nb_after_dropna != nb_before_dropna:
                logger.warning(f"Before dropna: {nb_before_dropna}, after: {nb_after_dropna} for file {imagedata_filepath}")
            if nb_after_dropna == 0:
                continue
            
            # recalculate duplicate rows (the -5 buffer can cause break ups?)
            image_data_recalculate_df = image_data_df.loc[image_data_df.index.duplicated()].groupby(id_column).agg({column: "mean" for column in statistic_columns_dict})
            image_data_df = image_data_df.loc[~image_data_df.index.duplicated()]
            image_data_df.append(image_data_recalculate_df)

            # Rename columns so column names stay unique
            for statistic_column in statistic_columns_dict:
                new_column_name = statistic_column + str(j+1)
                image_data_df.rename(columns={statistic_column: new_column_name},
                                        inplace=True)
                image_data_df[new_column_name] = image_data_df[new_column_name].astype(float)
                statistic_columns_dict[statistic_column].append(new_column_name)
                                    
            # Create 1 dataframe for all weekfiles - one row for each code_obj - using concat (code_obj = index)
            if period_band_data_df is None:
                period_band_data_df = image_data_df                
            else:
                period_band_data_df = pd.concat([period_band_data_df, image_data_df], axis=1, sort=False)
                # Apparently concat removes the index name in some situations
                period_band_data_df.index.name = id_column
                
        # Calculate max, mean, min, ...
        if period_band_data_df is not None:
            logger.debug('Calculate max, mean, min, ...')
            period_date_str_short = period_date.strftime('%Y%m%d')
            # Remark: prefix column names: sqlite doesn't like a numeric start
            column_basename = f"TS_{period_date_str_short}_" if prefix_columns else "TS_"

            if orbit is None:
                column_basename = f"{column_basename}{imagetype}_{band}_"
            else:
                column_basename = f"{column_basename}{imagetype}_{orbit}_{band}_"

            # Number of pixels
            # TODO: onderzoeken hoe aantal pixels best bijgehouden wordt : afwijkingen weglaten ? max nemen ? ...
            period_band_data_df[f"{column_basename}count"] = np.nanmax(period_band_data_df[statistic_columns_dict['count']], axis=1)
            # Maximum of all max columns
            period_band_data_df[f"{column_basename}max"] = np.nanmax(period_band_data_df[statistic_columns_dict['max']], axis=1)
            # Mean of all mean columns
            period_band_data_df[f"{column_basename}mean"] = np.nanmean(period_band_data_df[statistic_columns_dict['mean']], axis=1)
            # Mean of all median columns
            period_band_data_df[f"{column_basename}median"] = np.nanmean(period_band_data_df[statistic_columns_dict['median']], axis=1)
            # Minimum of all min columns
            period_band_data_df[f"{column_basename}min"] = np.nanmin(period_band_data_df[statistic_columns_dict['min']], axis=1)
            # Mean of all std columns
            period_band_data_df[f"{column_basename}std"] = np.nanmean(period_band_data_df[statistic_columns_dict['std']], axis=1)
            # Number of Files used
            period_band_data_df[f"{column_basename}used_files"] = period_band_data_df[statistic_columns_dict['max']].count(axis=1)
                            
            # Only keep the columns we want to keep
            columns_to_keep = [f"{column_basename}count", f"{column_basename}max", 
                            f"{column_basename}mean", f"{column_basename}median", f"{column_basename}min", 
                            f"{column_basename}std", f"{column_basename}used_files"] 
            period_band_data_df = period_band_data_df[columns_to_keep]

            # Merge the data with the other bands/orbits for this period
            if period_data_df is None:
                period_data_df = period_band_data_df
            else:
                period_data_df = pd.concat([period_band_data_df, period_data_df], axis=1, sort=False) 
                # Apparently concat removes the index name in some situations
                period_data_df.index.name = id_column
            
    return period_data_df

def get_file_info(filepath: Path) -> dict:
    """
    This function gets info of a timeseries data file.
    
    Args:
        filepath (Path): The filepath to the file to get info about.
        
    Returns:
        dict: a dict containing info about the file
    """

    try:
        # Split name on parcelinfo versus imageinfo
        filename_splitted = filepath.stem.split("__")
        filename_parcelinfo = filename_splitted[0]
        filename_imageinfo = filename_splitted[1]

        # Extract imageinfo
        imageinfo_values = filename_imageinfo.split("_")    

        # Satellite 
        satellite = imageinfo_values[0]

        # Get the date taken from the filename, depending on the satellite type
        # Remark: the datetime is in this format: '20180101T055812'
        imagetype = None
        filedatetime = None
        if satellite.startswith('S1'):
            # Check if it is a GRDH image
            if imageinfo_values[2] == 'GRDH':
                imagetype = IMAGETYPE_S1_GRD
                filedatetime = imageinfo_values[4]  
            elif imageinfo_values[1].startswith('S1'):
                imagetype = IMAGETYPE_S1_COHERENCE
                filedatetime = imageinfo_values[2]  
        elif satellite.startswith('S2'):
            imagetype = IMAGETYPE_S2_L2A
            filedatetime = imageinfo_values[2]  

        if(imagetype is None or filedatetime is None):
            raise Exception(f"Unsupported file: {filepath}")
        
        filedate = filedatetime.split("T")[0]  
        parseddate = datetime.strptime(filedate, '%Y%m%d') 
        fileweek = int(parseddate.strftime('%W'))

        # Get the band 
        fileband = imageinfo_values[-1]         # =last value

        # For S1 images, get the orbit 
        if satellite.startswith('S1'):
            fileorbit = imageinfo_values[-2]    # =2nd last value
        else:
            fileorbit = None

        # The file paths of these files sometimes are longer than 256
        # characters, so use trick on windows to support this anyway
        filepath_safe = filepath.as_posix()
        if(os.name == 'nt'
           and len(filepath.as_posix()) > 240):
            if filepath_safe.startswith("//"):
                filepath_safe = f"//?/UNC/{filepath_safe}"
            else:
                filepath_safe = f"//?/{filepath_safe}"

        filenameparts = {
                'filepath': filepath_safe,
                'imagetype': imagetype,
                'filestem' : filepath.stem,
                'date' : parseddate,
                'week' : fileweek,
                'band' : fileband, 
                'orbit' : fileorbit} # ASC/DESC

    except Exception as ex:
        message = f"Error extracting info from filename {filepath}"
        logger.exception(message)
        raise Exception(message) from ex

    return filenameparts

def get_monday(input_date: str) -> datetime:
    """
    This function gets the first monday before the date provided.
    She is being used to adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
       - inputformat:  %Y-%m-%d
       - outputformat: datetime
    """
    parseddate = datetime.strptime(input_date, '%Y-%m-%d')
    year_week = parseddate.strftime('%Y_%W')
    year_week_monday = datetime.strptime(year_week + '_1', '%Y_%W_%w') 
    return year_week_monday

def get_monday_from_week(year: str, week: int) -> datetime:
    return datetime.strptime(str(year) + '_' + str(week) + '_1', '%Y_%W_%w')

# If the script is run directly...
if __name__ == "__main__":
    raise Exception("Not implemented")
    