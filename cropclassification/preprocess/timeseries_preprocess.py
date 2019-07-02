# -*- coding: utf-8 -*-
"""
Create an input file for the sentinel timeseries processing.
"""

import logging
import os
import geopandas as gpd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.geofile as geofile_util
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_input(input_parcel_filepath: str,
                  output_imagedata_parcel_input_filepath: str,
                  output_parcel_nogeo_filepath: str = None,
                  force: bool = False):
    """
    This function creates a file that is preprocessed to be a good input file for
    timeseries extraction of sentinel images.

    Args
        input_parcel_filepath: input file
        output_imagedata_parcel_input_filepath: prepared output file
        output_parcel_nogeo_filepath: output file with a copy of the non-geo data
        force: force creation, even if output file(s) exist already

    """
    ##### Check if parameters are OK and init some extra params #####
    if not os.path.exists(input_parcel_filepath):
        raise Exception(f"Input file doesn't exist: {input_parcel_filepath}")
    
    # Check if the input file has a projection specified
    if geofile_util.get_crs(input_parcel_filepath) is None:
        message = f"The parcel input file doesn't have a projection/crs specified, so STOP: {input_parcel_filepath}"
        logger.critical(message)
        raise Exception(message)

    # If force == False Check and the output file exists already, stop.
    if(force is False 
       and os.path.exists(output_imagedata_parcel_input_filepath)
       and (output_parcel_nogeo_filepath is None or os.path.exists(output_parcel_nogeo_filepath))):
        logger.warning("prepare_input: force == False and output files exist, so stop: " 
                       + f"{output_imagedata_parcel_input_filepath}, "
                       + f"{output_parcel_nogeo_filepath}")
        return

    logger.info(f"Process input file {input_parcel_filepath}")

    # Create temp dir to store temporary data for tracebility
    output_dir, output_filename = os.path.split(output_imagedata_parcel_input_filepath)
    output_filename_noext = os.path.splitext(output_filename)[0]
    temp_output_dir = os.path.join(output_dir, 'temp')
    if not os.path.exists(temp_output_dir):
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
        
    if force is True or os.path.exists(output_parcel_nogeo_filepath) == False:
        logger.info(f"Save non-geo data to {output_parcel_nogeo_filepath}")
        parceldata_nogeo_df = parceldata_gdf.drop(['geometry'], axis = 1)
        pdh.to_file(parceldata_nogeo_df, output_parcel_nogeo_filepath)

    ##### Do the necessary conversions and write buffered file #####
    
    # If force == False Check and the output file exists already, stop.
    if(force is False 
       and os.path.exists(output_imagedata_parcel_input_filepath)):
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
        temp_empty_filepath = os.path.join(temp_output_dir, f"{output_filename_noext}_empty.sqlite")
        pdh.to_file(parceldata_buf_empty_df, temp_empty_filepath)

    # Export parcels that don't result in a (multi)polygon
    parceldata_buf_notempty_gdf = parceldata_buf_gdf.loc[
            parceldata_buf_gdf[conf.columns['geom']].is_empty == False]
    parceldata_buf_nopoly_gdf = parceldata_buf_notempty_gdf.loc[
            ~parceldata_buf_notempty_gdf[conf.columns['geom']].geom_type.isin(['Polygon', 'MultiPolygon'])]
    if len(parceldata_buf_nopoly_gdf.index) > 0:
        logger.info('Export parcels that are no (multi)polygons after buffer')
        parceldata_buf_nopoly_gdf.drop(conf.columns['geom'], axis=1, inplace=True)      
        temp_nopoly_filepath = os.path.join(temp_output_dir, f"{output_filename_noext}_nopoly.sqlite")
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

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
