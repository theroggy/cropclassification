# -*- coding: utf-8 -*-
"""
Create an input file for the sentinel timeseries processing.

@author: Pieter Roggemans
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
                  output_imagedata_parcel_input_4326_filepath: str = None,
                  output_parcel_nogeo_filepath: str = None,
                  force: bool = False):
    """
    This function creates a file that is preprocessed to be a good input file for
    timeseries extraction of sentinel images.

    Args
        input_parcel_filepath: input file
        output_imagedata_parcel_input_filepath: prepared output file
        output_imagedata_parcel_input_4326_filepath: prepared output file reprojected to 4326
        output_parcel_nogeo_filepath: output file with a copy of the non-geo data
        force: force creation, even if output file(s) exist already

    """

    # TODO: recently, the assets uploaded to GEE that aren't in EPSG:4326 are very bad quality,
    #       so reprojection should be added here...

    # If force == False Check and the output file exists already, stop.
    if(force is False 
       and os.path.exists(output_imagedata_parcel_input_filepath)
       and (output_imagedata_parcel_input_4326_filepath is None or os.path.exists(output_imagedata_parcel_input_4326_filepath))
       and (output_parcel_nogeo_filepath is None or os.path.exists(output_parcel_nogeo_filepath))):
        logger.warning("prepare_input: force == False and output files exist, so stop: " +
                f"{output_imagedata_parcel_input_filepath}, {output_imagedata_parcel_input_4326_filepath}, {output_parcel_nogeo_filepath}")
        return

    # Check if parameters are OK and init some extra params
    if not os.path.exists(input_parcel_filepath):
        raise Exception(f"Input file doesn't exist: {input_parcel_filepath}")
    
    # Check if the input file has a projection specified
    if geofile_util.get_crs(input_parcel_filepath) is None:
        message = f"The parcel input file doesn't have a projection/crs specified, so STOP: {input_parcel_filepath}"
        logger.critical(message)
        raise Exception(message)

    logger.info(f"Process input file {input_parcel_filepath}")

    # Create temp dir to store temporary data for tracebility
    output_dir, output_filename = os.path.split(output_imagedata_parcel_input_filepath)
    output_filename_noext = os.path.splitext(output_filename)[0]
    temp_output_dir = os.path.join(output_dir, 'temp')
    if not os.path.exists(temp_output_dir):
        os.mkdir(temp_output_dir)

    # Create a version with buffered geometries, to evade mixels
    temp_nopoly_filepath = os.path.join(temp_output_dir, f"{output_filename_noext}_nopoly.csv")
    temp_empty_filepath = os.path.join(temp_output_dir, f"{output_filename_noext}_empty.csv")

    # Read the parcel data and do the necessary conversions
    #--------------------------------------------------------------------------
    parceldata_gdf = geofile_util.read_file(input_parcel_filepath)
    logger.info(f'Parceldata read, shape: {parceldata_gdf.shape}')

    # Check if the id column is present...
    if conf.columns['id'] not in parceldata_gdf.columns:
        message = f"STOP: Column {conf.columns['id']} not found in input parcel file: {input_parcel_filepath}. Make sure the column is present or change the column name in global_constants.py"
        logger.critical(message)
        raise Exception(message)
        
    if force is True or os.path.exists(output_parcel_nogeo_filepath) == False:
        logger.info(f"Save non-geo data to {output_parcel_nogeo_filepath}")
        parceldata_nogeo_df = parceldata_gdf.drop(['geometry'], axis = 1)
        pdh.to_file(parceldata_nogeo_df, output_parcel_nogeo_filepath)

    logger.info('Apply buffer on parcel')
    parceldata_buf = parceldata_gdf.copy()

    # resolution = number of segments per circle
    parceldata_buf[conf.columns['geom']] = parceldata_buf[conf.columns['geom']].buffer(-conf.marker.getint('buffer'), resolution=5)

    # Export buffered geometries that result in empty geometries
    logger.info('Export parcel that are empty after buffer')
    parceldata_buf_empty = parceldata_buf[parceldata_buf[conf.columns['geom']].is_empty == True]
    parceldata_buf_empty.to_csv(temp_empty_filepath, index=False)

    # Export buffered geometries that don't result in polygons
    logger.info('Export parcel that are no (multi)polygons after buffer')
    parceldata_buf_notempty = parceldata_buf[parceldata_buf[conf.columns['geom']].is_empty == False]
    parceldata_buf_nopoly = parceldata_buf_notempty[-parceldata_buf_notempty[conf.columns['geom']]
                                                    .geom_type.isin(['Polygon', 'MultiPolygon'])]
    parceldata_buf_nopoly.to_csv(temp_nopoly_filepath, index=False)

    # Continue processing on the (multi)polygons
    logger.info('Export parcel that are (multi)polygons after buffer')
    parceldata_buf_poly = parceldata_buf_notempty[parceldata_buf_notempty[conf.columns['geom']]
                                                  .geom_type.isin(['Polygon', 'MultiPolygon'])]

    # Removeall columns except ID
    for column in parceldata_buf_poly.columns:
        if column not in [conf.columns['id'], conf.columns['geom']]:
            parceldata_buf_poly.drop(column, axis=1, inplace=True)
    logger.debug(f"parcel that are (multi)polygons, shape: {parceldata_buf_poly.shape}")

    # Check crs of the input file.  is WGS84 (epsg:4326), if not, create reprojected version as well.
    if output_imagedata_parcel_input_4326_filepath is not None:            
        target_epsg = 4326
        logger.info(f"Reproject features from {parceldata_buf_poly.crs} to epsg:{target_epsg}")
        parceldata_buf_poly_4326 = parceldata_buf_poly.to_crs(epsg=target_epsg)
        logger.info(f"Write reprojected features to {output_imagedata_parcel_input_4326_filepath}")
        parceldata_buf_poly_4326.to_file(output_imagedata_parcel_input_4326_filepath)

    logger.info(f"Write buffered features to {output_imagedata_parcel_input_filepath}")
    parceldata_buf_poly.to_file(output_imagedata_parcel_input_filepath)

    # If the needed to be created... it didn't exist yet and so it needs to be uploaded manually
    # to gee as an asset...
    raise Exception(f"The parcel file needs to be uploaded to GEE manually as an asset: {output_imagedata_parcel_input_filepath}")

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
