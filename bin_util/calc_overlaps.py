# -*- coding: utf-8 -*-
"""
Module calculate overlaps between two layers.
"""

import os
# TODO: the init of this doensn't seem to work properly... should be solved somewhere else?
#os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"
import pprint

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
import sys
[sys.path.append(i) for i in ['.', '..']]

import geopandas as gpd
import sqlite3

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.log_helper as log_helper
import cropclassification.helpers.geofile as geofile_helper

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def main():

    # Read the configuration
    segment_config_filepaths=['../config/general.ini', 
                              '../config/local_overrule_linux.ini']
    conf.read_config(segment_config_filepaths, 2018)
    
    # Main initialisation of the logging
    logger = log_helper.main_log_init(conf.dirs['log_dir'], __name__)      
    logger.info("Start")
    logger.info(f"Config used: \n{conf.pformat_config()}")

    logger.info(pprint.pformat(dict(os.environ)))

    # Init variables
    #parcels_filepath = r"X:\GIS\GIS DATA\Percelen_ALP\Vlaanderen\Perc_VL_2019_2019-07-28\perc_2019_met_k_2019-07-28.shp"
    #overlap_filepath = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_10\sealedsurfaces_10_orig.gpkg"
    input_preprocessed_dir = conf.dirs['input_preprocessed_dir']
    parcels_filepath = os.path.join(input_preprocessed_dir, 'Prc_BEFL_2019_2019-07-02_bufm5_32632.gpkg')
    overlap_filepath = os.path.join(input_preprocessed_dir, 'Prc_BEFL_2019_2019-07-02_bufm5_32632.gpkg')
    
    # Read parcels file to memory (isn't that large...)
    #parcels_gpd = geofile_helper.read_file(parcels_filepath)
    
    # Loop over parcels and calculate overlap
    logger.info(f"Connect to {overlap_filepath}")
    conn = sqlite3.connect(overlap_filepath) 
    conn.enable_load_extension(True)

    #now we can load the extension
    # depending on your OS and sqlite/spatialite version you might need to add 
    # '.so' (Linux) or '.dll' (Windows) to the extension name

    #mod_spatialite (recommended)
    #conn.execute("SELECT load_extension('spatialite.dll')")  
    conn.load_extension('mod_spatialite')
    conn.execute('SELECT InitSpatialMetaData(1);')  

    """
    # libspatialite
    conn.execute('SELECT load_extension("libspatialite")')
    conn.execute('SELECT InitSpatialMetaData();')
    """

    c = conn.cursor() 

    c.execute("SELECT sqlite_version()")
    for row in c:
        logger.info(f"test: {row}")

    c.execute("select name from sqlite_master where type = 'table'")
    for row in c:
        logger.info(f"Table: {row}")

    c.execute(
            """SELECT t.uid, t.fid, MbrMinX(t.geom), ST_GeometryType(t.geom), ST_AsText(GeomFromGPB(t.geom))
                 FROM info t
                 JOIN rtree_info_geom r ON t.fid = r.id
                 WHERE r.minx >= 50000
                   AND r.maxx <= 51000
            """)
    """SELECT t.fid, ST_AsText(t.geom)
            FROM info t
            JOIN rtree_info_geom r ON t.fid = r.id
    """

    """SELECT t.fid, AsText(t.geom)
            FROM "default" t
            JOIN rtree_default_geom r ON t.fid = r.id
        WHERE r.minx <= 200000
            AND r.maxx >= 205000
            AND r.miny <= 200000
            AND r.maxy >= 201000
    """

    logger.info(f"test")
    for i, row in enumerate(c):
        logger.info(f"test: {row}")
        if i >= 10:
            break
        # do_stuff_with_row

if __name__ == '__main__':
    main()
