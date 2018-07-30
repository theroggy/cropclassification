# -*- coding: utf-8 -*-
"""
Postprocess the classification results.

@author: Pieter Roggemans
"""

import logging
import os
import pandas as pd
import geopandas

# Get a logger...
logger = logging.getLogger(__name__)

def postprocess(input_filepath: str
               ,output_filepath: str
               ,id_columnname: str):
    '''
    '''
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')

    '''
    # Check if parameters are OK and init some extra params
    if not os.path.exists(input_filepath):
        raise Exception("Input file doesn't exist: {}".format(input_filepath))
    else:
        logger.info("Process input file {}".format(input_filepath))

    input_dir = os.path.split(input_filepath)[0]
    input_classes_filepath = os.path.join(input_dir, "MONGROEPEN_20180713.csv")
    if not os.path.exists(input_classes_filepath):
        raise Exception("Input classes file doesn't exist: {}".format(input_classes_filepath))
    else:
        logger.info("Process input class table file {}".format(input_classes_filepath))

    # Create temp dir to store temporary data for tracebility
    output_dir, output_filename = os.path.split(output_parcels_filepath)
    output_filename_noext, output_filename_ext = os.path.splitext(output_filename)
    temp_output_dir = os.path.join(output_dir, 'temp')
    '''

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')