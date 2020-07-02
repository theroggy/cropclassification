# -*- coding: utf-8 -*-
"""
Module that manages configuration data.
"""

import configparser
import json
import os
from pathlib import Path
import pprint
from typing import List, Optional

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def read_config(
        config_filepaths: List[Path], 
        default_basedir: Path = None):
            
    # Read the configuration
    global config
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
                        'listfloat': lambda x: [float(i.strip()) for i in x.split(',')],
                        'dict': lambda x: json.loads(x),
                        'path': lambda x: None if x is None else Path(x)},
            allow_no_value=True)

    # Check if all config filepaths are ok
    for config_filepath in config_filepaths:
        if not config_filepath.exists():
            raise Exception(f"Config file doesn't exist: {config_filepath}")

    config.read(config_filepaths)

    # If the data_dir parameter is a relative path, try to resolve it towards 
    # the default basedir of, if specfied.
    data_dir = config['dirs'].getpath('data_dir')
    if not data_dir.is_absolute():
        if default_basedir is None:
            raise Exception(f"Config parameter dirs.data_dir is relative, but no default_basedir supplied!")
        data_dir_absolute = (default_basedir / data_dir).resolve()
        print(f"Config parameter dirs.data_dir was relative, so is now resolved to {data_dir_absolute}")
        config['dirs']['data_dir'] = data_dir_absolute.as_posix()

    # If the marker_basedir parameter is a relative path, try to resolve it towards 
    # the default basedir of, if specfied.
    marker_basedir = config['dirs'].getpath('marker_basedir')
    if not marker_basedir.is_absolute():
        if default_basedir is None:
            raise Exception(f"Config parameter dirs.marker_basedir is relative, but no default_basedir supplied!")
        marker_basedir_absolute = (default_basedir / marker_basedir).resolve()
        print(f"Config parameter dirs.marker_basedir was relative, so is now resolved to {marker_basedir_absolute}")
        config['dirs']['marker_basedir'] = marker_basedir_absolute.as_posix()

    global config_filepaths_used
    config_filepaths_used = config_filepaths

    # Now set global variables to each section as shortcuts    
    global general
    general = config['general']
    global calc_timeseries_params
    calc_timeseries_params = config['calc_timeseries_params']
    global calc_marker_params
    calc_marker_params = config['calc_marker_params']
    global marker
    marker = config['marker']
    global timeseries
    timeseries = config['timeseries']
    global preprocess
    preprocess = config['preprocess']
    global classifier
    classifier = config['classifier']
    global postprocess
    postprocess = config['postprocess']
    global columns
    columns = config['columns']
    global dirs
    dirs = config['dirs']
        
def pformat_config():
    message = f"Config files used: {pprint.pformat(config_filepaths_used)} \n"
    message += "Config info listing:\n"
    message += pprint.pformat({section: dict(config[section]) for section in config.sections()})
    return message
    
def as_dict():
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict