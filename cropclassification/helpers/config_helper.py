# -*- coding: utf-8 -*-
"""
Module that manages configuration data.
"""

import configparser
import os
import pprint

def read_config(config_filepaths: [], 
                year: int):
            
    # Read the configuration
    global config
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                       converters={'list': lambda x: [i.strip() for i in x.split(',')],
                                                   'listint': lambda x: [int(i.strip()) for i in x.split(',')]})

    # Check if all config filepaths are ok
    for config_filepath in config_filepaths:
        if not os.path.exists(config_filepath):
            raise Exception(f"Config file doesn't exist: {config_filepath}")

    config.read(config_filepaths)

    # If the year is specified in parameter, set it.
    if year is not None:
        config['marker']['year'] = str(year)
    else:
        print("WARNING: the year passed is None, this can result in some parameters giving errors")

    global config_filepaths_used
    config_filepaths_used = config_filepaths

    # Now set global variables to each section as shortcuts    
    global general
    general = config['general']
    global marker
    marker = config['marker']
    global columns
    columns = config['columns']
    global classifier
    classifier = config['classifier']
    global preprocess
    preprocess = config['preprocess']
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