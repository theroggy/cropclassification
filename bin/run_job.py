# -*- coding: utf-8 -*-
"""
Process a job.
"""

import os
import sys
[sys.path.append(i) for i in ['.', '..']]

import configparser 

import cropclassification.helpers.config_helper as conf 
import cropclassification.marker_runner as runner 

def run_job():
    
    # First read the general config to format the full job filepath
    conf.read_config(config_filepaths=['config/general.ini',
                                       'config/local_overrule.ini'], year=None)
    job_dir = conf.dirs['job_dir']                                  
    job_filepath = os.path.join(job_dir, 'job.ini')    

    # Check if job file exists
    if not os.path.exists(job_filepath):
        raise Exception(f"Job file doesn't exist: {job_filepath}")

    job_config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            converters={'list': lambda x: [i.strip() for i in x.split(',')],
                        'listint': lambda x: [int(i.strip()) for i in x.split(',')]})
    job_config.read(job_filepath)
    markertypes_to_calc = job_config['job'].getlist('markertypes_to_calc')
    input_parcel_filename = job_config['job']['input_parcel_filename']
    input_parcel_filetype = job_config['job']['input_parcel_filetype']
    input_groundtruth_filename = job_config['job']['input_groundtruth_filename']
    year = job_config['job'].getint('year')
    country_code = job_config['job']['country_code']

    # Loop over the markertypes to calc and calc!
    for markertype_to_calc in markertypes_to_calc:
        runner.run(markertype_to_calc=markertype_to_calc,
                   input_parcel_filename=input_parcel_filename,
                   input_parcel_filetype=input_parcel_filetype,
                   input_groundtruth_filename=input_groundtruth_filename,
                   country_code=country_code,
                   year=year)
    
if __name__ == '__main__':
    run_job()