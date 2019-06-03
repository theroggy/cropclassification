# -*- coding: utf-8 -*-
"""
Process a job.
"""

import configparser
import glob 
import os
import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.helpers.config_helper as conf 
import cropclassification.marker_runner as marker_runner 

def run_job():
    
    # First read the general config to get the job and models dir
    conf.read_config(config_filepaths=['config/general.ini', 'config/local_overrule.ini'], year=-1)
    jobs_dir = conf.dirs['job_dir']                                  
    models_dir = conf.dirs['model_dir']                                  
    
    # Get the jobs and treat them
    job_filepaths = glob.glob(os.path.join(jobs_dir, "*.ini"))
    for job_filepath in job_filepaths:      
        # Create configparser and read job file!
        job_config = configparser.ConfigParser(
                interpolation=configparser.ExtendedInterpolation(),
                allow_no_value=True)
        job_config.read(job_filepath)

        # Now get the info we want from the job config
        markertype_to_calc = job_config['job']['markertype_to_calc']
        input_parcel_filename = job_config['job']['input_parcel_filename']
        input_parcel_filetype = job_config['job']['input_parcel_filetype']
        year = job_config['job'].getint('year')
        country_code = job_config['job']['country_code']
        input_groundtruth_filename = job_config['job']['input_groundtruth_filename']
        input_model_to_use_relativepath = job_config['job']['input_model_to_use_relativepath']
        if input_model_to_use_relativepath is not None:
            input_model_to_use_filepath = os.path.join(models_dir, input_model_to_use_relativepath)
        else:
            input_model_to_use_filepath = None

        # Run!
        marker_runner.run(
                markertype_to_calc=markertype_to_calc,
                input_parcel_filename=input_parcel_filename,
                input_parcel_filetype=input_parcel_filetype,
                country_code=country_code,
                year=year,
                input_groundtruth_filename=input_groundtruth_filename,
                input_model_to_use_filepath=input_model_to_use_filepath)

if __name__ == '__main__':
    run_job()