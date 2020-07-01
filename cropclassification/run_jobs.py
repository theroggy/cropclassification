# -*- coding: utf-8 -*-
"""
Process the jobs in the job directory.
"""

import argparse
import configparser
import glob
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
#if os.name == 'nt':
#    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\gdal"
#    os.environ['PROJ_LIB'] = r"C:\Tools\anaconda3\envs\orthoseg4\Library\share\proj"

def main():
       
    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-j', '--jobdir',
            help='The path to the dir where jobs (*.ini) to be run can be found.')
    # Add back help         
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')
    args = parser.parse_args()

    # If job dir is specified, use it
    if args.jobdir is not None:
        return run_jobs(jobdir=Path(args.jobdir))
    else:
        userjobdir = Path.home() / 'cropclassification' / 'job'
        if userjobdir.exists():
            return run_jobs(jobdir=userjobdir)
        else: 
            print(f"Error: no jobdir specified, and default job dir ({userjobdir}) does not exist, so stop\n")
            parser.print_help()
            sys.exit(1)

def run_jobs(jobdir: Path):
    
    # Get the jobs and treat them
    job_filepaths = sorted(jobdir.glob('*.ini'))
    for job_filepath in job_filepaths:      
        # Create configparser and read job file!
        job_config = configparser.ConfigParser(
                interpolation=configparser.ExtendedInterpolation(),
                allow_no_value=True)
        job_config.read(job_filepath)

        # Now get the info we want from the job config
        action = job_config['job'].get('action', 'calc_marker')

        if action == 'calc_marker':
            from cropclassification import calc_marker 
            calc_marker.calc_marker_job(job_path=job_filepath)
        else:
            raise Exception(f"Action not supported: {action}")

if __name__ == '__main__':
    main()