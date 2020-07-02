"""
Helper script to clean all downloaded gee timeseries csv files.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess.timeseries_calc_gee as ts_calc_gee

def main():        
    # Read the configuration
    conf.read_config([Path('config/general.ini')])

    # Go!
    timeseries_periodic_dir = conf.dirs.getpath('timeseries_periodic_dir')
    ts_calc_gee.clean_gee_downloaded_csvs_in_dir(str(timeseries_periodic_dir))
    
if __name__ == '__main__':
    main()

