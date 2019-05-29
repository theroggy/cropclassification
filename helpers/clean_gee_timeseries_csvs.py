"""
Helper script to clean all downloaded gee timeseries csv files.
"""

import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.helpers.config_helper as conf
import cropclassification.preprocess.timeseries_calc_gee as ts_calc_gee

def main():        
    # Read the configuration
    conf.read_config(['config/general.ini',
                      'config/local_overrule.ini'])

    # Go!
    imagedata_dir = conf.dirs['imagedata_dir']
    ts_calc_gee.clean_gee_downloaded_csvs_in_dir(imagedata_dir)
    
if __name__ == '__main__':
    main()

