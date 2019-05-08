import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.marker_runner as runner 

def main():        
    runner.run(config_filepaths=['marker/general.ini',
                                 'marker/marker_landcover.ini',
                                 'marker/local_overrule.ini'],
               reuse_last_run_dir=False)
    
if __name__ == '__main__':
    main()