import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.cropclassification_runner as runner

def main():        
    runner.run(config_filepaths=['marker/general.ini',
                                 'marker/marker_popular_crops.ini',
                                 'marker/local_overrule.ini'])
    
if __name__ == '__main__':
    main()