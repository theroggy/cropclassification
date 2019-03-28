import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.cropclassification_runner as runner

def main():        
    runner.run(config_filepaths=['general.ini',
                                 'marker_popular_crops.ini',
                                 'local_overrule.ini'])
    
if __name__ == '__main__':
    main()