import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.marker_runner as runner

def main():        
    runner.run(config_filepaths=['config/general.ini',
                                 'config/cropgroup.ini',
                                 'config/local_overrule.ini'])
    
if __name__ == '__main__':
    main()