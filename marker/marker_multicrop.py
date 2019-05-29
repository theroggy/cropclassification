# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:51:26 2018

@author: pierog
"""

"""
Testdata in 2018: Parcels with 2 crops in them:
"00002806653D6B97"   # 10%/90% twee gewassen
"00002806653D6ACD"   # 50%/50% twee gewassen
"0000280664E917CC"   # Verwaarloosbaar stukje in perceel dat er wat anders uit ziet
"""

import sys
[sys.path.append(i) for i in ['.', '..']]

import cropclassification.marker_runner as runner 

def main():        
    runner.run(config_filepaths=['config/general.ini',
                                 'config/multicrop.ini',
                                 'config/local_overrule.ini'])
    
if __name__ == '__main__':
    main()