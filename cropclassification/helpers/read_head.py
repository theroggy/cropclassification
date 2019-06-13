# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:09:47 2018

@author: pierog
"""

base_dir = 'X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-05-04_Monitoring_Classificatie'
filepath = base_dir + "\\2018_multicrop\\2018-09-26_Run1\\BEFL2018_bufm10_weekly_parcel_classdata.csv"

# Print the first 5 lines of the file
with open(filepath, "r") as file:
    for i in range(5):
        print(file.readline())
