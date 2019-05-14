# -*- coding: utf-8 -*-
"""
This is a temporary script file.

@author: Marina De Ketelaere
"""

import logging
import glob
import os
import pandas as pd
#import geopandas as gpd
#import global_settings as gs

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# Helpfunctions
#-------------------------------------------------------------

# Function to adapt a date to the monday of the same week
#          is used to adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
#   Inputformaat: %Y-%m-%d
#   outputformaat: %Y_%W_%w vb 2018_5_1 -  maandag van week 5 van 2018
def get_monday(input_date):
    """ Get the first monday before the date provided. """
    parseddate = datetime.strptime(input_date, '%Y-%m-%d')
    year_week = parseddate.strftime('%Y_%W')
    year_week_monday = datetime.strptime(year_week + '_1', '%Y_%W_%w') 
    return year_week_monday





#functie maken
# inputparams: inputdirectory (waar bestanden staan) - output dir (waarnaartoe) - datums van-tot - 
# bestanden per week samen nemen
#    lijstje maken van de bestanden 
#       -> commando: glob.glob()
#          eg Csv_InputFiles_List = glob.glob(os.path.join(input_filepath, f"*.csv"))
#       -> misschien in pandas dateframe steken om makkelijk op te kunnen filteren?
#    theoretische weken bepalen uit van-tot datum 
#       -> commando: datetime.??? zie timeseries_calc_gee.py
#    loop over weken 
#       zoek bestanden voor betreffende week 
#       bestandsnaam samenstellen voor betreffende week - gelijkaardig vb BEFL2018_bufm0_weekly_2018-03-26_S1AscDesc
#       bestanden uitlezen en samenvatten   
 
#-------------------------------------------------------------
# Start
#-------------------------------------------------------------
def calculate_weekly_data(input_filepath: str,
                          input_start_date: str,   #formaat: %Y-%m-%d
                          input_stop_date: str,    #formaat: %Y-%m-%d
                          input_VV_VH: str,        #VV of VH                #marina VRAAG: moet beide ook mogelijk zijn?
                          input_ASC_DESC: str,     #ASC of DESC             #marina VRAAG: moet beide ook mogelijk zijn?
                          output_filepath: str
                          ):
    """
    This function creates a file that is a weekly summarize of timeseries images from DIAS.
    """

    # Voorbeeld Sourcefilename : Prc_BEFL_2018_2018-08-02__S1A_IW_GRDH_1SDV_20180101T055812_20180101T055837_019957_021FBF_Orb_RBN_RTN_Cal_TC.CARD_DESC_VH.csv


    # Init
    year = 2018  #2019          #marina VRAAG: moet jaartal afgeleid worden uit sourcefilename? of parameter?
    country_code = 'BEFL'
    

    ##### 1. Create Dataframe of all files #####
    logger.info('Create Dataframe of all files')

    # Arrays aanmaken 
    filename_list = []
    filedate_list = []
    fileweek_list = []
    fileVVVH_list = []
    fileASCDESC_list = []
    
    for fullfilename in input_filepath: 
        # Filename (met extensie) = fullfilename
        
        # datum eruit halen
        # extensie verwijderen
        filename = os.path.splitext(fullfilename)[0] #marina VRAAG: is deze stap nodig?
        # in stukken splitsen
        param_values = filename.split("_")    
        filedate = param_values[3] #marina VRAAG: is dat ok om ervan uit te gaan dat het altijd het 3e deel gaat zijn?  #datum als string ok ?

        parseddate = datetime.strptime(filedate, '%Y-%m-%d')
        fileweek = parseddate.strftime('%W')

        # VV/VH eruit halen
        fileVVVH = param_values[18] #marina VRAAG: is dat ok om ervan uit te gaan dat het altijd het 18e deel gaat zijn?

        # Asc/Desc eruit halen
        fileASCDESC = param_values[17] #marina VRAAG: is dat ok om ervan uit te gaan dat het altijd het 17e deel gaat zijn?

        # Arrays opvullen
        filename_list.append(filename)
        filedate_list.append(filedate)
        fileweek_list.append(fileweek)
        fileVVVH_list.append(fileVVVH)
        fileASCDESC_list.append(fileASCDESC)
    
     
    AllInputFiles_df = pd.dataframe(data=[filename_list, filedate_list, fileweek_list, fileVVVH_list, fileASCDESC_list],
                                   ,columns=['FileName', 'FileDate', 'FileWeek', 'VV_VH', 'ASC_DESC'])

    
    ##### 2. Dataframe vernauwen : rekening houdend met start- en stopdatums, kanalen, ... #####
    logger.info('')
    
    # Start- en stopdatum omzetten naar maandagen
    logger.info('Adapt start_date and end_date so they are mondays')

    start_date = get_monday(input_start_date) # output: vb 2018_2_1 - maandag van week 2 van 2018
    end_date = get_monday(input_stop_date) 
    start_week = start_date.split("_")[1] # week eruit halen
    end_week = end_date.split("_")[1]
    start_date_monday = start_date.strftime('%Y-%m-%d') # terug omzetten naar Y/M/D
    end_date_monday = end_date.strftime('%Y-%m-%d')

    # Dataframe vernauwen tot start- en stopmaandagen, kanalen, ... 
    AllInputFiles_df = AllInputFiles_df[(AllInputFiles_df.FileDate >= start_date_monday) & (AllInputFiles_df.FileDate < end_date_monday) 
                                        & (AllInputFiles_df.VV_VH == input_VV_VH) & (AllInputFiles_df.ASC_DESC == input_ASC_DESC)]


    ##### 3. Samennemen per week #####
    logger.info('')

    # loop over weeknummer
    for i in range(start_week, end_week):

        # array maken van files voor week i
        WeekFiles_List = [AllInputFiles_df[(AllInputFiles_df.FileWeek == i)].FileName] #marina VRAAG: hier in df steken of array?

        # New file Name
        # vb BEFL2018_bufm0_weekly_2018-03-26_S1AscDesc
        # marina TO DO : buffer nog inbouwen
        # marina TO DO : {datumvanmaandag} van betreffende week nog bepalen
        NewFileName = f"{country_code}{year}_bufm0_weekly_{datumvanmaandag}_{input_VV_VH}_{input_ASC_DESC}.csv"
        dest_filepath = os.path.join(output_filepath, NewFileName)
        
        
        




    # kolommen van sourcebestanden : CODE_OBJ,count,max,mean,min,p25,p50,p75,std,band


    # eenmaal alles in 1 dataframe zit met een rij per object
    # dan onderstaand zoveel mogelijk met pandas functies doen
    
    #  per object :
    #      aantal bestanden bijhouden
    #      aantal pixels : onderzoeken - afwijkingen weglaten ? max nemen ? ...

    #      max : max van max nemen
    ColumnsToUse = ['VH_ASC_20180326_mean', 'VH_ASC_20180327_mean', 'VH_ASC_20180328_mean']
    perceel_info_df['weekly_VH_ASC_20180326_mean'] = perceel_info_df[columnsToUse].nanmean(axis=?)

    #      mean : mean van mean nemen
    ColumnsToUse = ['VH_ASC_20180326_mean', 'VH_ASC_20180327_mean', 'VH_ASC_20180328_mean']
    perceel_info_df['weekly_VH_ASC_20180326_mean'] = perceel_info_df[columnsToUse].nanmean(axis=?)

    #      min : min van min nemen
    ColumnsToUse = ['VH_ASC_20180326_mean', 'VH_ASC_20180327_mean', 'VH_ASC_20180328_mean']
    perceel_info_df['weekly_VH_ASC_20180326_mean'] = perceel_info_df[columnsToUse].nanmean(axis=?)

    #      percentielen : (voorlopig niet)
    
    #      standaard deviatie : mean nemen      
    ColumnsToUse = ['VH_ASC_20180326_stdDev', 'VH_ASC_20180327_stdDev', 'VH_ASC_20180328_stdDev']
    perceel_info_df['weekly_VH_ASC_20180326_stdDev'] = perceel_info_df[columnsToUse].nanmean()

    ColumnsToUse = [weekly_*]
    perceel_info_weekly_df = perceel_info_df[columnsToUse]
    perceel_info_weekly_df.to_csv(bestandsnaam)






#-------------------------------------------------------------
# onderaan functie aanroepen met params om te testen/runnen
#-------------------------------------------------------------
calculate_weekly_data('X:\Monitoring\Markers\playground\_algemeen\timeseries_dias',   #input_filepath
                      '2018-01-15', #input_start_date,   #formaat: %Y-%m-%d
                      '2018-02-15', #input_stop_date,   #formaat: %Y-%m-%d
                      'VV', #input_VV_VH
                      'ASC', #input_ASC_DESC
                      'X:\Monitoring\Markers\playground\_algemeen\timeseries_diasTODOWNLOAD\MarKet')   #output_filepath


