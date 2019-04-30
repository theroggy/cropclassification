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

#functie maken
# inputparams: inputdirectory (waar bestanden staan) - output dir (waarnaartoe) - datums van-tot - 
# bestanden per week samen nemen
#    lijstje maken van de bestanden 
#       -> commando: glob.glob()
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
def calculate_weekly_data(input_filepath_str: str,
                          input_start_date_str: str,   #formaat: %Y-%m-%d
                          input_stop_date_str: str,   #formaat: %Y-%m-%d
                          output_filepath_str: str):
    """
    This function creates a file that is a weekly summarize of timeseries images from DIAZ.
    """

    # Init
    year = 2018  #2019
    country_code = 'BEFL'

    base_filename = f"Prc_{country_code}{year}_" #VRAAG: die Prc blijft erbij?

    # Get the list of inputfiles    
    # marina: lijstje maken van de bestanden -- voorlopig voor VV -- later ook VH + combinaties van Asc en Desc
    Csv_InputFiles_List = glob.glob(os.path.join(input_filepath_str, f"*[VV]_.csv"))

    # marina: start en stop path definiëren
    filepath_start = os.path.join(input_filepath_str, f"{base_filename}_{input_start_date_str}.csv")
    filepath_end = os.path.join(input_filepath_str, f"{base_filename}_{input_stop_date_str}.csv")
    logger.debug(f'filepath_start_date: {filepath_start}')
    logger.debug(f'filepath_end_date: {filepath_end}')

    # marina: theoretische weken bepalen uit van-tot datum 
    
    # First adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
    logger.info('Adapt start_date and end_date so they are mondays')
    def get_monday(date_str):   # marina: vb 2018_5_1 -  maandag van week 5 van 2018
        """ Get the first monday before the date provided. """
#vraag: strptime -> strftime -> om eruit te halen en dan w te kunnen toevoegen?
#vraag: wanneer strptime en strftime - gevoelsmatig?
        parseddate = datetime.strptime(date_str, '%Y-%m-%d')
        year_week = parseddate.strftime('%Y_%W')
        year_week_monday = datetime.strptime(year_week + '_1', '%Y_%W_%w')
        return year_week_monday

    start_date = get_monday(input_start_date_str) # marina: b 2018_2_1 - maandag van week 2 van 2018
    end_date = get_monday(input_stop_date_str) # marina: vb 2018_5_1 -  maandag van week 5 van 2018
    start_date_monday = start_date.strftime('%Y-%m-%d') 
    end_date_monday = end_date.strftime('%Y-%m-%d')


    # Loop over all sorted input files 
    # vraag: maakt sorted iets uit?
    for curr_csv in sorted(Csv_InputFiles_List):
    
        # The only data we want to process is the data in the range of dates
        # marina: enkel de bestanden die tss start en eind datum liggen 
        if((curr_csv < filepath_start) or (curr_csv >= filepath_end)):
            logger.debug(f"File is not in date range asked, skip it: {curr_csv}")
            continue

        logger.info(f'Process file: {curr_csv}')

        # An empty file signifies that there wasn't any valable data for that period/sensor/...
        if os.path.getsize(curr_csv) == 0:
            logger.info(f"File is empty, so SKIP: {curr_csv}")
            continue

        # marina: per week samennemen - telkens - per week - een nieuw lijstje maken?
        # 
        # 
        # 
        # # New file name
        # marina: bestandsnaam samenstellen voor betreffende week - gelijkaardig vb BEFL2018_bufm0_weekly_2018-03-26_S1AscDesc
        # marina: current_filename = os.path.basename(input_filepath_str)
        # marina: TO DO : buffer nog inbouwen
        # marina: TO DO : {datumvanmaandag} nog bepalen
        # marina: TO DO : {kanalen} nog bepalen
        NewFileName = f"{country_code}{year}_bufm0_weekly_{datumvanmaandag}_{kanalen}.csv"
        dest_filepath = os.path.join(output_filepath_str, NewFileName)
        
        
        
        



       


    #(onderstaand zoveel mogelijk met pandas functies doen --> alles samenvoegen in 1 dataframe met een rij per object )
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

    #      percentielen : (voorlopig niets)
    
    #      standaard deviatie : mean nemen      
    ColumnsToUse = ['VH_ASC_20180326_stdDev', 'VH_ASC_20180327_stdDev', 'VH_ASC_20180328_stdDev']
    perceel_info_df['weekly_VH_ASC_20180326_stdDev'] = perceel_info_df[columnsToUse].nanmean()

    ColumnsToUse = [weekly_*]
    perceel_info_weekly_df = perceel_info_df[columnsToUse]
    perceel_info_weekly_df.to_csv(bestandsnaam)





# onderaan functie aanroepen met params om te testen/runnen
calculate_weekly_data('X:\Monitoring\Markers\playground\_algemeen\timeseries_dias',   #input_filepath_str
                      '2018-01-15', #input_start_date_str,   #formaat: %Y-%m-%d
                      '2018-02-15', #input_stop_date_str,   #formaat: %Y-%m-%d
                      'X:\Monitoring\Markers\playground\_algemeen\timeseries_diasTODOWNLOAD\MarKet')   #output_filepath_str




