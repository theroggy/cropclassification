# -*- coding: utf-8 -*-
"""
This is a temporary script file.

@author: Marina De Ketelaere
"""

import logging
import glob
import os
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
                          input_start_date_str: str,
                          input_stop_date_str: str,
                          output_filepath_str: str):
    """
    This function creates a file that is a weekly summarize of timeseries images from DIAZ.
    """

    # init
    year = 2019
    country_code = 'BEFL'

    # lijstje maken van de bestanden -- we starten met VV -- later ook VH + combinaties van Asc en Desc
    # Get the list of inputfiles    
    Csv_InputFiles_List = glob.glob(os.path.join(input_filepath_str, f"*[VV]_.csv"))




    # theoretische weken bepalen uit van-tot datum 
    # -> commando: datetime.??? zie timeseries_calc_gee.py
    
    # First adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
    logger.info('Adapt start_date and end_date so they are mondays')
    def get_monday(date_str):
        """ Get the first monday before the date provided. """
        parseddate = datetime.strptime(date_str, '%Y-%m-%d')
        year_week = parseddate.strftime('%Y_%W')
        year_week_monday = datetime.strptime(year_week + '_1', '%Y_%W_%w')
        return year_week_monday

    start_date = get_monday(start_date_str)
    end_date = get_monday(end_date_str)  
    # Loop over all input files
    for curr_csv in sorted(Csv_InputFiles_List):
   
 




    # bestandsnaam samenstellen voor betreffende week - gelijkaardig vb BEFL2018_bufm0_weekly_2018-03-26_S1AscDesc
        # Get the basename of the file -> new file name
        #current_filename = os.path.basename(input_filepath_str)
        NewFileName = f"{country_code}{year}_weekly_{datumvanmaandag}_{kanalen}.csv"
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



