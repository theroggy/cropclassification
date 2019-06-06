# -*- coding: utf-8 -*-
"""
This is a temporary script file.

@author: Marina De Ketelaere
"""

import datetime
import logging
import glob
import os
import pandas as pd
import numpy as np

import global_settings as gs


#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)


#-------------------------------------------------------------
# Helpfunctions
#-------------------------------------------------------------

# 
def get_monday(input_date):
    """
    This function gets the first monday before the date provided.
    She is being used to adapt start_date and end_date so they are mondays, so it becomes easier to reuse timeseries data
      Inputformaat: %Y-%m-%d
      outputformaat: %Y_%W_%w vb 2018_5_1 -  maandag van week 5 van 2018.
    """
    #logger.info('get_monday')

    parseddate = datetime.datetime.strptime(input_date, '%Y-%m-%d')
    year_week = parseddate.strftime('%Y_%W')
    year_week_monday = datetime.datetime.strptime(year_week + '_1', '%Y_%W_%w') 
    return year_week_monday

def decompose_filename(input_name):
    """
    This function returns the seperate parts of a filename.
      Input: Filename (met extensie)
      Output: Dictionary of seperate filename parts.
    """
    #logger.info('decompose_filename')

    # Get the date #
    # Remove extension
    filename = os.path.splitext(input_name)[0] 
    # Split into parts
    param_values = filename.split("_")    
    filedatetime = param_values[9]  #example: '20180101T055812'
    filedate = filedatetime.split("T")[0]  
    parseddate = datetime.datetime.strptime(filedate, '%Y%m%d') 

    # Get the week #
    #parseddate = datetime.datetime.strptime(filedate, '%Y%m%d')
    fileweek = int(parseddate.strftime('%W'))


    # Get the band #
    fileband = param_values[19] # to do : hier wordt uitgegaan dat dit steeds 19e deel is maar is niet steeds zo ...? 
                                # --> nog dynamisch maken afh van aantal datums in filenaam

    # Get the orbit # 
    fileorbit = param_values[18]  # to do : hier wordt uitgegaan dat dit steeds 18e deel is maar is niet steeds zo ...?  
                                  # --> nog dynamisch maken afh van aantal datums in filenaam
    
    filenameparts = {
		'name' : filename,
		'date' : parseddate,
		'week' : fileweek,
		'band' : fileband, #VV/VH
		'orbit' : fileorbit} #ASCDESC
	
    return filenameparts


#-------------------------------------------------------------
# Start
#-------------------------------------------------------------
def calculate_weekly_data(input_filepath: str,
                          input_start_date: str,   #formaat: %Y-%m-%d
                          input_stop_date: str,    #formaat: %Y-%m-%d
                          input_band: str,         #VV of VH                #marina VRAAG: moet beide ook mogelijk zijn?
                          input_orbit: str,        #ASC of DESC             #marina VRAAG: moet beide ook mogelijk zijn?
                          output_filepath: str
                          ):
    """
    This function creates a file that is a weekly summarize of timeseries images from DIAS.
    """
    logger.info('calculate_weekly_data')

    # example Sourcefilename : Prc_BEFL_2018_2018-08-02__S1A_IW_GRDH_1SDV_20180101T055812_20180101T055837_019957_021FBF_Orb_RBN_RTN_Cal_TC.CARD_DESC_VH.csv

    # Init
    country_code = 'BEFL'

    # to do : jaar afleiden uit input_start_date
    year = 2018  #2019

    # to do : paden afleiden uit ini file - moeten dan niet meer worden meegegeven als parameter
   

    ##### 1. Create Dataframe of all files #####
    logger.info('Create Dataframe of all files')

    # Arrays aanmaken
    row_column = ['FileName', 'FileDate', 'FileWeek', 'Band', 'Orbit']
    row_data = []

    for filename in os.listdir(input_filepath): 
        if filename.endswith(".csv"):

            # Get seperate filename parts
            filenameparts = decompose_filename(filename)

            # RowArray opvullen
            new_row = [filenameparts['name'], filenameparts['date'], filenameparts['week'], filenameparts['band'], filenameparts['orbit']]
            row_data.append(new_row)
    

    allinputfiles_df = pd.DataFrame(data=row_data, 
                                    columns=row_column)

    ##### 2. Dataframe vernauwen : rekening houdend met start- en stopdatums, kanalen, ... #####
    logger.info('start- & stopdates, bands')
    
    # Start- en stopdatum omzetten naar maandagen
    start_date = get_monday(input_start_date) # output: vb 2018_2_1 - maandag van week 2 van 2018
    end_date = get_monday(input_stop_date) 
    start_week = int(datetime.datetime.strftime(start_date , '%W'))
    end_week = int(datetime.datetime.strftime(end_date , '%W'))
    start_date_monday = start_date.strftime('%Y-%m-%d') # terug omzetten naar Y/M/D
    end_date_monday = end_date.strftime('%Y-%m-%d')

    # Dataframe vernauwen tot start- en stopmaandagen, kanalen, ... 
    allinputfiles_df = allinputfiles_df[(allinputfiles_df.FileDate >= start_date_monday) 
                                        & (allinputfiles_df.FileDate < end_date_monday) 
                                        & (allinputfiles_df.Band == input_band) 
                                        & (allinputfiles_df.Orbit == input_orbit)]

    ##### 3. Samennemen per week #####
    logger.info('a week')

    # For each week
    for i in range(start_week, end_week):

        # array maken van files voor week i
        weekfiles_list = allinputfiles_df[allinputfiles_df.FileWeek == i].FileName.tolist()
        
        # Loop all weekfiles
        df_result = None
        # numberfiles = None
        for countfiles, weekfile in enumerate(weekfiles_list):
            
            # Read the file - columns : code_obj,count,max,mean,min,p25,p50,p75,std,band
            # skip p25,p50,p75 and band for now
            df_in = pd.read_csv(os.path.join(input_filepath, f"{weekfile}.csv"), 
                                usecols=['CODE_OBJ', 'count', 'max', 'mean', 'min', 'std'],
                                index_col='CODE_OBJ')  
            '''print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print(df_in.max.isnull())

            if df_in.max.isnull() :
                numberfiles = 0
            else:    
                numberfiles = 1

            df_in[numberfiles] = 
            print(numberfiles)

            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')'''

            # rename columns - add unique number                   
            df_in.rename(columns={'count': 'count'+str(countfiles+1), 'max': 'max'+str(countfiles+1), 
                                  'mean': 'mean'+str(countfiles+1), 'min': 'min'+str(countfiles+1),
                                  'std': 'std'+str(countfiles+1)}, inplace=True)
            
            # Create 1 dataframe for all weekfiles - one row for each code_obj - using concat (code_obj = index)
            if df_result is None:
                df_result = df_in
            else:
                df_result = pd.concat([df_result, df_in], axis=1, sort=False) 
                df_result.count
        
        
        # Calculate max, mean, min, ...
        logger.info('Calculate max, mean, min, ...')

        countcolumnstouse = []
        maxcolumnstouse = []
        meancolumnstouse = []
        mincolumnstouse = []
        stdcolumnstouse = []

        for j in range (1, countfiles+1):
            countcolumnstouse.append(f"count{j}")
            maxcolumnstouse.append(f"max{j}")
            meancolumnstouse.append(f"mean{j}")
            mincolumnstouse.append(f"min{j}")
            stdcolumnstouse.append(f"std{j}")

        # 
        # df_result['NumberFiles'] = df_result[maxcolumnstouse].isna.sum(axis=1)


        # Get the date of the monday of week i
        week_i_monday = datetime.datetime.strptime(str(year) + '_' + str(i) + '_1', '%Y_%W_%w')
        date_week_i_monday = week_i_monday.strftime('%Y-%m-%d') 

        NewColumnName = f"weekly_{input_band}_{input_orbit}_{date_week_i_monday}" # marina : is dit eigenlijk nodig?

        # Number of pixels
        # TO DO: onderzoeken hoe aantal pixels best bijgehouden wordt : afwijkingen weglaten ? max nemen ? ...
        df_result[f"{NewColumnName}_count"] = np.nanmax(df_result[countcolumnstouse], axis=1)

        # Maximum of all max columns
        df_result[f"{NewColumnName}_max"] = np.nanmax(df_result[maxcolumnstouse], axis=1)

        # Mean of all mean columns
        df_result[f"{NewColumnName}_mean"] = np.nanmean(df_result[meancolumnstouse], axis=1)

        # Minimum of all min columns
        df_result[f"{NewColumnName}_min"] = np.nanmin(df_result[mincolumnstouse], axis=1)

        # Percentielen : (voorlopig niets mee doen)
 
        # Mean of all std columns
        df_result[f"{NewColumnName}_std"] = np.nanmean(df_result[stdcolumnstouse], axis=1)

        # Number of Files used
        #df_result[f"{NewColumnName}_countfiles"] = df_result.isna().sum()
        #print(df_result.isna(axis=1).sum())
        #for i in range(0,5) : #range(len(df_result.index()) :
            #print("Nan in row ", i , " : " ,  df_result.iloc[i].isnull().sum())



        # marina : to to : 
        #      aantal bestanden bijhouden per object --> niet lege !

        
        # New file Name
        # vb BEFL2018_bufm0_weekly_2018-03-26_S1AscDesc
        # for temp. files use parquet in stead of csv 
        # TO DO : buffer en soort sensor nog mee opnemen in de naam
        newfilename = f"{country_code}{year}_bufm0_weekly_{date_week_i_monday}_{input_band}_{input_orbit}.parquet"
        dest_filepath = os.path.join(output_filepath, newfilename)

        logger.info('write new file: '+newfilename)
        # wegschrijven naar nieuw bestand
        columnstouse = [f"{NewColumnName}_count", f"{NewColumnName}_max", f"{NewColumnName}_mean", f"{NewColumnName}_min", 
                        f"{NewColumnName}_std"] 
        
        perceel_info_weekly_df = df_result[columnstouse]

        perceel_info_weekly_df.to_csv(dest_filepath, index_label='CODE_OBJ')
        #perceel_info_weekly_df.to_parquet(dest_filepath, index_label='CODE_OBJ')
        
        logger.info('-THE END-')



# If the script is run directly...
if __name__ == "__main__":

    log_dir = "X:\\Monitoring\\Markers\\playground\\_algemeen\\timeseries_dias\\log"
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)

    # Set the general maximum log level...
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.flush()
        handler.close()

    # Remove all handlers and add the ones I want again, so a new log file is created for each run
    # Remark: the function removehandler doesn't seem to work?
    logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # ch.setFormatter(logging.Formatter('%(levelname)s|%(name)s|%(message)s'))
    ch.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
    logger.addHandler(ch)

    log_filepath = os.path.join(log_dir, f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_class_maincrop.log")
    fh = logging.FileHandler(filename=log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(name)s|%(message)s'))
    logger.addHandler(fh)

    #-------------------------------------------------------------
    # onderaan functie aanroepen met params om te testen/runnen
    #-------------------------------------------------------------
    input_filepath = 'X:\\Monitoring\\Markers\\playground\\_algemeen\\timeseries_dias'
    input_start_date = '2018-01-15'
    input_stop_date = '2018-02-15'
    input_band = 'VV'
    input_orbit = 'ASC'
    output_filepath = 'X:\\Monitoring\\Markers\\playground\\market\\output'
    
    calculate_weekly_data(input_filepath, input_start_date , input_stop_date, input_band, input_orbit, output_filepath)
    
