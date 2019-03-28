# -*- coding: utf-8 -*-
"""
Create the parcel classes that will be used for the classification.

This implementation will create +- 40 classes.
parcel that don't have a clear classification in the input file get class 'UNKNOWN'.

@author: Pieter Roggemans
"""

import logging
import os
import pandas as pd
import geopandas as gpd

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
import cropclassification.global_settings as gs

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_input(input_parcel_filepath: str,
                  input_classtype_to_prepare: str):
    """
    This function creates a file that is compliant with the assumptions used by the rest of the
    classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be classified to
    """
    if input_classtype_to_prepare == 'MONITORING_CROPGROUPS':
        return prepare_input_cropgroups(input_parcel_filepath=input_parcel_filepath)
    elif input_classtype_to_prepare == 'MONITORING_LANDCOVER':
        return prepare_input_landcover(input_parcel_filepath=input_parcel_filepath)
    elif input_classtype_to_prepare == 'MONITORING_MOST_POPULAR_CROPS':
        return prepare_input_most_popular_crops(input_parcel_filepath=input_parcel_filepath)
    elif input_classtype_to_prepare == 'MONITORING_CROPGROUPS_GROUNDTRUTH':
        return prepare_input_cropgroups(input_parcel_filepath=input_parcel_filepath
                                        , crop_columnname='HOOFDTEELT_CTRL_COD')
    elif input_classtype_to_prepare == 'MONITORING_LANDCOVER_GROUNDTRUTH':
        return prepare_input_landcover(input_parcel_filepath=input_parcel_filepath
                                        , crop_columnname='HOOFDTEELT_CTRL_COD')    
    elif input_classtype_to_prepare == 'MONITORING_MOST_POPULAR_CROPS_GROUNDTRUTH':
        return prepare_input_most_popular_crops(input_parcel_filepath=input_parcel_filepath
                                                , crop_columnname='HOOFDTEELT_CTRL_COD')
    else:
        message = f"Unknown value for parameter input_classtype_to_prepare: {input_classtype_to_prepare}"
        logger.fatal(message)
        raise Exception(message)

def prepare_input_cropgroups(input_parcel_filepath: str,
                             crop_columnname: str = 'GWSCOD_H'):
    """
    This function creates a file that is compliant with the assumptions used by the rest of the
    classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be classified to

    This specific implementation converts the typiscal export format used in BE-Flanders to
    this format.
    """
    logger.info('Start of create_classes_cropgroups')

    # Check if parameters are OK and init some extra params
    #--------------------------------------------------------------------------
    if not os.path.exists(input_parcel_filepath):
        raise Exception(f"Input file doesn't exist: {input_parcel_filepath}")
    else:
        logger.info(f"Process input file {input_parcel_filepath}")

    input_dir = os.path.split(input_parcel_filepath)[0]
    input_classes_filepath = os.path.join(input_dir, "MONGROEPEN_20180713.csv")
    if not os.path.exists(input_classes_filepath):
        raise Exception(f"Input classes file doesn't exist: {input_classes_filepath}")
    else:
        logger.info(f"Process input class table file {input_classes_filepath}")

    # Read and cleanup the mapping table from crop codes to classes
    #--------------------------------------------------------------------------
    # REM: needs to be read as ANSI, as excel apperently saves as ANSI
    logger.info(f"Read classes conversion table from {input_classes_filepath}")
    df_classes = pd.read_csv(input_classes_filepath, low_memory=False, sep=';', encoding='ANSI')
    logger.info(f"Read classes conversion table ready, info(): {df_classes.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be converted to
    # unicode to be able to do comparisons with the other data
    df_classes[crop_columnname] = df_classes['Gewas'].astype('unicode')

    # Map column MON_group to orig classname
    df_classes[gs.class_orig_column] = df_classes['MON_groep']

    # Remove unneeded columns
    for column in df_classes.columns:
        if (column not in [gs.class_orig_column, crop_columnname]):
            df_classes.drop(column, axis=1, inplace=True)

    # Set the index
    df_classes.set_index(crop_columnname, inplace=True, verify_integrity=True)

    # Read the parcel data and do the necessary conversions
    #--------------------------------------------------------------------------
    logger.info(f"Read parceldata from {input_parcel_filepath}")
    if os.path.splitext(input_parcel_filepath)[1] == '.csv':
        df_parceldata = pd.read_csv(input_parcel_filepath)
    else:
        df_parceldata = gpd.read_file(input_parcel_filepath)
    logger.info(f"Read Parceldata ready, info(): {df_parceldata.info()}")

    # Check if the id column is present...
    if gs.id_column not in df_parceldata.columns:
        message = f"STOP: Column {gs.id_column} not found in input parcel file: {input_parcel_filepath}. Make sure the column is present or change the column name in global_constants.py"
        logger.critical(message)
        raise Exception(message)

    # Rename the column containing the object id to OBJECT_ID
#    df.rename(columns = {'CODE_OBJ':'object_id'}, inplace = True)

    # Convert the crop to unicode, in case the input is int...
    df_parceldata[crop_columnname] = df_parceldata[crop_columnname].astype('unicode')

    # Join/merge the classname
    logger.info('Add the classes to the parceldata')
    df_parceldata = df_parceldata.merge(df_classes, how='left',
                                        left_on=crop_columnname,
                                        right_index=True,
                                        validate='many_to_one')

    # Data verwijderen
#    df = df[df[classname] != 'Andere subsidiabele gewassen']  # geen boomkweek

    # Add column to signify that the parcel is eligible and set ineligible crop types (important
    # for reporting afterwards)
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    df_parceldata.insert(loc=0, column=gs.is_eligible_column, value=1)
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['1', '2']), gs.is_eligible_column] = 1

    # Add column to signify if the crop/class is permanent, so can/should be followed up in the
    # LPIS upkeep
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    df_parceldata.insert(loc=0, column=gs.is_permanent_column, value=1)
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['1', '2', '3']), gs.is_permanent_column] = 1
    if 'GESP_PM' in df_parceldata.columns:
        # Serres, tijdelijke overkappingen en loodsen
        df_parceldata.loc[df_parceldata['GESP_PM'].isin(['SER', 'SGM', 'LOO']), gs.is_permanent_column] = 1

    # Copy orig classname to classification classname
    df_parceldata.insert(loc=0, column=gs.class_column, value=df_parceldata[gs.class_orig_column])

    # If a column with extra info exists, use it as well to fine-tune the classification classes.
    if 'GESP_PM' in df_parceldata.columns:
        # Serres, tijdelijke overkappingen en loodsen
        df_parceldata.loc[df_parceldata['GESP_PM'].isin(['SER', 'SGM']), gs.class_column] = 'SERRES'
        df_parceldata.loc[df_parceldata['GESP_PM'].isin(['PLA', 'PLO', 'NPO']), gs.class_column] = 'TIJDELIJKE_OVERK'
        df_parceldata.loc[df_parceldata['GESP_PM'] == 'LOO', gs.class_column] = 'MON_STAL'           # Een loods is hetzelfde als een stal...
        df_parceldata.loc[df_parceldata['GESP_PM'] == 'CON', gs.class_column] = 'MON_CONTAINERS'     # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        #df_parceldata.loc[df_parceldata['GESP_PM'] == 'CIV', class_columnname] = 'MON_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning("The column 'GESP_PM' doesn't exist, so this part of the code was skipped!")

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info("Set classes that are still empty, not specific enough or that contain to little values to 'UNKNOWN'")
    df_parceldata.loc[df_parceldata[gs.class_column].str.startswith('nvt', na=True), gs.class_column] = 'UNKNOWN'

    # 'MON_ANDERE_SUBSID_GEWASSEN': very low classification rate (< 1%), as it is a group with several very different classes in it
    # 'MON_AARDBEIEN': low classification rate (~10%), as many parcel actually are temporary greenhouses but aren't correctly applied
    # 'MON_BRAAK': very low classification rate (< 1%), spread over a lot of classes, but most popular are MON_BOOM, MON_GRASSEN, MON_FRUIT
    # 'MON_KLAVER': log classification rate (25%), spread over quite some classes, but MON GRASSES has 20% as well.
    # 'MON_MENGSEL': 25% correct classifications: rest spread over many other classes. Too heterogenous in group?
    # 'MON_POEL': 0% correct classifications: most are classified as MON_CONTAINER, MON_FRUIT. Almost nothing was misclassified as being POEL
    # 'MON_RAAPACHTIGEN': 25% correct classifications: rest spread over many other classes
    # 'MON_STRUIK': 10%
    #    TODO: nakijken, wss opsplitsen of toevoegen aan MON_BOOMKWEEK???
    classes_badresults = ['MON_ANDERE_SUBSID_GEWASSEN', 'MON_AARDBEIEN', 'MON_BRAAK', 'MON_KLAVER'
                          , 'MON_MENGSEL', 'MON_POEL', 'MON_RAAPACHTIGEN', 'MON_STRUIK']
    df_parceldata.loc[df_parceldata[gs.class_column].isin(classes_badresults), gs.class_column] = 'UNKNOWN'

    # MON_BONEN en MON_WIKKEN have omongst each other a very large percentage of false
    # positives/negatives, so they seem very similar... lets create a class that combines both
    df_parceldata.loc[df_parceldata[gs.class_column].isin(['MON_BONEN', 'MON_WIKKEN']), gs.class_column] = 'MON_BONEN_WIKKEN'

    # MON_BOOM includes now also the growing new plants/trees, which is too differenct from grown
    # trees -> put growing new trees is seperate group
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['9602', '9603', '9604', '9560']), gs.class_column] = 'MON_BOOMKWEEK'

    # 'MON_FRUIT': has a good accuracy (91%), but also has as much false positives (115% -> mainly
    #              'MON_GRASSEN' that are (mis)classified as 'MON_FRUIT')
    # 'MON_BOOM': has very bad accuracy (28%) and also very much false positives (450% -> mainly
    #              'MON_GRASSEN' that are misclassified as 'MON_BOOM')
    # MON_FRUIT and MON_BOOM are permanent anyway, so not mandatory that they are checked in
    # monitoring process.
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE_DIFFICULT_PERMANENT_CLASS
    df_parceldata.loc[df_parceldata[gs.class_column].isin(['MON_BOOM', 'MON_FRUIT']), gs.class_column] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'

    # Put MON_STAL, SERRES en TIJDELIJKE OVERK together, too many misclassifiactions amongst
    # each other
    df_parceldata.loc[df_parceldata[gs.class_column].isin(['MON_STAL', 'SERRES', 'TIJDELIJKE_OVERK']), gs.class_column] = 'MON_STAL_SER'

    # Set classes with very few elements to UNKNOWN!
    for index, row in df_parceldata.groupby(gs.class_column).size().reset_index(name='count').iterrows():
        if row['count'] <= 100:
            logger.info(f"Class <{row[gs.class_column]}> only contains {row['count']} elements, so put them to UNKNOWN")
            df_parceldata.loc[df_parceldata[gs.class_column] == row[gs.class_column], gs.class_column] = 'UNKNOWN'

    # For columns that aren't needed for the classification:
    #    - Rename the ones interesting for interpretation
    #    - Drop the columns that aren't useful at all
    for column in df_parceldata.columns:
        if column in (['GRAF_OPP', 'GWSCOD_H', 'GESP_PM']):
            if column == 'GESP_PM':
                df_parceldata['GESP_PM'] = df_parceldata['GESP_PM'].str.replace(',', ';')
            df_parceldata.rename(columns={column:'m#' + column}, inplace=True)

        elif(column not in [gs.id_column, gs.class_column]
             and (not column.startswith('m#'))):
            df_parceldata.drop(column, axis=1, inplace=True)

    return df_parceldata

def prepare_input_landcover(input_parcel_filepath: str,
                             crop_columnname: str = 'GWSCOD_H'):
    """
    This function creates a file that is compliant with the assumptions used by the rest of the
    classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be classified to

    This specific implementation converts the typiscal export format used in BE-Flanders to
    this format.
    """
    logger.info('Start of create_classes_landcover')

    # Check if parameters are OK and init some extra params
    #--------------------------------------------------------------------------
    if not os.path.exists(input_parcel_filepath):
        raise Exception(f"Input file doesn't exist: {input_parcel_filepath}")
    else:
        logger.info(f"Process input file {input_parcel_filepath}")

    input_dir = os.path.split(input_parcel_filepath)[0]
    # TODO: Check MON_LC_groep classificatie (3 klassen gras akker nietsub ) met business 
    input_classes_filepath = os.path.join(input_dir, "LANDCOVERGROEPEN_20190118.csv")
    if not os.path.exists(input_classes_filepath):
        raise Exception(f"Input classes file doesn't exist: {input_classes_filepath}")
    else:
        logger.info(f"Process input class table file {input_classes_filepath}")

    # Read and cleanup the mapping table from crop codes to classes
    #--------------------------------------------------------------------------
    # REM: needs to be read as ANSI, as excel apperently saves as ANSI
    logger.info(f"Read classes conversion table from {input_classes_filepath}")
    df_classes = pd.read_csv(input_classes_filepath, low_memory=False, sep=';', encoding='ANSI')
    logger.info(f"Read classes conversion table ready, info(): {df_classes.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be converted to
    # unicode to be able to do comparisons with the other data
    df_classes[crop_columnname] = df_classes['Gewas'].astype('unicode')

    # Map column MON_group to orig classname
    df_classes[gs.class_orig_column] = df_classes['MON_LC_groep']

    # Remove unneeded columns
    for column in df_classes.columns:
        if (column not in [gs.class_orig_column, crop_columnname]):
            df_classes.drop(column, axis=1, inplace=True)

    # Set the index
    df_classes.set_index(crop_columnname, inplace=True, verify_integrity=True)

    # Read the parcel data and do the necessary conversions
    #--------------------------------------------------------------------------
    logger.info(f"Read parceldata from {input_parcel_filepath}")
    if os.path.splitext(input_parcel_filepath)[1] == '.csv':
        df_parceldata = pd.read_csv(input_parcel_filepath)
    else:
        df_parceldata = gpd.read_file(input_parcel_filepath)
    logger.info(f"Read Parceldata ready, info(): {df_parceldata.info()}")

    # Check if the id column is present...
    if gs.id_column not in df_parceldata.columns:
        message = f"STOP: Column {gs.id_column} not found in input parcel file: {input_parcel_filepath}. Make sure the column is present or change the column name in global_constants.py"
        logger.critical(message)
        raise Exception(message)

    # Rename the column containing the object id to OBJECT_ID
#    df.rename(columns = {'CODE_OBJ':'object_id'}, inplace = True)

    # Convert the crop to unicode, in case the input is int...
    df_parceldata[crop_columnname] = df_parceldata[crop_columnname].astype('unicode')

    # Join/merge the classname
    logger.info('Add the classes to the parceldata')
    df_parceldata = df_parceldata.merge(df_classes, how='left',
                                        left_on=crop_columnname,
                                        right_index=True,
                                        validate='many_to_one')

    # Data verwijderen
#    df = df[df[classname] != 'Andere subsidiabele gewassen']  # geen boomkweek

    # Add column to signify that the parcel is eligible and set ineligible crop types (important
    # for reporting afterwards)
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    df_parceldata.insert(loc=0, column=gs.is_eligible_column, value=1)
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['1', '2']), gs.is_eligible_column] = 1

    # Add column to signify if the crop/class is permanent, so can/should be followed up in the
    # LPIS upkeep
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    df_parceldata.insert(loc=0, column=gs.is_permanent_column, value=1)
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['1', '2', '3']), gs.is_permanent_column] = 1
    if 'GESP_PM' in df_parceldata.columns:
        # Serres, tijdelijke overkappingen en loodsen
        df_parceldata.loc[df_parceldata['GESP_PM'].isin(['SER', 'SGM', 'LOO']), gs.is_permanent_column] = 1

    # Copy orig classname to classification classname
    df_parceldata.insert(loc=0, column=gs.class_column, value=df_parceldata[gs.class_orig_column])

    # If a column with extra info exists, use it as well to fine-tune the classification classes.
    if 'GESP_PM' in df_parceldata.columns:
        # Serres, tijdelijke overkappingen en loodsen
        df_parceldata.loc[df_parceldata['GESP_PM'].isin(['SER', 'SGM']), gs.class_column] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'
        df_parceldata.loc[df_parceldata['GESP_PM'].isin(['PLA', 'PLO', 'NPO']), gs.class_column] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'
        df_parceldata.loc[df_parceldata['GESP_PM'] == 'LOO', gs.class_column] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'           # Een loods is hetzelfde als een stal...
        df_parceldata.loc[df_parceldata['GESP_PM'] == 'CON', gs.class_column] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'     # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        #df_parceldata.loc[df_parceldata['GESP_PM'] == 'CIV', class_columnname] = 'MON_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning("The column 'GESP_PM' doesn't exist, so this part of the code was skipped!")

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info("Set classes that are still empty, not specific enough or that contain to little values to 'UNKNOWN'")
    df_parceldata.loc[df_parceldata[gs.class_column].str.startswith('nvt', na=True), gs.class_column] = 'UNKNOWN'

    # 'MON_ANDERE_SUBSID_GEWASSEN': very low classification rate (< 1%), as it is a group with several very different classes in it
    # 'MON_AARDBEIEN': low classification rate (~10%), as many parcel actually are temporary greenhouses but aren't correctly applied
    # 'MON_BRAAK': very low classification rate (< 1%), spread over a lot of classes, but most popular are MON_BOOM, MON_GRASSEN, MON_FRUIT
    # 'MON_KLAVER': log classification rate (25%), spread over quite some classes, but MON GRASSES has 20% as well.
    # 'MON_MENGSEL': 25% correct classifications: rest spread over many other classes. Too heterogenous in group?
    # 'MON_POEL': 0% correct classifications: most are classified as MON_CONTAINER, MON_FRUIT. Almost nothing was misclassified as being POEL
    # 'MON_RAAPACHTIGEN': 25% correct classifications: rest spread over many other classes
    # 'MON_STRUIK': 10%
    #    TODO: nakijken, wss opsplitsen of toevoegen aan MON_BOOMKWEEK???
    #classes_badresults = ['MON_ANDERE_SUBSID_GEWASSEN', 'MON_AARDBEIEN', 'MON_BRAAK', 'MON_KLAVER'
    #                     , 'MON_MENGSEL', 'MON_POEL', 'MON_RAAPACHTIGEN', 'MON_STRUIK']
    # TODO: MON_POEL, 'MON_STRUIK eventueel verhuizen naar IGNORE_...
    #df_parceldata.loc[df_parceldata[gs.class_column].isin(classes_badresults), gs.class_column] = 'UNKNOWN'

    # MON_BONEN en MON_WIKKEN have omongst each other a very large percentage of false
    # positives/negatives, so they seem very similar... lets create a class that combines both
    #df_parceldata.loc[df_parceldata[gs.class_column].isin(['MON_BONEN', 'MON_WIKKEN']), gs.class_column] = 'MON_BONEN_WIKKEN'

    # MON_BOOM includes now also the growing new plants/trees, which is too differenct from grown
    # trees -> put growing new trees is seperate group
    #df_parceldata.loc[df_parceldata[crop_columnname].isin(['9602', '9603', '9604', '9560']), gs.class_column] = 'MON_BOOMKWEEK'

    # 'MON_FRUIT': has a good accuracy (91%), but also has as much false positives (115% -> mainly
    #              'MON_GRASSEN' that are (mis)classified as 'MON_FRUIT')
    # 'MON_BOOM': has very bad accuracy (28%) and also very much false positives (450% -> mainly
    #              'MON_GRASSEN' that are misclassified as 'MON_BOOM')
    # MON_FRUIT and MON_BOOM are permanent anyway, so not mandatory that they are checked in
    # monitoring process.
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE_DIFFICULT_PERMANENT_CLASS
    #df_parceldata.loc[df_parceldata[gs.class_column].isin(['MON_BOOM', 'MON_FRUIT']), gs.class_column] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'

    # Put MON_STAL, SERRES en TIJDELIJKE OVERK together, too many misclassifiactions amongst
    # each other
    #df_parceldata.loc[df_parceldata[gs.class_column].isin(['MON_STAL', 'SERRES', 'TIJDELIJKE_OVERK']), gs.class_column] = 'MON_STAL_SER'

    # Set classes with very few elements to UNKNOWN!
    #for index, row in df_parceldata.groupby(gs.class_column).size().reset_index(name='count').iterrows():
    #    if row['count'] <= 100:
    #        logger.info(f"Class <{row[gs.class_column]}> only contains {row['count']} elements, so put them to UNKNOWN")
    #        df_parceldata.loc[df_parceldata[gs.class_column] == row[gs.class_column], gs.class_column] = 'UNKNOWN'

    # For columns that aren't needed for the classification:
    #    - Rename the ones interesting for interpretation
    #    - Drop the columns that aren't useful at all
    for column in df_parceldata.columns:
        if column in (['GRAF_OPP', 'GWSCOD_H', 'GESP_PM']):
            if column == 'GESP_PM':
                df_parceldata['GESP_PM'] = df_parceldata['GESP_PM'].str.replace(',', ';')
            df_parceldata.rename(columns={column:'m#' + column}, inplace=True)            

        elif(column not in [gs.id_column, gs.class_column]
             and (not column.startswith('m#'))):
            df_parceldata.drop(column, axis=1, inplace=True)

    return df_parceldata


def prepare_input_most_popular_crops(input_parcel_filepath: str,
                                     crop_columnname: str = 'GWSCOD_H'):
    """
    This function creates a file that is compliant with the assumptions used by the rest of the
    classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be classified to

    This specific implementation converts the typiscal export format used in BE-Flanders to this
    format.
    """

    logger.info('Start of create_classes_most_popular_crops')

    # Check if parameters are OK and init some extra params
    #--------------------------------------------------------------------------
    if not os.path.exists(input_parcel_filepath):
        raise Exception(f"Input file doesn't exist: {input_parcel_filepath}")
    else:
        logger.info(f"Process input file {input_parcel_filepath}")

    # Read and cleanup the mapping table from crop codes to classes
    #--------------------------------------------------------------------------
    logger.info('Start prepare of inputfile')
#    df = pd.read_csv(input_parcel_filepath, low_memory=False)

    logger.info(f'Read parceldata from {input_parcel_filepath}')
    if os.path.splitext(input_parcel_filepath)[1] == '.csv':
        df_parceldata = pd.read_csv(input_parcel_filepath)
    else:
        df_parceldata = gpd.read_file(input_parcel_filepath)
    logger.info(f'Read Parceldata ready, shape: {df_parceldata.shape}')

    # Rename the column containing the object id to OBJECT_ID
#    df.rename(columns = {'CODE_OBJ':'object_id'}, inplace = True)

    # Add columns for the class to use...
    df_parceldata.insert(0, gs.class_column, None)

    df_parceldata.loc[df_parceldata[crop_columnname].isin(['60', '700', '3432']), gs.class_column] = 'Grassland'  # Grassland
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['201', '202']), gs.class_column] = 'Maize'            # Maize
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['901', '904']), gs.class_column] = 'Potatoes'         # Potatoes
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['311', '36']), gs.class_column] = 'WinterWheat'       # Winter wheat of spelt
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['91']), gs.class_column] = 'SugarBeat'               # Sugar beat
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['321']), gs.class_column] = 'WinterBarley'           # Winter barley
    df_parceldata.loc[df_parceldata[crop_columnname].isin(['71']), gs.class_column] = 'FodderBeat'              # Fodder beat

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info("Set classes that are empty to 'IGNORE_UNIMPORTANT_CLASS' so they are ignored further on...")
    df_parceldata.loc[df_parceldata[gs.class_column].isnull(), gs.class_column] = 'IGNORE_UNIMPORTANT_CLASS'

    # Set small parcel to UNKNOWN as well so they are ignored as well...
#    logger.info("Set small parcel 'IGNORE_SMALL' so they are ignored further on...")
#    df.loc[df['GRAF_OPP'] <= 0.3, class_columnname] = 'IGNORE_SMALL'

    # For columns that aren't needed for the classification:
    #    - Rename the ones interesting for interpretation
    #    - Drop the columns that aren't useful at all
    for column in df_parceldata.columns:
        if column in ['GRAF_OPP']:
            df_parceldata.rename(columns={column:'m#' + column}, inplace=True)
        elif (column not in [gs.id_column, gs.class_column]
              and (not column.startswith('m#'))):
            df_parceldata.drop(column, axis=1, inplace=True)

    # Return result
    return df_parceldata

# If the script is run directly...
if __name__ == "__main__":

    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
