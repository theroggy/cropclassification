# -*- coding: utf-8 -*-
"""
Create the parcel classes that will be used for the classification.

This implementation will create +- 40 classes.
parcel that don't have a clear classification in the input file get class 'UNKNOWN'.
"""

import logging
import os

import pandas as pd
import geopandas as gpd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh
import cropclassification.helpers.geofile as geofile_util

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def prepare_input(input_parcel_filepath: str,
                  classtype_to_prepare: str):
    """
    This function creates a file that is compliant with the assumptions used by the rest of the
    classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be classified to
    """
    if classtype_to_prepare == 'MONITORING_CROPGROUPS':
        return prepare_input_cropgroups(input_parcel_filepath=input_parcel_filepath)
    elif classtype_to_prepare == 'MONITORING_CROPGROUPS_GROUNDTRUTH':
        return prepare_input_cropgroups(input_parcel_filepath=input_parcel_filepath,
                                        crop_columnname='HOOFDTEELT_CTRL_COD')
    elif classtype_to_prepare == 'MONITORING_LANDCOVER':
        return prepare_input_landcover(input_parcel_filepath=input_parcel_filepath)
    elif classtype_to_prepare == 'MONITORING_LANDCOVER_GROUNDTRUTH':
        return prepare_input_landcover(input_parcel_filepath=input_parcel_filepath,
                                       crop_columnname='HOOFDTEELT_CTRL_COD')  
    elif classtype_to_prepare == 'MONITORING_MOST_POPULAR_CROPS':
        return prepare_input_most_popular_crops(input_parcel_filepath=input_parcel_filepath)      
    elif classtype_to_prepare == 'MONITORING_MOST_POPULAR_CROPS_GROUNDTRUTH':
        return prepare_input_most_popular_crops(input_parcel_filepath=input_parcel_filepath,
                                                crop_columnname='HOOFDTEELT_CTRL_COD')
    else:
        message = f"Unknown value for parameter classtype_to_prepare: {classtype_to_prepare}"
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

    # Read and cleanup the mapping table from crop codes to classes
    #--------------------------------------------------------------------------
    input_dir = conf.dirs['input_dir']
    input_classes_filepath = os.path.join(input_dir, "refe_mon_cropgroups_landcover_2018.csv")
    if not os.path.exists(input_classes_filepath):
        raise Exception(f"Input classes file doesn't exist: {input_classes_filepath}")
    else:
        logger.info(f"Process input class table file {input_classes_filepath}")

    # REM: needs to be read as ANSI, as SQLDetective apparently saves as ANSI
    logger.info(f"Read classes conversion table from {input_classes_filepath}")
    classes_df = pd.read_csv(input_classes_filepath, low_memory=False, sep=',', encoding='ANSI')
    logger.info(f"Read classes conversion table ready, info(): {classes_df.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be converted to
    # unicode to be able to do comparisons with the other data
    classes_df[crop_columnname] = classes_df['CROPCODE'].astype('unicode')

    # Map column with the classname to orig classname
    classes_df[conf.columns['class_orig']] = classes_df['MON_CROPGROUP']

    # Remove unneeded columns
    for column in classes_df.columns:
        if (column not in [conf.columns['class_orig'], crop_columnname]):
            classes_df.drop(column, axis=1, inplace=True)

    # Set the index
    classes_df.set_index(crop_columnname, inplace=True, verify_integrity=True)

    # Read the parcel data and do the necessary conversions
    #--------------------------------------------------------------------------
    logger.info(f"Read parceldata from {input_parcel_filepath}")
    if geofile_util.is_geofile(input_parcel_filepath):
        parceldata_df = geofile_util.read_file(input_parcel_filepath)
    else:
        parceldata_df = pdh.read_file(input_parcel_filepath)

    logger.info(f"Read Parceldata ready, info(): {parceldata_df.info()}")

    # Check if the id column is present...
    if conf.columns['id'] not in parceldata_df.columns:
        message = f"STOP: Column {conf.columns['id']} not found in input parcel file: {input_parcel_filepath}. Make sure the column is present or change the column name in global_constants.py"
        logger.critical(message)
        raise Exception(message)

    # Rename the column containing the object id to OBJECT_ID
#    df.rename(columns = {'CODE_OBJ':'object_id'}, inplace = True)

    # Convert the crop to unicode, in case the input is int...
    parceldata_df[crop_columnname] = parceldata_df[crop_columnname].astype('unicode')

    # Join/merge the classname
    logger.info('Add the classes to the parceldata')
    parceldata_df = parceldata_df.merge(
            classes_df, how='left', left_on=crop_columnname, right_index=True, validate='many_to_one')

    # Copy orig classname to classification classname
    parceldata_df.insert(loc=0, column=conf.columns['class'], value=parceldata_df[conf.columns['class_orig']])
    
    # Add column to signify that the parcel is eligible and set ineligible crop types (important
    # for reporting afterwards)
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    parceldata_df.insert(loc=0, column=conf.columns['is_eligible'], value=1)
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['1', '2']), conf.columns['is_eligible']] = 1

    # Add column to signify if the crop/class is permanent, so can/should be followed up in the
    # LPIS upkeep
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    parceldata_df.insert(loc=0, column=conf.columns['is_permanent'], value=1)
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['1', '2', '3']), conf.columns['is_permanent']] = 1
    if 'GESP_PM' in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[parceldata_df['GESP_PM'].isin(['SER', 'SGM', 'LOO']), conf.columns['is_permanent']] = 1

    # If a column with extra info exists, use it as well to fine-tune the classification classes.
    if 'GESP_PM' in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[parceldata_df['GESP_PM'].isin(['SER', 'SGM']), conf.columns['class']] = 'SERRES'
        parceldata_df.loc[parceldata_df['GESP_PM'].isin(['PLA', 'PLO', 'NPO']), conf.columns['class']] = 'TIJDELIJKE_OVERK'
        parceldata_df.loc[parceldata_df['GESP_PM'] == 'LOO', conf.columns['class']] = 'MON_STAL'           # Een loods is hetzelfde als een stal...
        parceldata_df.loc[parceldata_df['GESP_PM'] == 'CON', conf.columns['class']] = 'MON_CONTAINERS'     # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        #parceldata_df.loc[parceldata_df['GESP_PM'] == 'CIV', class_columnname] = 'MON_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning("The column 'GESP_PM' doesn't exist, so this part of the code was skipped!")

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info("Set classes that are still empty, not specific enough or that contain to little values to 'UNKNOWN'")
    parceldata_df.loc[parceldata_df[conf.columns['class']].str.startswith('nvt', na=True), conf.columns['class']] = 'UNKNOWN'

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
    parceldata_df.loc[parceldata_df[conf.columns['class']].isin(classes_badresults), conf.columns['class']] = 'UNKNOWN'

    # MON_BONEN en MON_WIKKEN have omongst each other a very large percentage of false
    # positives/negatives, so they seem very similar... lets create a class that combines both
    parceldata_df.loc[parceldata_df[conf.columns['class']].isin(['MON_BONEN', 'MON_WIKKEN']), conf.columns['class']] = 'MON_BONEN_WIKKEN'

    # MON_BOOM includes now also the growing new plants/trees, which is too differenct from grown
    # trees -> put growing new trees is seperate group
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['9602', '9603', '9604', '9560']), conf.columns['class']] = 'MON_BOOMKWEEK'

    # 'MON_FRUIT': has a good accuracy (91%), but also has as much false positives (115% -> mainly
    #              'MON_GRASSEN' that are (mis)classified as 'MON_FRUIT')
    # 'MON_BOOM': has very bad accuracy (28%) and also very much false positives (450% -> mainly
    #              'MON_GRASSEN' that are misclassified as 'MON_BOOM')
    # MON_FRUIT and MON_BOOM are permanent anyway, so not mandatory that they are checked in
    # monitoring process.
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE_DIFFICULT_PERMANENT_CLASS
    parceldata_df.loc[parceldata_df[conf.columns['class']].isin(['MON_BOOM', 'MON_FRUIT']), conf.columns['class']] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'

    # Put MON_STAL, SERRES en TIJDELIJKE OVERK together, too many misclassifiactions amongst
    # each other
    parceldata_df.loc[parceldata_df[conf.columns['class']].isin(['MON_STAL', 'SERRES', 'TIJDELIJKE_OVERK']), conf.columns['class']] = 'MON_STAL_SER'

    # Set classes with very few elements to UNKNOWN!
    for index, row in parceldata_df.groupby(conf.columns['class']).size().reset_index(name='count').iterrows():
        if row['count'] <= 100:
            logger.info(f"Class <{row[conf.columns['class']]}> only contains {row['count']} elements, so put them to UNKNOWN")
            parceldata_df.loc[parceldata_df[conf.columns['class']] == row[conf.columns['class']], conf.columns['class']] = 'UNKNOWN'

    # For columns that aren't needed for the classification:
    #    - Rename the ones interesting for interpretation
    #    - Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if column in (['GRAF_OPP', 'GWSCOD_H', 'GESP_PM']):
            if column == 'GESP_PM':
                parceldata_df['GESP_PM'] = parceldata_df['GESP_PM'].str.replace(',', ';')
            parceldata_df.rename(columns={column:'m__' + column}, inplace=True)

        elif(column not in [conf.columns['id'], conf.columns['class']]
             and column not in conf.preprocess.getlist('extra_export_columns')
             and not column.startswith('m__')):
            parceldata_df.drop(column, axis=1, inplace=True)

    return parceldata_df

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

    # Read and cleanup the mapping table from crop codes to classes
    #--------------------------------------------------------------------------
    input_dir = conf.dirs['input_dir']
    input_classes_filepath = os.path.join(input_dir, "refe_mon_cropgroups_landcover_2018.csv")
    if not os.path.exists(input_classes_filepath):
        raise Exception(f"Input classes file doesn't exist: {input_classes_filepath}")
    else:
        logger.info(f"Process input class table file {input_classes_filepath}")

    # REM: needs to be read as ANSI, as SQLDetective apparently saves as ANSI
    logger.info(f"Read classes conversion table from {input_classes_filepath}")
    classes_df = pd.read_csv(input_classes_filepath, low_memory=False, sep=',', encoding='ANSI')
    logger.info(f"Read classes conversion table ready, info(): {classes_df.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be converted to
    # unicode to be able to do comparisons with the other data
    classes_df[crop_columnname] = classes_df['CROPCODE'].astype('unicode')

    # Map column MON_group to orig classname
    classes_df[conf.columns['class_orig']] = classes_df['MON_LC_GROUP']

    # Remove unneeded columns
    for column in classes_df.columns:
        if (column not in [conf.columns['class_orig'], crop_columnname]):
            classes_df.drop(column, axis=1, inplace=True)

    # Set the index
    classes_df.set_index(crop_columnname, inplace=True, verify_integrity=True)

    # Read the parcel data and do the necessary conversions
    #--------------------------------------------------------------------------
    logger.info(f"Read parceldata from {input_parcel_filepath}")
    if geofile_util.is_geofile(input_parcel_filepath):
        parceldata_df = geofile_util.read_file(input_parcel_filepath)
    else:
        parceldata_df = pdh.read_file(input_parcel_filepath)

    logger.info(f"Read Parceldata ready, info(): {parceldata_df.info()}")

    # Check if the id column is present...
    if conf.columns['id'] not in parceldata_df.columns:
        message = f"STOP: Column {conf.columns['id']} not found in input parcel file: {input_parcel_filepath}. Make sure the column is present or change the column name in the ini file.py"
        logger.critical(message)
        raise Exception(message)

    # Rename the column containing the object id to OBJECT_ID
#    df.rename(columns = {'CODE_OBJ':'object_id'}, inplace = True)

    # Convert the crop to unicode, in case the input is int...
    parceldata_df[crop_columnname] = parceldata_df[crop_columnname].astype('unicode')

    # Join/merge the classname
    logger.info('Add the classes to the parceldata')
    parceldata_df = parceldata_df.merge(classes_df, how='left',
                                        left_on=crop_columnname,
                                        right_index=True,
                                        validate='many_to_one')

    # Data verwijderen
#    df = df[df[classname] != 'Andere subsidiabele gewassen']  # geen boomkweek

    # Add column to signify that the parcel is eligible and set ineligible crop types (important
    # for reporting afterwards)
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    parceldata_df.insert(loc=0, column=conf.columns['is_eligible'], value=1)
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['1', '2']), conf.columns['is_eligible']] = 1

    # Add column to signify if the crop/class is permanent, so can/should be followed up in the
    # LPIS upkeep
    # TODO: would be cleaner if this is based on refe as well instead of hardcoded
    parceldata_df.insert(loc=0, column=conf.columns['is_permanent'], value=1)
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['1', '2', '3']), conf.columns['is_permanent']] = 1
    if 'GESP_PM' in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[parceldata_df['GESP_PM'].isin(['SER', 'SGM', 'LOO']), conf.columns['is_permanent']] = 1

    # Copy orig classname to classification classname
    parceldata_df.insert(loc=0, column=conf.columns['class'], value=parceldata_df[conf.columns['class_orig']])

    # If a column with extra info exists, use it as well to fine-tune the classification classes.
    if 'GESP_PM' in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[parceldata_df['GESP_PM'].isin(['SER', 'PLA', 'PLO']), conf.columns['class']] = 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS'
        parceldata_df.loc[parceldata_df['GESP_PM'].isin(['SGM', 'NPO', 'LOO', 'CON']), conf.columns['class']] = 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS'
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        #parceldata_df.loc[parceldata_df['GESP_PM'] == 'CIV', class_columnname] = 'MON_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning("The column 'GESP_PM' doesn't exist, so this part of the code was skipped!")

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info("Set classes that are still empty, not specific enough or that contain to little values to 'UNKNOWN'")
    parceldata_df.loc[parceldata_df[conf.columns['class']].str.startswith('nvt', na=True), conf.columns['class']] = 'MON_LC_UNKNOWN'

    # For columns that aren't needed for the classification:
    #    - Rename the ones interesting for interpretation
    #    - Drop the columns that aren't useful at all, except thoses excluded for export in the ini file
    for column in parceldata_df.columns:
        if column in (['GRAF_OPP', 'GWSCOD_H', 'GESP_PM']):
            if column == 'GESP_PM':
                parceldata_df['GESP_PM'] = parceldata_df['GESP_PM'].str.replace(',', ';')
            parceldata_df.rename(columns={column:'m__' + column}, inplace=True)            

        elif(column not in [conf.columns['id'], conf.columns['class']]
             and column not in conf.preprocess.getlist('extra_export_columns')
             and not column.startswith('m__')):
            parceldata_df.drop(column, axis=1, inplace=True)

    return parceldata_df

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
        parceldata_df = pd.read_csv(input_parcel_filepath)
    else:
        parceldata_df = gpd.read_file(input_parcel_filepath)
    logger.info(f'Read Parceldata ready, shape: {parceldata_df.shape}')

    # Rename the column containing the object id to OBJECT_ID
#    df.rename(columns = {'CODE_OBJ':'object_id'}, inplace = True)

    # Add columns for the class to use...
    parceldata_df.insert(0, conf.columns['class'], None)

    parceldata_df.loc[parceldata_df[crop_columnname].isin(['60', '700', '3432']), conf.columns['class']] = 'Grassland'  # Grassland
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['201', '202']), conf.columns['class']] = 'Maize'            # Maize
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['901', '904']), conf.columns['class']] = 'Potatoes'         # Potatoes
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['311', '36']), conf.columns['class']] = 'WinterWheat'       # Winter wheat of spelt
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['91']), conf.columns['class']] = 'SugarBeat'               # Sugar beat
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['321']), conf.columns['class']] = 'WinterBarley'           # Winter barley
    parceldata_df.loc[parceldata_df[crop_columnname].isin(['71']), conf.columns['class']] = 'FodderBeat'              # Fodder beat

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info("Set classes that are empty to 'IGNORE_UNIMPORTANT_CLASS' so they are ignored further on...")
    parceldata_df.loc[parceldata_df[conf.columns['class']].isnull(), conf.columns['class']] = 'IGNORE_UNIMPORTANT_CLASS'

    # Set small parcel to UNKNOWN as well so they are ignored as well...
#    logger.info("Set small parcel 'IGNORE_SMALL' so they are ignored further on...")
#    df.loc[df['GRAF_OPP'] <= 0.3, class_columnname] = 'IGNORE_SMALL'

    # For columns that aren't needed for the classification:
    #    - Rename the ones interesting for interpretation
    #    - Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if column in ['GRAF_OPP']:
            parceldata_df.rename(columns={column:'m__' + column}, inplace=True)
        elif (column not in [conf.columns['id'], conf.columns['class']]
              and column not in conf.preprocess.getlist('extra_export_columns')
              and (not column.startswith('m__'))):
            parceldata_df.drop(column, axis=1, inplace=True)

    # Return result
    return parceldata_df

# If the script is run directly...
if __name__ == "__main__":

    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
