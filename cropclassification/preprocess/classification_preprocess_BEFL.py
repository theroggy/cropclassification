# -*- coding: utf-8 -*-
"""
Create the parcel classes that will be used for the classification.

This implementation will create +- 40 classes.
parcel that don't have a clear classification in the input file get class 'UNKNOWN'.
"""

import logging
from pathlib import Path

import geofileops as gfo

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# Get a logger...
logger = logging.getLogger(__name__)

# define some specific BEFL column names
column_BEFL_earlylate = "MON_EARLY_LATE"  # Is it an early or a late crop?
column_BEFL_gesp_pm = "GESP_PM"  # Gespecialiseerde productiemethode
column_BEFL_gis_area = "GRAF_OPP"  # GIS Area
column_BEFL_status_perm_grass = "STAT_BGV"  # Status permanent grassland

ndvi_latecrop_count = "latecrop_ndvi_count"
ndvi_latecrop_median = "latecrop_ndvi_median"

# The real work
# -------------------------------------------------------------


def prepare_input(
    input_parcel_path: Path,
    classtype_to_prepare: str,
    classes_refe_path: Path,
    output_dir: Path,
):
    """
    This function creates a file that is compliant with the assumptions used by the
    rest of the classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be
          classified to
    """
    # Check if input parameters are OK
    if not input_parcel_path.exists():
        raise Exception(f"Input file doesn't exist: {input_parcel_path}")
    else:
        logger.info(f"Process input file {input_parcel_path}")

    # Read input file
    logger.info(f"Read parceldata from {input_parcel_path}")
    if gfo.is_geofile(input_parcel_path):
        parceldata_df = gfo.read_file(input_parcel_path)
    else:
        parceldata_df = pdh.read_file(input_parcel_path)
    logger.info(f"Read Parceldata ready, info(): {parceldata_df.info()}")

    # Check if the id column is present...
    if conf.columns["id"] not in parceldata_df.columns:
        message = (
            f"Column {conf.columns['id']} not found in input parcel file: "
            f"{input_parcel_path}. Make sure the column is present or change the "
            "column name in global_constants.py"
        )
        logger.critical(message)
        raise Exception(message)

    # Now start prepare
    if classtype_to_prepare == "CROPGROUP":
        parceldata_df = prepare_input_cropgroup(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_cropgroup(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CROPGROUP-GROUNDTRUTH":
        return prepare_input_cropgroup(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_gt_verified"],
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CROPGROUP-EARLY":
        parceldata_df = prepare_input_cropgroup_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_cropgroup_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CROPGROUP-EARLY-GROUNDTRUTH":
        return prepare_input_cropgroup_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_gt_verified"],
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LANDCOVER":
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "MULTICROP":
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LANDCOVER-GROUNDTRUTH":
        return prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_gt_verified"],
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LANDCOVER-EARLY":
        parceldata_df = prepare_input_landcover_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_landcover_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LANDCOVER-EARLY-GROUNDTRUTH":
        return prepare_input_landcover_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_gt_verified"],
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "POPULAR-CROP":
        parceldata_df = prepare_input_most_popular_crop(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_most_popular_crop(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "POPULAR-CROP-GROUNDTRUTH":
        return prepare_input_most_popular_crop(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_gt_verified"],
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LATECROP-EARLY":
        return prepare_input_latecrop_early(
            parceldata_df=parceldata_df,
            column_BEFL_latecrop=conf.columns["late_crop"],
            column_BEFL_latecrop2=conf.columns["late_crop2"],
            column_BEFL_maincrop=conf.columns["main_crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LATECROP":
        return prepare_input_latecrop(
            parceldata_df=parceldata_df,
            column_BEFL_latecrop=conf.columns["late_crop"],
            column_BEFL_latecrop2=conf.columns["late_crop2"],
            column_BEFL_maincrop=conf.columns["main_crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LATECROP-EARLY-GROUNDTRUTH":
        return prepare_input_latecrop_early(
            parceldata_df=parceldata_df,
            column_BEFL_latecrop=conf.columns["late_crop"],
            column_BEFL_latecrop2=conf.columns["late_crop2"],
            column_BEFL_maincrop=conf.columns["main_crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LATECROP-GROUNDTRUTH":
        return prepare_input_latecrop(
            parceldata_df=parceldata_df,
            column_BEFL_latecrop=conf.columns["crop_declared"],
            column_BEFL_latecrop2=column_BEFL_latecrop2,
            column_BEFL_maincrop=conf.columns["main_crop_declared"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "FABACEAE":
        parceldata_df = prepare_input_fabaceae(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_fabaceae(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop"],
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "FABACEAE-GROUNDTRUTH":
        return prepare_input_fabaceae(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=conf.columns["crop_gt_verified"],
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    else:
        message = (
            f"Unknown value for parameter classtype_to_prepare: {classtype_to_prepare}"
        )
        logger.critical(message)
        raise Exception(message)


def prepare_input_cropgroup(
    parceldata_df,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    This function creates a file that is compliant with the assumptions used by the rest
    of the classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be
          classified to

    This specific implementation converts the typical export format used in BE-Flanders
    to this format.
    """
    # Check if parameters are OK and init some extra params
    # --------------------------------------------------------------------------
    if not classes_refe_path.exists():
        raise Exception(f"Input classes file doesn't exist: {classes_refe_path}")

    # Convert the crop to unicode, in case the input is int...
    if column_BEFL_cropcode in parceldata_df.columns:
        parceldata_df[column_BEFL_cropcode] = parceldata_df[
            column_BEFL_cropcode
        ].astype("unicode")

    # Read and cleanup the mapping table from crop codes to classes
    # --------------------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)
    logger.info(f"Read classes conversion table ready, info(): {classes_df.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("unicode")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    columns_to_keep = conf.preprocess.getlist("columns_to_keep")
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_to_keep
        ):
            classes_df.drop(column, axis=1, inplace=True)

    # Set the index
    classes_df.set_index(column_BEFL_cropcode, inplace=True, verify_integrity=True)

    # Get only the columns in the classes_df that don't exist yet in parceldata_df
    cols_to_join = classes_df.columns.difference(parceldata_df.columns)

    # Join/merge the classname
    logger.info("Add the classes to the parceldata")
    parceldata_df = parceldata_df.merge(
        classes_df[cols_to_join],
        how="left",
        left_on=column_BEFL_cropcode,
        right_index=True,
        validate="many_to_one",
    )

    # Copy orig classname to classification classname
    parceldata_df.insert(
        loc=0, column=column_output_class, value=parceldata_df[column_output_class_orig]
    )

    # For rows with no class, set to UNKNOWN
    parceldata_df.fillna(value={column_output_class: "UNKNOWN"}, inplace=True)

    # If a column with extra info exists, use it as well to fine-tune the classification
    # classes.
    if column_BEFL_gesp_pm in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["SER", "SGM"]), column_output_class
        ] = "MON_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"]),
            column_output_class,
        ] = "MON_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "LOO", column_output_class
        ] = "MON_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "CON", column_output_class
        ] = "MON_CONTAINERS"  # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == "CIV", class_columnname
        # ] = "MON_CONTAINERS"  # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so this part of the code "
            "was skipped!"
        )

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    # logger.info(
    #     "Set classes that are still empty, not specific enough or that contain to "
    #     "little values to 'UNKNOWN'"
    # )
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].str.startswith('nvt', na=True),
    #     column_output_class
    # ] = 'UNKNOWN'

    # 'MON_ANDERE_SUBSID_GEWASSEN': very low classification rate (< 1%), as it is a
    #     group with several very different classes in it
    # 'MON_AARDBEIEN': low classification rate (~10%), as many parcel actually are
    #     temporary greenhouses but aren't correctly applied
    # 'MON_BRAAK': very low classification rate (< 1%), spread over a lot of classes,
    #     but most popular are MON_BOOM, MON_GRASSEN, MON_FRUIT
    # 'MON_KLAVER': log classification rate (25%), spread over quite some classes, but
    #     MON GRASSES has 20% as well.
    # 'MON_MENGSEL': 25% correct classifications: rest spread over many other classes.
    #     Too heterogenous in group?
    # 'MON_POEL': 0% correct classifications: most are classified as MON_CONTAINER,
    #     MON_FRUIT. Almost nothing was misclassified as being POEL
    # 'MON_RAAPACHTIGEN': 25% correct classifications: rest spread over many other
    #     classes
    # 'MON_STRUIK': 10%
    #    TODO: nakijken, wss opsplitsen of toevoegen aan MON_BOOMKWEEK???
    # classes_badresults = [
    #     'MON_ANDERE_SUBSID_GEWASSEN', 'MON_AARDBEIEN', 'MON_BRAAK', 'MON_KLAVER',
    #     'MON_MENGSEL', 'MON_POEL', 'MON_RAAPACHTIGEN', 'MON_STRUIK'
    # ]
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(classes_badresults),
    #     column_output_class
    # ] = 'UNKNOWN'

    # MON_BONEN en MON_WIKKEN have omongst each other a very large percentage of false
    # positives/negatives, so they seem very similar... lets create a class that
    # combines both
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(['MON_BONEN', 'MON_WIKKEN']),
    #     column_output_class
    # ] = 'MON_BONEN_WIKKEN'

    # MON_BOOM includes now also the growing new plants/trees, which is too differenct
    # from grown trees -> put growing new trees is seperate group
    # parceldata_df.loc[
    #     parceldata_df[column_BEFL_cropcode].isin(['9602', '9603', '9604', '9560']),
    #     column_output_class
    # ] = 'MON_BOOMKWEEK'

    # 'MON_FRUIT': has a good accuracy (91%), but also has as much false positives
    #     (115% -> mainly 'MON_GRASSEN' that are (mis)classified as 'MON_FRUIT')
    # 'MON_BOOM': has very bad accuracy (28%) and also very much false positives
    #     (450% -> mainly 'MON_GRASSEN' that are misclassified as 'MON_BOOM')
    # MON_FRUIT and MON_BOOM are permanent anyway, so not mandatory that they are
    # checked in monitoring process.
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE_DIFFICULT_PERMANENT_CLASS
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(['MON_BOOM', 'MON_FRUIT']),
    #     column_output_class
    # ] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'

    # Set classes with very few elements to IGNORE_NOT_ENOUGH_SAMPLES!
    for _, row in (
        parceldata_df.groupby(column_output_class)
        .size()
        .reset_index(name="count")
        .iterrows()
    ):
        if row["count"] <= conf.preprocess.getint("min_parcels_in_class"):
            logger.info(
                f"Class <{row[column_output_class]}> only contains {row['count']} "
                f"elements, so put them to IGNORE_NOT_ENOUGH_SAMPLES"
            )
            parceldata_df.loc[
                parceldata_df[column_output_class] == row[column_output_class],
                column_output_class,
            ] = "IGNORE_NOT_ENOUGH_SAMPLES"

    # Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if (
            column
            not in [
                column_output_class,
                conf.columns["id"],
                conf.columns["class_groundtruth"],
                conf.columns["class_declared"],
            ]
            and column not in conf.preprocess.getlist("extra_export_columns")
            and column not in columns_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_fabaceae(
    parceldata_df,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    This function creates a file that is compliant with the assumptions used by the rest
    of the classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be
          classified to

    This specific implementation converts the typical export format used in BE-Flanders
    to this format.
    """
    # Check if parameters are OK and init some extra params
    # --------------------------------------------------------------------------
    if not classes_refe_path.exists():
        raise Exception(f"Input classes file doesn't exist: {classes_refe_path}")

    # Convert the crop to unicode, in case the input is int...
    if column_BEFL_cropcode in parceldata_df.columns:
        parceldata_df[column_BEFL_cropcode] = parceldata_df[
            column_BEFL_cropcode
        ].astype("unicode")

    # Read and cleanup the mapping table from crop codes to classes
    # --------------------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)
    logger.info(f"Read classes conversion table ready, info(): {classes_df.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("unicode")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    columns_to_keep = conf.preprocess.getlist("columns_to_keep")
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_to_keep
        ):
            classes_df.drop(column, axis=1, inplace=True)

    # Set the index
    classes_df.set_index(column_BEFL_cropcode, inplace=True, verify_integrity=True)

    # Get only the columns in the classes_df that don't exist yet in parceldata_df
    cols_to_join = classes_df.columns.difference(parceldata_df.columns)

    # Filter the parcels to only keep the classes we are interested in
    # TODO: this shouldn't be hardcoded!
    parceldata_df = parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["60", "700", "660", "732", "723"])
    ]

    # Join/merge the classname
    logger.info("Add the classes to the parceldata")
    parceldata_df = parceldata_df.merge(
        classes_df[cols_to_join],
        how="left",
        left_on=column_BEFL_cropcode,
        right_index=True,
        validate="many_to_one",
    )

    # Copy orig classname to classification classname
    parceldata_df.insert(
        loc=0, column=column_output_class, value=parceldata_df[column_output_class_orig]
    )

    # For rows with no class, set to UNKNOWN
    parceldata_df.fillna(value={column_output_class: "UNKNOWN"}, inplace=True)

    # If a column with extra info exists, use it as well to fine-tune the classification
    # classes.
    if column_BEFL_gesp_pm in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["SER", "SGM"]), column_output_class
        ] = "MON_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"]),
            column_output_class,
        ] = "MON_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "LOO", column_output_class
        ] = "MON_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "CON", column_output_class
        ] = "MON_CONTAINERS"  # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == 'CIV', class_columnname
        # ] = 'MON_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so this part of the code "
            "was skipped!"
        )

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    # logger.info(
    #     "Set classes that are still empty, not specific enough or that contain to "
    #     "little values to 'UNKNOWN'"
    # )
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].str.startswith('nvt', na=True),
    #     column_output_class
    # ] = 'UNKNOWN'

    # 'MON_ANDERE_SUBSID_GEWASSEN': very low classification rate (< 1%), as it is a
    #     group with several very different classes in it
    # 'MON_AARDBEIEN': low classification rate (~10%), as many parcel actually are
    #     temporary greenhouses but aren't correctly applied
    # 'MON_BRAAK': very low classification rate (< 1%), spread over a lot of classes,
    #     but most popular are MON_BOOM, MON_GRASSEN, MON_FRUIT
    # 'MON_KLAVER': log classification rate (25%), spread over quite some classes,
    #     but MON GRASSES has 20% as well.
    # 'MON_MENGSEL': 25% correct classifications: rest spread over many other classes.
    #     Too heterogenous in group?
    # 'MON_POEL': 0% correct classifications: most are classified as MON_CONTAINER,
    #     MON_FRUIT. Almost nothing was misclassified as being POEL
    # 'MON_RAAPACHTIGEN': 25% correct classifications: rest spread over many other
    #     classes
    # 'MON_STRUIK': 10%
    #    TODO: nakijken, wss opsplitsen of toevoegen aan MON_BOOMKWEEK???
    # classes_badresults = [
    #     'MON_ANDERE_SUBSID_GEWASSEN', 'MON_AARDBEIEN', 'MON_BRAAK', 'MON_KLAVER',
    #     'MON_MENGSEL', 'MON_POEL', 'MON_RAAPACHTIGEN', 'MON_STRUIK'
    # ]
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(classes_badresults),
    #     column_output_class
    # ] = 'UNKNOWN'

    # MON_BONEN en MON_WIKKEN have omongst each other a very large percentage of false
    # positives/negatives, so they seem very similar... lets create a class that
    # combines both
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(['MON_BONEN', 'MON_WIKKEN']),
    #     column_output_class
    # ] = 'MON_BONEN_WIKKEN'

    # MON_BOOM includes now also the growing new plants/trees, which is too different
    # from grown trees -> put growing new trees is seperate group
    # parceldata_df.loc[
    #     parceldata_df[column_BEFL_cropcode].isin(['9602', '9603', '9604', '9560']),
    #     column_output_class
    # ] = 'MON_BOOMKWEEK'

    # 'MON_FRUIT': has a good accuracy (91%), but also has as much false positives
    #     (115% -> mainly 'MON_GRASSEN' that are (mis)classified as 'MON_FRUIT')
    # 'MON_BOOM': has very bad accuracy (28%) and also very much false positives
    #     (450% -> mainly 'MON_GRASSEN' that are misclassified as 'MON_BOOM')
    # MON_FRUIT and MON_BOOM are permanent anyway, so not mandatory that they are
    # checked in monitoring process.
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE_DIFFICULT_PERMANENT_CLASS
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(['MON_BOOM', 'MON_FRUIT']),
    #     column_output_class
    # ] = 'IGNORE_DIFFICULT_PERMANENT_CLASS'

    # Set classes with very few elements to IGNORE_NOT_ENOUGH_SAMPLES!
    for _, row in (
        parceldata_df.groupby(column_output_class)
        .size()
        .reset_index(name="count")
        .iterrows()
    ):
        if row["count"] <= conf.preprocess.getint("min_parcels_in_class"):
            logger.info(
                f"Class <{row[column_output_class]}> only contains {row['count']} "
                "elements, so put them to IGNORE_NOT_ENOUGH_SAMPLES"
            )
            parceldata_df.loc[
                parceldata_df[column_output_class] == row[column_output_class],
                column_output_class,
            ] = "IGNORE_NOT_ENOUGH_SAMPLES"

    # Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if (
            column
            not in [
                column_output_class,
                conf.columns["id"],
                conf.columns["class_groundtruth"],
                conf.columns["class_declared"],
            ]
            and column not in conf.preprocess.getlist("extra_export_columns")
            and column not in columns_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_latecrop(
    parceldata_df,
    column_BEFL_latecrop: str,
    column_BEFL_latecrop2: str,
    column_BEFL_maincrop: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    This function creates a file that is compliant with the assumptions used by the rest
    of the classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be
          classified to

    This specific implementation converts the typical export format used in BE-Flanders
    to this format.
    """
    # Check if parameters are OK and init some extra params
    # -----------------------------------------------------
    if not classes_refe_path.exists():
        raise Exception(f"Input classes file doesn't exist: {classes_refe_path}")

    # Convert the crops to unicode, in case the input is int
    if column_BEFL_latecrop in parceldata_df.columns:
        parceldata_df[column_BEFL_latecrop] = parceldata_df[
            column_BEFL_latecrop
        ].astype("unicode")
    if column_BEFL_latecrop2 in parceldata_df.columns:
        parceldata_df[column_BEFL_latecrop2] = parceldata_df[
            column_BEFL_latecrop2
        ].astype("unicode")
    if column_BEFL_maincrop in parceldata_df.columns:
        parceldata_df[column_BEFL_maincrop] = parceldata_df[
            column_BEFL_maincrop
        ].astype("unicode")

    # Read and cleanup the mapping table from crop codes to classes
    # -------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)
    logger.info(f"Read classes conversion table ready, info(): {classes_df.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_latecrop] = classes_df["CROPCODE"].astype("unicode")
    classes_df[column_BEFL_latecrop2] = classes_df["CROPCODE"].astype("unicode")
    classes_df[column_BEFL_maincrop] = classes_df["CROPCODE"].astype("unicode")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    columns_to_keep = conf.preprocess.getlist("columns_to_keep")
    for column in classes_df.columns:
        if (
            column
            not in [
                column_output_class_orig,
                column_BEFL_latecrop,
                column_BEFL_latecrop2,
                column_BEFL_maincrop,
                column_BEFL_status_perm_grass,
                conf.columns["class_declared"],
                conf.columns["class"],
                ndvi_latecrop_count,
                ndvi_latecrop_median,
            ]
            and column not in columns_to_keep
            and column not in ["MON_CROPGROUP", "IS_PERM_BEDEKKING", "IS_BOUWLAND"]
        ):
            classes_df.drop(column, axis=1, inplace=True)

    # Set the index
    classes_df.set_index(column_BEFL_latecrop, inplace=True, verify_integrity=True)

    # Get only the columns in the classes_df that don't exist yet in parceldata_df
    cols_to_join = classes_df.columns.difference(parceldata_df.columns)

    # Join/merge the classname
    logger.info("Add the classes to the parceldata")
    parceldata_df = parceldata_df.merge(
        classes_df[cols_to_join],
        how="left",
        left_on=column_BEFL_latecrop,
        right_index=True,
        validate="many_to_one",
    )
    parceldata_df = parceldata_df.merge(
        classes_df[cols_to_join],
        how="left",
        left_on=column_BEFL_latecrop2,
        right_index=True,
        validate="many_to_one",
        suffixes=(None, "_LATECROP2"),
    )
    parceldata_df = parceldata_df.merge(
        classes_df[cols_to_join],
        how="left",
        left_on=column_BEFL_maincrop,
        right_index=True,
        validate="many_to_one",
        suffixes=(None, "_MAINCROP"),
    )

    # Copy orig classname to classification classname
    parceldata_df.insert(
        loc=0, column=column_output_class, value=parceldata_df[column_output_class_orig]
    )

    # For rows with no class, set to UNKNOWN
    parceldata_df.fillna(value={column_output_class: "UNKNOWN"}, inplace=True)

    # Add ignore_for_training column: if 1, ignore for training
    parceldata_df["ignore_for_training"] = 0

    # If ndvi_latecrop_count data columns available, use them
    if ndvi_latecrop_count in parceldata_df.columns:
        # If no NDVI data avalable, not possible to determine bare soil
        # -> ignore parcel in training
        parceldata_df.loc[
            parceldata_df[ndvi_latecrop_count] < 10, "ignore_for_training"
        ] = 1
    else:
        logger.warning(
            f"No column {ndvi_latecrop_count} available to set ignore_for_training"
        )

    # TODO: MON_BRAAK opkuisen met Timo?
    # TODO: controle obv ndvi > 0.3 => geen braak?
    #       braak van in de zomer => geen "braak" meer vanwege onkruid

    """
    # If permanent grassland, there will typically still be grass on the parcels
    # BUT: 60 is in ECO_430_GRASLAND_KLAVER_LUZ_GRAAN!!!
    parceldata_df.loc[
        (parceldata_df[column_output_class] == "UNKNOWN")
        & (parceldata_df[column_BEFL_status_perm_grass] is not None)
        & (parceldata_df["MON_CROPGROUP_MAINCROP"] == "MON_GRASSEN"),
        column_output_class,
    ] = "MON_GRASSEN"
    """

    # Assumption: permanent crops except grassland won't/can't have a latecrop, so
    # don't bother predicting them
    parceldata_df.loc[
        (parceldata_df[column_output_class] == "UNKNOWN")
        # & (parceldata_df[column_BEFL_status_perm_grass] is None)
        & (parceldata_df["MON_CROPGROUP_MAINCROP"] != "MON_GRASSEN")
        & (parceldata_df["IS_PERM_BEDEKKING_MAINCROP"] == 1),
        column_output_class,
    ] = "IGNORE_PERMANENT_BEDEKKING"

    # If a column with extra info exists, use it as well to fine-tune the
    # classification classes.
    if column_BEFL_gesp_pm in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm].isin(["SER", "SGM"])),
            column_output_class,
        ] = "MON_OVERK_LOO"
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"])),
            column_output_class,
        ] = "MON_OVERK_LOO"
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm] == "LOO"),
            column_output_class,
        ] = "MON_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm] == "CON"),
            column_output_class,
        ] = "MON_CONTAINERS"  # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == 'CIV', class_columnname
        # ] = 'MON_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so couldn't be used!"
        )

    # Set classes with very few elements to IGNORE_NOT_ENOUGH_SAMPLES!
    for _, row in (
        parceldata_df.groupby(column_output_class)
        .size()
        .reset_index(name="count")
        .iterrows()
    ):
        if row["count"] <= conf.preprocess.getint("min_parcels_in_class"):
            logger.info(
                f"Class <{row[column_output_class]}> only contains {row['count']} "
                "elements, so put them to IGNORE_NOT_ENOUGH_SAMPLES"
            )
            parceldata_df.loc[
                parceldata_df[column_output_class] == row[column_output_class],
                column_output_class,
            ] = "IGNORE_NOT_ENOUGH_SAMPLES"

    # Add copy of class as class_declared
    parceldata_df[conf.columns["class_declared"]] = parceldata_df[column_output_class]

    # If the median ndvi <= 0.3 parcel is still a bare field (for training)
    if ndvi_latecrop_median in parceldata_df.columns:
        # If also no grass, train as MON_BRAAK
        parceldata_df.loc[
            (parceldata_df[ndvi_latecrop_median] <= 0.3)
            # & (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df["MON_CROPGROUP_MAINCROP"] != "MON_GRASSEN"),
            column_output_class,
        ] = "MON_BRAAK"

        # For all other classes, don't use for training if bare soil
        parceldata_df.loc[
            (parceldata_df[ndvi_latecrop_median] <= 0.3)
            & (parceldata_df[column_output_class] != "MON_BRAAK"),
            "ignore_for_training",
        ] = 1
    else:
        logger.warning(
            f"No column {ndvi_latecrop_count} available to set ignore_for_training"
        )

    # Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if (
            column
            not in [
                column_output_class,
                conf.columns["id"],
                conf.columns["class_groundtruth"],
                conf.columns["class_declared"],
                ndvi_latecrop_count,
                ndvi_latecrop_median,
                "ignore_for_training",
            ]
            and column not in conf.preprocess.getlist("extra_export_columns")
            and column not in columns_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_latecrop_early(
    parceldata_df,
    column_BEFL_latecrop: str,
    column_BEFL_latecrop2: str,
    column_BEFL_maincrop: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    Prepare a dataframe based on the BEFL input file so it onclused a classname
    column ready to classify the landcover for early crops.
    """
    # First run the standard latecrop prepare
    parceldata_df = prepare_input_latecrop(
        parceldata_df=parceldata_df,
        column_BEFL_latecrop=column_BEFL_latecrop,
        column_BEFL_latecrop2=column_BEFL_latecrop2,
        column_BEFL_maincrop=column_BEFL_maincrop,
        column_output_class=column_output_class,
        classes_refe_path=classes_refe_path,
    )

    # Set crops not in early crops to ignore
    # 396: mengteelt mais en vlinderbloemigen
    # "71", "91", "8532", "9532": bieten
    # TODO: should be in REFE instead of hardcoded!!!
    parceldata_df.loc[
        parceldata_df[column_BEFL_maincrop].isin(
            ["396", "201", "202", "71", "91", "8532", "9532"]
        ),
        column_output_class,
    ] = "IGNORE_LATE_MAINCROP"
    parceldata_df.loc[
        parceldata_df[column_BEFL_maincrop].isin(
            ["396", "201", "202", "71", "91", "8532", "9532"]
        ),
        conf.columns["class_declared"],
    ] = "IGNORE_LATE_MAINCROP"

    return parceldata_df


def prepare_input_cropgroup_early(
    parceldata_df,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    Prepare a dataframe based on the BEFL input file so it onclused a classname
    column ready to classify the cropgroups for early crops.
    """
    # First run the standard landcover prepare
    parceldata_df = prepare_input_cropgroup(
        parceldata_df, column_BEFL_cropcode, column_output_class, classes_refe_path
    )

    # Set late crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE_LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE_NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}!"
        )

    return parceldata_df


def prepare_input_landcover(
    parceldata_df,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    This function creates a file that is compliant with the assumptions used by the rest
    of the classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be
          classified to

    This specific implementation converts the typical export format used in BE-Flanders
    to this format.
    """
    # Check if parameters are OK and init some extra params
    # -----------------------------------------------------
    if not classes_refe_path.exists():
        raise Exception(f"Input classes file doesn't exist: {classes_refe_path}")

    # Convert the crop to unicode, in case the input is int...
    if column_BEFL_cropcode in parceldata_df.columns:
        parceldata_df[column_BEFL_cropcode] = parceldata_df[
            column_BEFL_cropcode
        ].astype("unicode")

    # Read and cleanup the mapping table from crop codes to classes
    # -------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)
    logger.info(f"Read classes conversion table ready, info(): {classes_df.info()}")

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("unicode")

    # Map column MON_group to orig classname
    column_output_class_orig = column_output_class + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    columns_to_keep = conf.preprocess.getlist("columns_to_keep")
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_to_keep
        ):
            classes_df.drop(column, axis=1, inplace=True)

    # Set the index
    classes_df.set_index(column_BEFL_cropcode, inplace=True, verify_integrity=True)

    # Get only the columns in the classes_df that don't exist yet in parceldata_df
    cols_to_join = classes_df.columns.difference(parceldata_df.columns)

    # Join/merge the classname
    logger.info("Add the classes to the parceldata")
    parceldata_df = parceldata_df.merge(
        classes_df[cols_to_join],
        how="left",
        left_on=column_BEFL_cropcode,
        right_index=True,
        validate="many_to_one",
    )

    # Copy orig classname to classification classname
    parceldata_df.insert(
        loc=0, column=column_output_class, value=parceldata_df[column_output_class_orig]
    )

    # If a column with extra info exists, use it as well to fine-tune the classification
    # classes.
    if column_BEFL_gesp_pm in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(
                ["SER", "PLA", "PLO", "SGM", "NPO", "LOO"]
            ),
            column_output_class,
        ] = "MON_LC_OVERK_LOO"
        # Containers
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["CON"]), column_output_class
        ] = "MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS"
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == "CIV", class_columnname
        # ] = "MON_CONTAINERS"  # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so this part of the code "
            "was skipped!"
        )

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info(
        "Set classes that are still empty, not specific enough or that contain to "
        "little values to 'UNKNOWN'"
    )
    parceldata_df.loc[
        parceldata_df[column_output_class].str.startswith("nvt", na=True),
        column_output_class,
    ] = "MON_LC_UNKNOWN"

    # Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if (
            column
            not in [
                column_output_class,
                conf.columns["id"],
                conf.columns["class_groundtruth"],
                conf.columns["class_declared"],
            ]
            and column not in conf.preprocess.getlist("extra_export_columns")
            and column not in columns_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_landcover_early(
    parceldata_df,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    Prepare a dataframe based on the BEFL input file so it onclused a classname
    column ready to classify the landcover for early crops.
    """
    # First run the standard landcover prepare
    parceldata_df = prepare_input_landcover(
        parceldata_df, column_BEFL_cropcode, column_output_class, classes_refe_path
    )

    # Set crops not in early crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE_LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE_NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}, so "
            "new grassland cannot be ignored!"
        )

    return parceldata_df


def prepare_input_most_popular_crop(
    parceldata_df,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
):
    """
    This function creates a file that is compliant with the assumptions used by the rest
    of the classification functionality.

    It should be a csv file with the following columns:
        - object_id: column with a unique identifier
        - classname: a string column with a readable name of the classes that will be
          classified to

    This specific implementation converts the typical export format used in BE-Flanders
    to this format.
    """

    # Add columns for the class to use...
    parceldata_df.insert(0, column_output_class, None)

    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["60", "700", "3432"]),
        column_output_class,
    ] = "Grassland"  # Grassland
    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["201", "202"]), column_output_class
    ] = "Maize"  # Maize
    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["901", "904"]), column_output_class
    ] = "Potatoes"  # Potatoes
    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["311", "36"]), column_output_class
    ] = "WinterWheat"  # Winter wheat of spelt
    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["91"]), column_output_class
    ] = "SugarBeat"  # Sugar beat
    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["321"]), column_output_class
    ] = "WinterBarley"  # Winter barley
    parceldata_df.loc[
        parceldata_df[column_BEFL_cropcode].isin(["71"]), column_output_class
    ] = "FodderBeat"  # Fodder beat

    # Some extra cleanup: classes starting with 'nvt' or empty ones
    logger.info(
        "Set classes that are empty to 'IGNORE_UNIMPORTANT_CLASS' so they are ignored "
        "further on..."
    )
    parceldata_df.loc[
        parceldata_df[column_output_class].isnull(), column_output_class
    ] = "IGNORE_UNIMPORTANT_CLASS"

    # Drop the columns that aren't useful at all
    columns_to_keep = conf.preprocess.getlist("columns_to_keep")
    for column in parceldata_df.columns:
        if (
            column not in [conf.columns["id"], column_output_class]
            and column not in conf.preprocess.getlist("extra_export_columns")
            and column not in columns_to_keep
            and not column == column_BEFL_cropcode
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    # Return result
    return parceldata_df
