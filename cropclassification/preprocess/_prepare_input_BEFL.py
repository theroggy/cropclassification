"""Create the parcel classes that will be used for the classification.

This implementation will create +- 40 classes.
parcel that don't have a clear classification in the input file get class 'UNKNOWN'.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from typing_extensions import deprecated

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

# Get a logger...
logger = logging.getLogger(__name__)

# define some specific BEFL column names
column_BEFL_earlylate = "MON_VROEG_LAAT"  # Is it an early or a late crop?
column_BEFL_gesp_pm = "GESP_PM"  # Gespecialiseerde productiemethode
column_BEFL_gis_area = "GRAF_OPP"  # GIS Area
column_BEFL_status_perm_grass = "STAT_BGV"  # Status permanent grassland
column_BEFL_crop = "GWSCOD_H"
column_BEFL_crop_declared = "GWSCOD_H_A"
column_BEFL_crop_gt_verified = "HOOFDTEELT_CTRL_COD"
column_BEFL_crop_gt_unverified = "HOOFDTEELT_CTRL_COD_ORIG"
column_BEFL_latecrop = "GWSCOD_N"
column_BEFL_latecrop2 = "GWSCOD_N2"
column_BEFL_latecrop_gt_verified = "NATEELT_CTRL_COD"
column_BEFL_latecrop_gt_unverified = "NATEELT_CTRL_COD_ORIG"
column_BEFL_latecrop2_gt_verified = "NATEELT2_CTRL_COD"
column_BEFL_latecrop2_gt_unverified = "NATEELT2_CTRL_COD_ORIG"

# BEFL specific columns we want keep
columns_BEFL_to_keep = [
    column_BEFL_earlylate,
    column_BEFL_gesp_pm,
    column_BEFL_gis_area,
    column_BEFL_status_perm_grass,
    column_BEFL_crop_gt_verified,
    column_BEFL_crop_gt_unverified,
    column_BEFL_crop,
    column_BEFL_crop_declared,
    column_BEFL_latecrop,
    column_BEFL_latecrop2,
    column_BEFL_latecrop_gt_verified,
    column_BEFL_latecrop_gt_unverified,
    column_BEFL_latecrop2_gt_verified,
    column_BEFL_latecrop2_gt_unverified,
    "NATEELT_CTRL_DATUM",
]

ndvi_latecrop_count = "latecrop_ndvi_count"
ndvi_latecrop_median = "latecrop_ndvi_median"

# Set parcels having a main crop that stays late on the field to another class, as
# the main crop will still be on the field:
# TODO: should be in REFE instead of hardcoded!!!
late_main_crops = {
    "MON_CG_BIETEN": ["71", "91", "8532", "9532"],
    "MON_CG_MAIS_OF_MENGTEELT": ["201", "202", "396"],
}


def prepare_input(
    input_parcel_path: Path,
    classtype_to_prepare: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
) -> pd.DataFrame:
    """Creates a file for use in the marker_cropclass functionality.

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
    parceldata_df = pdh.read_file(input_parcel_path)
    logger.info(f"Read Parceldata ready, {len(parceldata_df)=}")

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
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_cropgroup(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "CROPGROUP-GROUNDTRUTH":
        parceldata_df = prepare_input_cropgroup(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    elif classtype_to_prepare == "CROPGROUP-EARLY":
        parceldata_df = prepare_input_cropgroup_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_cropgroup_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "CROPGROUP-EARLY-GROUNDTRUTH":
        parceldata_df = prepare_input_cropgroup_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    elif classtype_to_prepare == "CROPROTATION":
        parceldata_df = prepare_input_croprotation(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_croprotation(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CROPROTATION-GROUNDTRUTH":
        return prepare_input_croprotation(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CROPROTATION-EARLY":
        parceldata_df = prepare_input_croprotation_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_croprotation_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CROPROTATION-EARLY-GROUNDTRUTH":
        return prepare_input_croprotation_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CARBONSUPPLY":
        parceldata_df = prepare_input_carbonsupply(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_carbonsupply(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CARBONSUPPLY-GROUNDTRUTH":
        return prepare_input_carbonsupply(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CARBONSUPPLY-EARLY":
        parceldata_df = prepare_input_carbonsupply_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
        )
        return prepare_input_carbonsupply_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "CARBONSUPPLY-EARLY-GROUNDTRUTH":
        return prepare_input_carbonsupply_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
        )
    elif classtype_to_prepare == "LANDCOVER":
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "MULTICROP":
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "LANDCOVER-GROUNDTRUTH":
        parceldata_df = prepare_input_landcover(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    elif classtype_to_prepare == "LANDCOVER-EARLY":
        parceldata_df = prepare_input_landcover_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_landcover_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "LANDCOVER-EARLY-GROUNDTRUTH":
        parceldata_df = prepare_input_landcover_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    elif classtype_to_prepare.startswith("LATECROP"):
        if classtype_to_prepare.startswith("LATECROP-EARLY"):
            scope = "EARLY_MAINCROP"
        elif classtype_to_prepare.startswith("LATECROP-LATE"):
            scope = "LATE_MAINCROP"
        elif classtype_to_prepare.startswith("LATECROP-ALL"):
            scope = "ALL"
        else:
            raise ValueError(f"Unknown late crop scope: {classtype_to_prepare}")

        is_groundtruth = classtype_to_prepare.endswith("-GROUNDTRUTH")
        if not is_groundtruth:
            column_latecrop = column_BEFL_latecrop
            column_latecrop2 = column_BEFL_latecrop2
            column_output_class = conf.columns["class"]
        else:
            column_latecrop = column_BEFL_latecrop_gt_verified
            column_latecrop2 = None
            column_output_class = conf.columns["class_groundtruth"]

        parceldata_df = prepare_input_latecrop(
            parceldata_df=parceldata_df,
            column_BEFL_latecrop=column_latecrop,
            column_BEFL_latecrop2=column_latecrop2,
            column_BEFL_maincrop=column_BEFL_crop,
            column_output_class=column_output_class,
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=is_groundtruth,
            scope=scope,  # type: ignore[arg-type]
        )
    elif classtype_to_prepare == "FABACEAE":
        parceldata_df = prepare_input_fabaceae(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_fabaceae(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "FABACEAE-GROUNDTRUTH":
        parceldata_df = prepare_input_fabaceae(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    elif classtype_to_prepare == "RUGGENTEELT":
        parceldata_df = prepare_input_ruggenteelt(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_ruggenteelt(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "RUGGENTEELT-GROUNDTRUTH":
        parceldata_df = prepare_input_ruggenteelt(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    elif classtype_to_prepare == "RUGGENTEELT-EARLY":
        parceldata_df = prepare_input_ruggenteelt_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_declared,
            column_output_class=conf.columns["class_declared"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=1,
            is_groundtruth=False,
        )
        parceldata_df = prepare_input_ruggenteelt_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop,
            column_output_class=conf.columns["class"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=False,
        )
    elif classtype_to_prepare == "RUGGENTEELT-EARLY-GROUNDTRUTH":
        parceldata_df = prepare_input_ruggenteelt_early(
            parceldata_df=parceldata_df,
            column_BEFL_cropcode=column_BEFL_crop_gt_verified,
            column_output_class=conf.columns["class_groundtruth"],
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            is_groundtruth=True,
        )
    else:
        message = (
            f"Unknown value for parameter classtype_to_prepare: {classtype_to_prepare}"
        )
        logger.critical(message)
        raise ValueError(message)

    return parceldata_df


def prepare_input_cropgroup(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the cropgroup marker.

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
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # --------------------------------------------------------------------------
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("string")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_BEFL_to_keep
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
        ] = "MON_CG_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"]),
            column_output_class,
        ] = "MON_CG_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "LOO", column_output_class
        ] = "MON_CG_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "CON", column_output_class
        ] = "MON_CG_CONTAINERS"  # Containers, niet op volle grond...
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

    # 'MON_CG_ANDERE_SUBSID_GEWASSEN': very low classification rate (< 1%), as it is a
    #     group with several very different classes in it
    # 'MON_CG_AARDBEIEN': low classification rate (~10%), as many parcel actually are
    #     temporary greenhouses but aren't correctly applied
    # 'MON_CG_BRAAK': very low classification rate (< 1%), spread over a lot of classes,
    #     but most popular are MON_BOOM, MON_GRASSEN, MON_FRUIT
    # 'MON_CG_KLAVER': log classification rate (25%), spread over quite some classes,
    #     but MON GRASSES has 20% as well.
    # 'MON_CG_MENGSEL': 25% correct classifications: rest spread over many other classes
    #     Too heterogenous in group?
    # 'MON_CG_POEL': 0% correct classifications: most are classified as MON_CONTAINER,
    #     MON_FRUIT. Almost nothing was misclassified as being POEL
    # 'MON_CG_RAAPACHTIGEN': 25% correct classifications: rest spread over many other
    #     classes
    # 'MON_CG_STRUIK': 10%
    #    TODO: nakijken, wss opsplitsen of toevoegen aan MON_BOOMKWEEK???
    # classes_badresults = [
    #     'MON_CG_ANDERE_SUBSID_GEWASSEN', 'MON_CG_AARDBEIEN',
    #     'MON_CG_BRAAK', 'MON_CG_KLAVER', 'MON_CG_MENGSEL',
    #     'MON_CG_POEL', 'MON_CG_RAAPACHTIGEN', 'MON_CG_STRUIK'
    # ]
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(classes_badresults),
    #     column_output_class
    # ] = 'UNKNOWN'

    # MON_BONEN en MON_WIKKEN have omongst each other a very large percentage of false
    # positives/negatives, so they seem very similar... lets create a class that
    # combines both
    # parceldata_df.loc[parceldata_df[
    #     column_output_class].isin(['MON_CG_BONEN', 'MON_CG_WIKKEN']),
    #     column_output_class
    # ] = 'MON_CG_BONEN_WIKKEN'

    # MON_BOOM includes now also the growing new plants/trees, which is too differenct
    # from grown trees -> put growing new trees is seperate group
    # parceldata_df.loc[parceldata_df[
    #     column_BEFL_cropcode].isin(['9602', '9603', '9604', '9560']),
    #     column_output_class
    # ] = 'MON_CG_BOOMKWEEK'

    # 'MON_CG_FRUIT': has a good accuracy (91%), but also has as much false positives
    #     (115% -> mainly 'MON_CG_GRASSEN' that are (mis)classified as 'MON_CG_FRUIT')
    # 'MON_CG_BOOM': has very bad accuracy (28%) and also very much false positives
    #     (450% -> mainly 'MON_CG_GRASSEN' that are misclassified as 'MON_CG_BOOM')
    # MON_FRUIT and MON_BOOM are permanent anyway, so not mandatory that they are
    # checked in monitoring process.
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE:DIFFICULT_PERMANENT_CLASS
    # parceldata_df.loc[parceldata_df[
    #     column_output_class].isin(['MON_CG_BOOM', 'MON_CG_FRUIT']),
    #     column_output_class
    # ] = 'IGNORE:DIFFICULT_PERMANENT_CLASS'

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    if not is_groundtruth and min_parcels_in_class > 1:
        for _, row in (
            parceldata_df.groupby(column_output_class)
            .size()
            .reset_index(name="count")
            .iterrows()
        ):
            if row["count"] < min_parcels_in_class:
                logger.info(
                    f"Class <{row[column_output_class]}> only contains {row['count']} "
                    "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
                )
                parceldata_df.loc[
                    parceldata_df[column_output_class] == row[column_output_class],
                    column_output_class,
                ] = "IGNORE:NOT_ENOUGH_SAMPLES"

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
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_croprotation(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
) -> pd.DataFrame:
    """Prepare input file for use in the croprotation marker.

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
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # --------------------------------------------------------------------------
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("string")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_BEFL_to_keep
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
        ] = "MON_VRU_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"]),
            column_output_class,
        ] = "MON_VRU_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "LOO", column_output_class
        ] = "MON_VRU_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "CON", column_output_class
        ] = "MON_VRU_CONTAINERS"  # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == "CIV", class_columnname
        # ] = "MON_VRU_CONTAINERS"  # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so this part of the code "
            "was skipped!"
        )

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    for _, row in (
        parceldata_df.groupby(column_output_class)
        .size()
        .reset_index(name="count")
        .iterrows()
    ):
        if row["count"] < conf.preprocess.getint("min_parcels_in_class"):
            logger.info(
                f"Class <{row[column_output_class]}> only contains {row['count']} "
                "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
            )
            parceldata_df.loc[
                parceldata_df[column_output_class] == row[column_output_class],
                column_output_class,
            ] = "IGNORE:NOT_ENOUGH_SAMPLES"

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
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_carbonsupply(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
) -> pd.DataFrame:
    """Prepare input file for use in the carbonsupply marker.

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
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # --------------------------------------------------------------------------
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("string")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_BEFL_to_keep
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
        ] = "MON_OC_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"]),
            column_output_class,
        ] = "MON_OC_OVERK_LOO"
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "LOO", column_output_class
        ] = "MON_OC_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm] == "CON", column_output_class
        ] = "MON_OC_CONTAINERS"  # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == "CIV", class_columnname
        # ] = "MON_OC_CONTAINERS"  # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so this part of the code "
            "was skipped!"
        )

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    for _, row in (
        parceldata_df.groupby(column_output_class)
        .size()
        .reset_index(name="count")
        .iterrows()
    ):
        if row["count"] < conf.preprocess.getint("min_parcels_in_class"):
            logger.info(
                f"Class <{row[column_output_class]}> only contains {row['count']} "
                "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
            )
            parceldata_df.loc[
                parceldata_df[column_output_class] == row[column_output_class],
                column_output_class,
            ] = "IGNORE:NOT_ENOUGH_SAMPLES"

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
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


@deprecated("fabacae is deprecated, add refe before using")
def prepare_input_fabaceae(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the fabaceae marker.

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
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # --------------------------------------------------------------------------
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("string")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_BEFL_to_keep
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
    # Conclusion: put MON_BOOM and MON_FRUIT to IGNORE:DIFFICULT_PERMANENT_CLASS
    # parceldata_df.loc[
    #     parceldata_df[column_output_class].isin(['MON_BOOM', 'MON_FRUIT']),
    #     column_output_class
    # ] = 'IGNORE:DIFFICULT_PERMANENT_CLASS'

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    if not is_groundtruth and min_parcels_in_class > 1:
        for _, row in (
            parceldata_df.groupby(column_output_class)
            .size()
            .reset_index(name="count")
            .iterrows()
        ):
            if row["count"] < min_parcels_in_class:
                logger.info(
                    f"Class <{row[column_output_class]}> only contains {row['count']} "
                    "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
                )
                parceldata_df.loc[
                    parceldata_df[column_output_class] == row[column_output_class],
                    column_output_class,
                ] = "IGNORE:NOT_ENOUGH_SAMPLES"

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
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_latecrop(
    parceldata_df: pd.DataFrame,
    column_BEFL_latecrop: str,
    column_BEFL_latecrop2: str | None,
    column_BEFL_maincrop: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
    scope: Literal["ALL", "EARLY_MAINCROP", "LATE_MAINCROP"],
) -> pd.DataFrame:
    """Prepare input file for use in the latecrop marker.

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
        raise ValueError(f"Input classes file doesn't exist: {classes_refe_path}")
    if scope is None or scope not in ["ALL", "EARLY_MAINCROP", "LATE_MAINCROP"]:
        raise ValueError(f"Invalid value for scope: {scope}")

    # Convert the crops to unicode, in case the input is int
    if column_BEFL_latecrop in parceldata_df.columns:
        parceldata_df[column_BEFL_latecrop] = parceldata_df[
            column_BEFL_latecrop
        ].astype("string")
    if column_BEFL_latecrop2 in parceldata_df.columns:
        parceldata_df[column_BEFL_latecrop2] = parceldata_df[
            column_BEFL_latecrop2
        ].astype("string")
    if column_BEFL_maincrop in parceldata_df.columns:
        parceldata_df[column_BEFL_maincrop] = parceldata_df[
            column_BEFL_maincrop
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # -------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_latecrop] = classes_df["CROPCODE"].astype("string")
    classes_df[column_BEFL_latecrop2] = classes_df["CROPCODE"].astype("string")
    classes_df[column_BEFL_maincrop] = classes_df["CROPCODE"].astype("string")

    # Map column with the classname to orig classname
    column_output_class_orig = conf.columns["class"] + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
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
                "MON_CG",
                "IS_PERM_BEDEKKING",
            ]
            and column not in columns_BEFL_to_keep
        ):
            classes_df.drop(column, axis=1, inplace=True)

    # Get only the columns in the classes_df that don't exist yet in parceldata_df
    cols_to_join = classes_df.columns.difference(parceldata_df.columns)

    # Join/merge the 1st late crop to the classname
    logger.info("Add the classes to the parceldata")
    # Set the index
    classes_latecrop_df = classes_df.set_index(
        column_BEFL_latecrop, verify_integrity=True
    )
    parceldata_df = parceldata_df.merge(
        classes_latecrop_df[cols_to_join],
        how="left",
        left_on=column_BEFL_latecrop,
        right_index=True,
        validate="many_to_one",
    )
    if column_BEFL_latecrop2 is not None:
        parceldata_df = parceldata_df.merge(
            classes_latecrop_df[cols_to_join],
            how="left",
            left_on=column_BEFL_latecrop2,
            right_index=True,
            validate="many_to_one",
            suffixes=(None, "_LATECROP2"),
        )

    # Set the index
    classes_maincrop_df = classes_df.set_index(
        column_BEFL_latecrop, verify_integrity=True
    )
    parceldata_df = parceldata_df.merge(
        classes_maincrop_df[cols_to_join],
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
    if column_BEFL_latecrop2 is not None:
        parceldata_df.insert(
            loc=0,
            column=conf.columns["class_declared2"],
            value=parceldata_df[f"{column_output_class_orig}_LATECROP2"],
        )

    # If limited scope, apply it
    if scope == "EARLY_MAINCROP":
        # Only process parcels with an early main crop.
        # Hence, set parcels with a main crop that stays late on the field to a class
        # that can be ignored later on.
        for class_name, cropcodes in late_main_crops.items():
            parceldata_df.loc[
                parceldata_df[column_BEFL_maincrop].isin(cropcodes), column_output_class
            ] = class_name
            if not is_groundtruth:
                parceldata_df.loc[
                    parceldata_df[column_BEFL_maincrop].isin(cropcodes),
                    conf.columns["class_declared"],
                ] = class_name

    elif scope == "LATE_MAINCROP":
        # Only process parcels with a late main crop.
        # Hence, set parcels with a main crop that is removed early to an IGNORE class.
        early_maincrop_classname = "IGNORE:EARLY_MAINCROP"

        early_maincrops = []
        for _, cropcodes in late_main_crops.items():
            early_maincrops.extend(cropcodes)

        parceldata_df.loc[
            ~parceldata_df[column_BEFL_maincrop].isin(early_maincrops),
            column_output_class,
        ] = early_maincrop_classname
        if not is_groundtruth:
            parceldata_df.loc[
                ~parceldata_df[column_BEFL_maincrop].isin(early_maincrops),
                conf.columns["class_declared"],
            ] = early_maincrop_classname

    elif scope == "ALL":
        # Process all parcels, so no action needed
        pass
    else:
        raise ValueError(f"Invalid value for scope: {scope}")

    # For rows with no class, set to UNKNOWN
    parceldata_df.fillna(value={column_output_class: "UNKNOWN"}, inplace=True)

    """
    # If permanent grassland, there will typically still be grass on the parcels
    # BUT: 60 is in ECO_430_GRASLAND_KLAVER_LUZ_GRAAN!!!
    parceldata_df.loc[
        (parceldata_df[column_output_class] == "UNKNOWN")
        & (parceldata_df[column_BEFL_status_perm_grass] is not None)
        & (parceldata_df["MON_CG_MAINCROP"] == "MON_CG_GRASSEN"),
        column_output_class,
    ] = "MON_CG_GRASSEN"
    """

    # Assumption: permanent crops except grassland won't/can't have a latecrop, so
    # don't bother predicting them
    parceldata_df.loc[
        (parceldata_df[column_output_class] == "UNKNOWN")
        # & (parceldata_df[column_BEFL_status_perm_grass] is None)
        & (parceldata_df["MON_CG_MAINCROP"] != "MON_CG_GRASSEN")
        & (parceldata_df["IS_PERM_BEDEKKING_MAINCROP"] == 1),
        column_output_class,
    ] = "IGNORE:PERMANENT_BEDEKKING"

    # If a column with extra info exists, use it as well to fine-tune the
    # classification classes.
    if column_BEFL_gesp_pm in parceldata_df.columns:
        # Serres, tijdelijke overkappingen en loodsen
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm].isin(["SER", "SGM"])),
            column_output_class,
        ] = "MON_CG_OVERK_LOO"
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm].isin(["PLA", "PLO", "NPO"])),
            column_output_class,
        ] = "MON_CG_OVERK_LOO"
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm] == "LOO"),
            column_output_class,
        ] = "MON_CG_OVERK_LOO"  # Een loods is hetzelfde als een stal...
        parceldata_df.loc[
            (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df[column_BEFL_gesp_pm] == "CON"),
            column_output_class,
        ] = "MON_CG_CONTAINERS"  # Containers, niet op volle grond...
        # TODO: CIV, containers in volle grond, lijkt niet zo specifiek te zijn...
        # parceldata_df.loc[
        #     parceldata_df[column_BEFL_gesp_pm] == 'CIV', class_columnname
        # ] = 'MON_CG_CONTAINERS'   # Containers, niet op volle grond...
    else:
        logger.warning(
            f"The column {column_BEFL_gesp_pm} doesn't exist, so couldn't be used!"
        )

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    if not is_groundtruth and min_parcels_in_class > 1:
        for _, row in (
            parceldata_df.groupby(column_output_class)
            .size()
            .reset_index(name="count")
            .iterrows()
        ):
            if row["count"] < min_parcels_in_class:
                logger.info(
                    f"Class <{row[column_output_class]}> only contains {row['count']} "
                    "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
                )
                parceldata_df.loc[
                    parceldata_df[column_output_class] == row[column_output_class],
                    column_output_class,
                ] = "IGNORE:NOT_ENOUGH_SAMPLES"

    # Add copy of class as class_declared
    if not is_groundtruth:
        parceldata_df[conf.columns["class_declared"]] = parceldata_df[
            column_output_class
        ]

    # Add a column with a correction factor to use when applying doubt thresholds.
    # Determine the correction factor based on the EOC score at the end of
    # class_declared.
    def get_eoc_score(class_declared: str) -> float:
        """Get the EOC score based on the declared class."""
        # If there is an EOC score, it is the last value in a "_" separated string
        class_declared_parts = class_declared.split("_")
        eoc_score_str = class_declared_parts[-1]
        try:
            eoc_score = float(eoc_score_str)
        except ValueError:
            eoc_score = 0.0

        return eoc_score

    if not is_groundtruth:
        parceldata_df["eoc_score"] = parceldata_df[
            conf.columns["class_declared"]
        ].apply(lambda x: get_eoc_score(x))
        eoc_score_max = parceldata_df["eoc_score"].max()
        if eoc_score_max == 0:
            raise ValueError(
                "Maximum EOC score is 0, cannot calculate correction factor"
            )
        parceldata_df["proba_correction_factor"] = (
            parceldata_df["eoc_score"] / eoc_score_max
        )

    """
    # Add IGNORE:FOR_TRAINING column: if 1, ignore for training
    parceldata_df["IGNORE:FOR_TRAINING"] = 0

    # If ndvi_latecrop_count data columns available, use them
    if ndvi_latecrop_count in parceldata_df.columns:
        # If no NDVI data avalable, not possible to determine bare soil
        # -> ignore parcel in training
        parceldata_df.loc[
            parceldata_df[ndvi_latecrop_count] < 10, "IGNORE:FOR_TRAINING"
        ] = 1
    else:
        logger.warning(
            f"No column {ndvi_latecrop_count} available to set IGNORE:FOR_TRAINING"
        )

    # TODO: MON_CG_BRAAK opkuisen met Timo?
    # TODO: controle obv ndvi > 0.3 => geen braak?
    #       braak van in de zomer => geen "braak" meer vanwege onkruid

    # If the median ndvi <= 0.3 parcel is still a bare field (for training)
    if ndvi_latecrop_median in parceldata_df.columns:
        # If also no grass, train as MON_CG_BRAAK
        parceldata_df.loc[
            (parceldata_df[ndvi_latecrop_median] <= 0.3)
            # & (parceldata_df[column_output_class] == "UNKNOWN")
            & (parceldata_df["MON_CG_MAINCROP"] != "MON_CG_GRASSEN"),
            column_output_class,
        ] = "MON_CG_BRAAK"

        # For all other classes, don't use for training if bare soil
        parceldata_df.loc[
            (parceldata_df[ndvi_latecrop_median] <= 0.3)
            & (parceldata_df[column_output_class] != "MON_BRAAK"),
            "IGNORE:FOR_TRAINING",
        ] = 1
    else:
        logger.warning(
            f"No column {ndvi_latecrop_count} available to set IGNORE:FOR_TRAINING"
        )
    """

    # Drop the columns that aren't useful at all
    for column in parceldata_df.columns:
        if (
            column
            not in [
                column_output_class,
                conf.columns["id"],
                conf.columns["class_groundtruth"],
                conf.columns["class_declared"],
                conf.columns["class_declared2"],
                ndvi_latecrop_count,
                ndvi_latecrop_median,
                "IGNORE:FOR_TRAINING",
                "proba_correction_factor",
            ]
            and column not in conf.preprocess.getlist("extra_export_columns")
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(columns=[column], inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_cropgroup_early(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the cropgroup_early marker."""
    # First run the standard landcover prepare
    parceldata_df = prepare_input_cropgroup(
        parceldata_df,
        column_BEFL_cropcode,
        column_output_class,
        classes_refe_path,
        min_parcels_in_class=min_parcels_in_class,
        is_groundtruth=is_groundtruth,
    )

    # Set late crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE:LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE:NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}!"
        )

    return parceldata_df


def prepare_input_croprotation_early(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
) -> pd.DataFrame:
    """Prepare input file for use in the croprotation_early marker."""
    # First run the standard landcover prepare
    parceldata_df = prepare_input_croprotation(
        parceldata_df, column_BEFL_cropcode, column_output_class, classes_refe_path
    )

    # Set late crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE:LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE:NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}!"
        )

    return parceldata_df


def prepare_input_carbonsupply_early(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
) -> pd.DataFrame:
    """Prepare input file for use in the carbonsupply_early marker."""
    # First run the standard landcover prepare
    parceldata_df = prepare_input_carbonsupply(
        parceldata_df, column_BEFL_cropcode, column_output_class, classes_refe_path
    )

    # Set late crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE:LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE:NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}!"
        )

    return parceldata_df


def prepare_input_landcover(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the landcover marker.

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
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # -------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("string")

    # Map column MON_group to orig classname
    column_output_class_orig = column_output_class + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_BEFL_to_keep
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
        ] = "MON_LC_CONTAINERS"
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
    ] = "MON_LC_ONBEKEND_MET_KLASSIFICATIE"

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    if not is_groundtruth and min_parcels_in_class > 1:
        for _, row in (
            parceldata_df.groupby(column_output_class)
            .size()
            .reset_index(name="count")
            .iterrows()
        ):
            if row["count"] < min_parcels_in_class:
                logger.info(
                    f"Class <{row[column_output_class]}> only contains {row['count']} "
                    "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
                )
                parceldata_df.loc[
                    parceldata_df[column_output_class] == row[column_output_class],
                    column_output_class,
                ] = "IGNORE:NOT_ENOUGH_SAMPLES"

        # Add copy of class as class_declared
        if conf.columns["class_declared"] not in parceldata_df.columns:
            parceldata_df[conf.columns["class_declared"]] = parceldata_df[
                column_output_class
            ]

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
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_landcover_early(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the landcover_early marker."""
    # First run the standard landcover prepare
    parceldata_df = prepare_input_landcover(
        parceldata_df,
        column_BEFL_cropcode,
        column_output_class,
        classes_refe_path,
        min_parcels_in_class=min_parcels_in_class,
        is_groundtruth=is_groundtruth,
    )

    # Set crops not in early crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE:LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE:NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}, so "
            "new grassland cannot be ignored!"
        )

    return parceldata_df


def prepare_input_ruggenteelt(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the ruggenteelt marker.

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
        ].astype("string")

    # Read and cleanup the mapping table from crop codes to classes
    # -------------------------------------------------------------
    logger.info(f"Read classes conversion table from {classes_refe_path}")
    classes_df = pdh.read_file(classes_refe_path)

    # Because the file was read as ansi, and gewas is int, so the data needs to be
    # converted to unicode to be able to do comparisons with the other data
    classes_df[column_BEFL_cropcode] = classes_df["CROPCODE"].astype("string")

    # Map column MON_group to orig classname
    column_output_class_orig = column_output_class + "_orig"
    classes_df[column_output_class_orig] = classes_df[conf.columns["class_refe"]]

    # Remove unneeded columns
    for column in classes_df.columns:
        if (
            column not in [column_output_class_orig, column_BEFL_cropcode]
            and column not in columns_BEFL_to_keep
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
        ] = "MON_RUG_OVERK_LOO"
        # Containers
        parceldata_df.loc[
            parceldata_df[column_BEFL_gesp_pm].isin(["CON"]), column_output_class
        ] = "MON_RUG_CONTAINERS"
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
    ] = "MON_RUG_ONBEKEND_MET_KLASSIFICATIE"

    # Set classes with very few elements to IGNORE:NOT_ENOUGH_SAMPLES!
    if not is_groundtruth and min_parcels_in_class > 1:
        for _, row in (
            parceldata_df.groupby(column_output_class)
            .size()
            .reset_index(name="count")
            .iterrows()
        ):
            if row["count"] < min_parcels_in_class:
                logger.info(
                    f"Class <{row[column_output_class]}> only contains {row['count']} "
                    "elements, so put them to IGNORE:NOT_ENOUGH_SAMPLES"
                )
                parceldata_df.loc[
                    parceldata_df[column_output_class] == row[column_output_class],
                    column_output_class,
                ] = "IGNORE:NOT_ENOUGH_SAMPLES"

        # Add copy of class as class_declared
        if conf.columns["class_declared"] not in parceldata_df.columns:
            parceldata_df[conf.columns["class_declared"]] = parceldata_df[
                column_output_class
            ]

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
            and column not in columns_BEFL_to_keep
        ):
            parceldata_df.drop(column, axis=1, inplace=True)
        elif column == column_BEFL_gesp_pm:
            parceldata_df[column_BEFL_gesp_pm] = parceldata_df[
                column_BEFL_gesp_pm
            ].str.replace(",", ";")

    return parceldata_df


def prepare_input_ruggenteelt_early(
    parceldata_df: pd.DataFrame,
    column_BEFL_cropcode: str,
    column_output_class: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    is_groundtruth: bool,
) -> pd.DataFrame:
    """Prepare input file for use in the ruggenteelt_early marker."""
    # First run the standard ruggenteelt prepare
    parceldata_df = prepare_input_ruggenteelt(
        parceldata_df,
        column_BEFL_cropcode,
        column_output_class,
        classes_refe_path,
        min_parcels_in_class=min_parcels_in_class,
        is_groundtruth=is_groundtruth,
    )

    # Set crops not in early crops to ignore
    parceldata_df.loc[
        parceldata_df[column_BEFL_earlylate] != "MON_TEELTEN_VROEGE",
        column_output_class,
    ] = "IGNORE:LATE_CROP"

    # Set new grass to ignore
    if column_BEFL_status_perm_grass in parceldata_df.columns:
        parceldata_df.loc[
            (parceldata_df[column_BEFL_cropcode] == "60")
            & (
                (parceldata_df[column_BEFL_status_perm_grass] == "BG1")
                | (parceldata_df[column_BEFL_status_perm_grass].isnull())
            ),
            column_output_class,
        ] = "IGNORE:NEW_GRASSLAND"
    else:
        logger.warning(
            f"Source file doesn't contain column {column_BEFL_status_perm_grass}, so "
            "new grassland cannot be ignored!"
        )

    return parceldata_df
