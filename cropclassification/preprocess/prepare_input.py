"""
Module with helper functions to preprocess the data to use for the classification.
"""

import logging
import os
import shutil
from pathlib import Path

import geofileops as gfo

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh
import cropclassification.preprocess._prepare_input_BEFL as befl

# Get a logger...
logger = logging.getLogger(__name__)


def prepare_input(
    input_parcel_path: Path,
    input_parcel_filetype: str,
    timeseries_periodic_dir: Path,
    base_filename: str,
    data_ext: str,
    classtype_to_prepare: str,
    classes_refe_path: Path,
    min_parcels_in_class: int,
    output_parcel_path: Path,
    force: bool = False,
):
    """
    Prepare a raw input file by eg. adding the classification classes to use for the
    classification,...
    """

    # If force == False Check and the output file exists already, stop.
    if not force and output_parcel_path.exists():
        logger.warning(
            f"prepare_input: output file exists + force is False: {output_parcel_path}"
        )
        return

    # If it exists, copy the refe file to the run dir, so we always keep knowing which
    # refe was used
    if classes_refe_path is not None:
        shutil.copy(classes_refe_path, output_parcel_path.parent)

    if input_parcel_filetype == "BEFL":
        # classes_refe_path must exist for BEFL!
        if not classes_refe_path.exists():
            raise Exception(f"Input classes file doesn't exist: {classes_refe_path}")

        df_parceldata = befl.prepare_input(
            input_parcel_path=input_parcel_path,
            classtype_to_prepare=classtype_to_prepare,
            classes_refe_path=classes_refe_path,
            min_parcels_in_class=min_parcels_in_class,
            output_dir=output_parcel_path.parent,
        )
    else:
        message = f"Unknown value for input_parcel_filetype: {input_parcel_filetype}"
        logger.critical(message)
        raise Exception(message)

    # Load pixcount data and join it
    parcel_pixcount_path = (
        timeseries_periodic_dir / f"{base_filename}_pixcount{data_ext}"
    )
    if parcel_pixcount_path.exists():
        logger.info(f"Read pixcount file {parcel_pixcount_path}")
        df_pixcount = pdh.read_file(parcel_pixcount_path)
        logger.debug(f"Read pixcount file ready, shape: {df_pixcount.shape}")
    else:
        # Find the s1 image having the largest percentage of pixcount >= 1, and use that
        logger.info("Determine pixel count based on s1 timeseries data")
        pixcount_path = None
        nb_pixcount_ok_max = -1
        sql_stmt = """
            SELECT COUNT(*) AS aantal_pixcount_gte1 FROM info WHERE info.count >= 1
        """
        for path in timeseries_periodic_dir.glob("*s1*.sqlite"):
            nb_pixcount_ok = gfo.read_file(path, sql_stmt=sql_stmt)[
                "aantal_pixcount_gte1"
            ][0].item()
            if nb_pixcount_ok > nb_pixcount_ok_max:
                nb_pixcount_ok_max = nb_pixcount_ok
                pixcount_path = path
        if pixcount_path is None:
            raise RuntimeError(
                f"No valid timeseries found to get pixel count for {input_parcel_path}"
            )

        df_pixcount = pdh.read_file(pixcount_path)
        df_pixcount = df_pixcount[[conf.columns["id"], "count"]].rename(
            columns={"count": conf.columns["pixcount_s1s2"]}
        )
        pdh.to_file(df_pixcount, parcel_pixcount_path)

    if df_pixcount.index.name != conf.columns["id"]:
        df_pixcount.set_index(conf.columns["id"], inplace=True)

    df_parceldata.set_index(conf.columns["id"], inplace=True)
    df_parceldata = df_parceldata.join(
        df_pixcount[conf.columns["pixcount_s1s2"]], how="left"
    )
    df_parceldata.fillna({conf.columns["pixcount_s1s2"]: 0}, inplace=True)

    # Export result to file
    output_ext = os.path.splitext(output_parcel_path)[1]
    for column in df_parceldata.columns:
        # if the output asked is a csv... we don't need the geometry...
        if column == conf.columns["geom"] and output_ext == ".csv":
            df_parceldata.drop(column, axis=1, inplace=True)

    logger.info(f"Write output to {output_parcel_path}")
    # If extension is not .shp, write using pandas (=a lot faster!)
    if output_ext.lower() != ".shp":
        pdh.to_file(df_parceldata, output_parcel_path)
    else:
        df_parceldata.to_file(output_parcel_path, index=False)
