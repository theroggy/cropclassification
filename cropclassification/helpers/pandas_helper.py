# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of pandas.
"""

from pathlib import Path
import os
from typing import List, Optional, Union

import pandas as pd
import sqlite3


def read_file(
    path: Path, table_name: str = "info", columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the
    path extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass
    # them to next function, example encoding, float_format,...
    """
    if columns is not None and not isinstance(columns, list):
        raise Exception(f"Parameter columns should be list, but is {type(columns)}")

    ext_lower = path.suffix.lower()
    if ext_lower == ".csv":
        try:
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                low_memory=False,
            )
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using ANSI encoding
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                low_memory=False,
                encoding="ANSI",
            )
        return data_read_df
    elif ext_lower == ".tsv":
        try:
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                sep="\t",
                low_memory=False,
            )
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using ANSI encoding
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                sep="\t",
                low_memory=False,
                encoding="ANSI",
            )
        return data_read_df
    elif ext_lower == ".parquet":
        return pd.read_parquet(str(path), columns=columns)
    elif ext_lower in (".sqlite", ".gpkg"):
        sql_db = None
        try:
            sql_db = sqlite3.connect(str(path))
            if columns is None:
                cols_to_select = "*"
            else:
                cols_to_select = ", ".join(columns)
            data_read_df = pd.read_sql_query(
                f'select {cols_to_select} from "{table_name}"', sql_db
            )
        except Exception as ex:
            raise Exception(f"Error reading data from {str(path)}") from ex
        finally:
            if sql_db is not None:
                sql_db.close()
        return data_read_df
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")


def to_file(
    df: Union[pd.DataFrame, pd.Series],
    path: Path,
    table_name: str = "info",
    index: bool = True,
    append: bool = False,
):
    """
    Reads a pandas dataframe to file. The file format is detected based on the path
    extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass
    # them to next function, example encoding, float_format,...
    """
    ext_lower = path.suffix.lower()
    if ext_lower == ".csv":
        if append:
            raise Exception("Append is not supported for csv files")
        df.to_csv(str(path), float_format="%.10f", encoding="utf-8", index=index)
    elif ext_lower == ".tsv":
        if append:
            raise Exception("Append is not supported for tsv files")
        df.to_csv(
            str(path), sep="\t", float_format="%.10f", encoding="utf-8", index=index
        )
    elif ext_lower == ".parquet":
        if append:
            raise Exception("Append is not supported for parquet files")
        df.to_parquet(str(path), index=index)
    elif ext_lower == ".sqlite":
        if_exists = "fail"
        if append:
            if_exists = "append"
        elif os.path.exists(path):
            os.remove(path)
        sql_db = None
        try:
            sql_db = sqlite3.connect(str(path))
            df.to_sql(
                name=table_name,
                con=sql_db,
                if_exists=if_exists,  # type: ignore[arg-type]
                index=index,
                chunksize=50000,
            )
        except Exception as ex:
            raise Exception(f"Error in to_file to file {str(path)}") from ex
        finally:
            if sql_db is not None:
                sql_db.close()
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
