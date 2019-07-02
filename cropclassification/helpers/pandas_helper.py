# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of pandas.
"""

import os

import pandas as pd
import sqlite3

def read_file(filepath: str,
              table_name: str = 'info',
              columns: [] = None) -> pd.DataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    if columns is not None and type(columns) is not list:
        raise Exception(f"Parameter columns should be list, but is {type(columns)}")
    _, ext = os.path.splitext(filepath)

    ext_lower = ext.lower()
    if ext_lower == '.csv':
        try:
            data_read_df = pd.read_csv(filepath, usecols=columns, low_memory=False)
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using ANSI encoding
            data_read_df = pd.read_csv(filepath, usecols=columns, low_memory=False, encoding='ANSI')
        return data_read_df
    elif ext_lower == '.tsv':
        try:
            data_read_df = pd.read_csv(filepath, usecols=columns, sep='\t', low_memory=False)
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using ANSI encoding
            data_read_df = pd.read_csv(filepath, usecols=columns, sep='\t', low_memory=False, encoding='ANSI')
        return data_read_df
    elif ext_lower == '.parquet':
        return pd.read_parquet(filepath, columns=columns)
    elif ext_lower == '.sqlite':
        sql_db = sqlite3.connect(filepath)
        if columns is None:
            cols_to_select = '*'
        else:
            cols_to_select = ', '.join(columns)
        try:
            data_read_df = pd.read_sql_query(f"select {cols_to_select} from {table_name}", sql_db)
        except Exception as ex:
            raise Exception(f"Error reading data from {filepath}") from ex
        finally:
            sql_db.close()
        return data_read_df
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(df: pd.DataFrame,
            filepath: str,
            table_name: str = 'info',
            index: bool = True,
            append: bool = False):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(filepath)

    ext_lower = ext.lower()
    if ext_lower == '.csv':
        if append:
            raise Exception("Append is not supported for csv files")
        df.to_csv(filepath, float_format='%.10f', encoding='utf-8', index=index)
    elif ext_lower == '.tsv':
        if append:
            raise Exception("Append is not supported for tsv files")
        df.to_csv(filepath, sep='\t', float_format='%.10f', encoding='utf-8', index=index)
    elif ext_lower == '.parquet':
        if append:
            raise Exception("Append is not supported for parquet files")
        df.to_parquet(filepath, index=index)
    elif ext_lower == '.sqlite':
        if_exists = 'fail'
        if append:
            if_exists = 'append'
        elif os.path.exists(filepath):
            os.remove(filepath)

        sql_db = sqlite3.connect(filepath)
        df.to_sql(name=table_name, con=sql_db, if_exists=if_exists, index=index, 
                  chunksize=50000)
        sql_db.close()
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
