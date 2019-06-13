# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of pandas.
"""

import os

import pandas as pd

def read_file(filepath: str,
              columns: [] = None) -> pd.DataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(filepath)

    ext_lower = ext.lower()
    if ext_lower == '.csv':
        try:
            data_read_df = pd.read_csv(filepath, low_memory=False)
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using ANSI encoding
            data_read_df = pd.read_csv(filepath, low_memory=False, encoding='ANSI')
        return data_read_df
    elif ext_lower == '.tsv':
        try:
            data_read_df = pd.read_csv(filepath, sep='\t', low_memory=False)
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using ANSI encoding
            data_read_df = pd.read_csv(filepath, sep='\t', low_memory=False, encoding='ANSI')
        return data_read_df
    elif ext_lower == '.parquet':
        return pd.read_parquet(filepath, columns=columns)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(df: pd.DataFrame,
            filepath: str,
            index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    _, ext = os.path.splitext(filepath)

    ext_lower = ext.lower()
    if ext_lower == '.csv':
        df.to_csv(filepath, float_format='%.10f', encoding='utf-8', index=index)
    elif ext_lower == '.tsv':
        df.to_csv(filepath, sep='\t', float_format='%.10f', encoding='utf-8', index=index)
    elif ext_lower == '.parquet':
        df.to_parquet(filepath, index=index)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
