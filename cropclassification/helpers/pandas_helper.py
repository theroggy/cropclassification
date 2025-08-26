"""Module with helper functions to expand on some features of pandas."""

import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import pyogrio


def get_table_info(path: Path, table_name: str = "info") -> dict[str, Any]:
    """Gte information about a table in a database file.

    Args:
        path (Path): path to the database file.
        table_name (str, optional): name of the table to get info from.
            Defaults to "info".

    Returns:
        dict[str, Any]: information about the table.
    """
    ext_lower = path.suffix.lower()
    if ext_lower in (".sqlite", ".gpkg"):
        sql_db = None
        try:
            sql_db = sqlite3.connect(str(path))
            df = pd.read_sql_query(
                f'select count(*) featurecount from "{table_name}"', sql_db
            )
            featurecount = df["featurecount"].iloc[0].item()

            df = pd.read_sql_query(f"PRAGMA table_info('{table_name}')", sql_db)
            columns = df["name"].to_list()

            result = {
                "featurecount": featurecount,
                "columns": columns,
            }
            return result

        except Exception as ex:
            raise RuntimeError(f"Error reading data from {path!s}") from ex
        finally:
            if sql_db is not None:
                sql_db.close()
    else:
        raise ValueError(f"Not implemented for extension {ext_lower}")


def read_file(
    path: Path,
    table_name: str = "info",
    columns: Optional[list[str]] = None,
    sql: Optional[str] = None,
    ignore_geometry: bool = True,
) -> pd.DataFrame:
    """Read a file to a pandas dataframe.

    The fileformat is detected based on the path extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass
    # them to next function, example encoding, float_format,...
    """
    if columns is not None and not isinstance(columns, list):
        raise Exception(f"Parameter columns should be list, but is {type(columns)}")

    ext_lower = path.suffix.lower()
    if sql is not None and ext_lower not in (".sqlite", ".gpkg"):
        raise ValueError("sql parameter is only supported for sqlite and gpkg files")

    if ext_lower == ".csv":
        try:
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                low_memory=False,
            )
        except UnicodeDecodeError:
            # If a unicode decode error is thrown, try again using CP1252 encoding
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                low_memory=False,
                encoding="cp1252",
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
            # If a unicode decode error is thrown, try again using CP1252 encoding
            data_read_df = pd.read_csv(
                str(path),
                usecols=columns,
                sep="\t",
                low_memory=False,
                encoding="cp1252",
            )
        return data_read_df
    elif ext_lower == ".parquet":
        return pd.read_parquet(str(path), columns=columns)
    elif ext_lower in (".sqlite", ".gpkg"):
        return pyogrio.read_dataframe(
            path,
            columns=columns,
            read_geometry=not ignore_geometry,
            sql=sql,
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Not implemented for extension {ext_lower}")


def to_file(
    df: Optional[Union[pd.DataFrame, pd.Series]],
    path: Path,
    table_name: str = "info",
    index: bool = True,
    append: bool = False,
):
    """Writes a pandas dataframe to file.

    The file format is detected based on the path extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass
    # them to next function, example encoding, float_format,...
    """
    if df is None:
        raise ValueError("Dataframe df is None")
    if not append and path.exists():
        raise ValueError(f"path already exists and append is False: {path}")

    ext_lower = path.suffix.lower()
    if ext_lower == ".csv":
        if append:
            raise ValueError("Append is not supported for csv files")
        df.to_csv(str(path), float_format="%.10f", encoding="utf-8", index=index)
    elif ext_lower == ".tsv":
        if append:
            raise ValueError("Append is not supported for tsv files")
        df.to_csv(
            str(path), sep="\t", float_format="%.10f", encoding="utf-8", index=index
        )
    elif ext_lower == ".parquet":
        if append:
            raise ValueError("Append is not supported for parquet files")
        df.to_parquet(str(path), index=index)
    elif ext_lower == ".sqlite":
        if_exists = "fail"
        if append:
            if_exists = "append"
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
            path_length = len(str(path))
            if path_length > 250:
                message = f"Error in to_file (note: {path_length=}) with {path=!s}"
            else:
                message = f"Error in to_file with {path=!s}"
            raise RuntimeError(message) from ex
        finally:
            if sql_db is not None:
                sql_db.close()
    else:
        raise ValueError(f"Not implemented for extension {ext_lower}")


def to_excel(
    stats_df: pd.DataFrame, path: Path, sheet_name: str = "data", index: bool = True
) -> None:
    """Write dataframe to excel.

    Args:
        stats_df (pd.DataFrame): _description_
        path (Path): _description_
        sheet_name (str, optional): _description_. Defaults to "data".
        index (bool, optional): _description_. Defaults to True.
    """
    # Write to Excel
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        stats_df.to_excel(writer, sheet_name=sheet_name, index=index)
        columns_format = _get_columns_for_formatting(stats_df, index=index)
        _apply_formatting(writer, sheet_name, columns_format)


def _get_columns_for_formatting(df, index: bool) -> dict:
    def get_format(col: str, dtype) -> Optional[dict]:
        if pd.api.types.is_integer_dtype(dtype):
            return {"num_format": "0"}
        elif pd.api.types.is_numeric_dtype(dtype):
            if col.startswith(("pct_", "percentage")) or col.endswith("_pct"):
                return {"num_format": "0.00%"}
            return {"num_format": "0.00"}
        else:
            return None

    # First we find the maximum length of the index column if needed
    result: dict = {}
    column_idx_offset = 0
    if index:
        column_idx_offset = 1
        index_name = str(df.index.name)
        result[index_name] = {}
        result[index_name]["index"] = 0
        # Count the column title as one character extra as it is bold
        result[index_name]["width"] = max(
            [len(str(s)) for s in df.index.values] + [len(index_name) + 1]
        )
        result[index_name]["format"] = get_format(index_name, df.index.dtype)

    # Now all other columns
    for column_idx, col in enumerate(df.columns):
        result[col] = {}
        result[col]["index"] = column_idx + column_idx_offset
        # Count the column title as one character extra as it is bold
        result[col]["width"] = max(
            [len(str(s)) for s in df[col].values] + [len(col) + 1]
        )
        result[col]["format"] = get_format(col, df[col].dtype)

    return result


def _apply_formatting(writer, sheet_name: str, format_info: dict):
    # Apply format info
    for column in format_info:
        format = None
        if format_info[column]["format"] is not None:
            format = writer.book.add_format(format_info[column]["format"])
        writer.sheets[sheet_name].set_column(
            first_col=format_info[column]["index"],
            last_col=format_info[column]["index"],
            width=format_info[column]["width"],
            cell_format=format,
        )
