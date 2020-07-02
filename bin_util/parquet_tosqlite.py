# -----Example Python Program to alter an SQLite Table-----

import glob
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from cropclassification.helpers import pandas_helper as pdh

def main():

    dir = Path("X:/Monitoring/Markers/playground/pierog/tmp/Run_2019-06-25_007_imported")
    in_filepaths = dir.glob("*.parquet")

    # Convert all files found
    for in_filepath in in_filepaths:

        # Read input file
        print(f"Read {in_filepath}")
        df = pdh.read_file(in_filepath)

        # Write to new file
        out_filepath = in_filepath.parent / f"{in_filepath.stem}.sqlite"
        print(f"Write {out_filepath}")
        pdh.to_file(df, out_filepath)

# If the script is run directly...
if __name__ == "__main__":
    main()
