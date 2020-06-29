# -----Example Python Program to alter an SQLite Table-----

import glob
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from cropclassification.helpers import pandas_helper as pdh

def main():

    dir = r"X:\Monitoring\Markers\playground\pierog\tmp\Run_2019-06-25_007_imported"
    glob_search = os.path.join(dir, "*.parquet")

    in_filepaths = glob.glob(glob_search)

    # Convert all files found
    for in_filepath in in_filepaths:

        # Read input file
        print(f"Read {in_filepath}")
        df = pdh.read_file(in_filepath)

        # Write to new file
        in_filepath_noext, _ = os.path.splitext(in_filepath)
        out_filepath = in_filepath_noext + '.sqlite'
        print(f"Write {out_filepath}")
        pdh.to_file(df, out_filepath)

# If the script is run directly...
if __name__ == "__main__":
    main()
