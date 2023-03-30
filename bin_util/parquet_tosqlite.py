# -----Example Python Program to alter an SQLite Table-----

from pathlib import Path

from cropclassification.helpers import pandas_helper as pdh


def main():
    dir = Path(
        "X:/Monitoring/Markers/playground/pierog/tmp/Run_2019-06-25_007_imported"
    )
    in_paths = dir.glob("*.parquet")

    # Convert all files found
    for in_path in in_paths:
        # Read input file
        print(f"Read {in_path}")
        df = pdh.read_file(in_path)

        # Write to new file
        out_path = in_path.parent / f"{in_path.stem}.sqlite"
        print(f"Write {out_path}")
        pdh.to_file(df, out_path)


# If the script is run directly...
if __name__ == "__main__":
    main()
