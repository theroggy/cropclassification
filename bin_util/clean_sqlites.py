# -----Example Python Program to alter an SQLite Table-----

import glob
import os

import sqlite3


def rename_table(sqlite_path: str, tablename_src: str, tablename_dest: str):

    # Create a connection object
    connection = sqlite3.connect(sqlite_path)

    # Get a cursor
    cursor = connection.cursor()

    # Rename the SQLite Table
    rename_table = f'ALTER TABLE "{tablename_src}" RENAME TO {tablename_dest}'
    try:
        print(f"Execute {rename_table} on {os.path.basename(sqlite_path)}")
        cursor.execute(rename_table)
    except Exception as ex:
        print(f"Exception: {ex}")

    # close the database connection
    connection.close()


def main():

    # dir = r"X:\Monitoring\Markers\playground\_timeseries_per_image\Prc_BEFL_2019_2019-06-25_bufm5"
    # dir = r"X:\Monitoring\Markers\playground\_timeseries_per_image\Prc_BEFL_2018_2019-06-14_bufm5"
    dir = r"X:\Monitoring\Markers\playground\_timeseries_per_image\Prc_BEFL_2018_2019-07-02_bufm5"
    # dir = r"/home/pierog/playground/_timeseries_per_image/Prc_BEFL_2019_2019-06-25_bufm5"
    # dir = r"/home/pierog/playground/_timeseries_per_image/Prc_BEFL_2018_2019-06-14_bufm5"
    glob_search = os.path.join(dir, "*.sqlite")

    sqlite_files = glob.glob(glob_search)

    for path in sqlite_files:
        rename_table(path, "default", "info")

        if path.endswith("_10m.sqlite"):
            new_path = path.replace("_10m.sqlite", "-10m.sqlite")
            os.rename(path, new_path)
        elif path.endswith("_20m.sqlite"):
            new_path = path.replace("_20m.sqlite", "-20m.sqlite")
            os.rename(path, new_path)


# If the script is run directly...
if __name__ == "__main__":
    main()
