# -----Example Python Program to alter an SQLite Table-----

import glob
import os

import sqlite3

def rename_table(sqlite_filepath: str,
                 tablename_src: str,
                 tablename_dest: str):

    # Create a connection object
    connection  = sqlite3.connect(sqlite_filepath)

    # Get a cursor
    cursor = connection.cursor()

    # Rename the SQLite Table
    rename_table = f'ALTER TABLE "{tablename_src}" RENAME TO {tablename_dest}'
    try:
        print(f"Execute {rename_table} on {os.path.basename(sqlite_filepath)}")
        cursor.execute(rename_table)  
    except Exception as ex:
        print(f"Exception: {ex}")

    # close the database connection
    connection.close()

def main():

    #dir = r"X:\Monitoring\Markers\playground\_timeseries_per_image\Prc_BEFL_2019_2019-06-25_bufm5"
    #dir = r"X:\Monitoring\Markers\playground\_timeseries_per_image\Prc_BEFL_2018_2019-06-14_bufm5"
    dir = r"X:\Monitoring\Markers\playground\_timeseries_per_image\Prc_BEFL_2018_2019-07-02_bufm5"
    #dir = r"/home/pierog/playground/_timeseries_per_image/Prc_BEFL_2019_2019-06-25_bufm5"
    #dir = r"/home/pierog/playground/_timeseries_per_image/Prc_BEFL_2018_2019-06-14_bufm5"
    glob_search = os.path.join(dir, "*.sqlite")

    sqlite_files = glob.glob(glob_search)

    for filepath in sqlite_files:
        rename_table(filepath, 'default', 'info') 

        if filepath.endswith('_10m.sqlite'):
            new_filepath = filepath.replace('_10m.sqlite', '-10m.sqlite')
            os.rename(filepath, new_filepath)
        elif filepath.endswith('_20m.sqlite'):
            new_filepath = filepath.replace('_20m.sqlite', '-20m.sqlite')
            os.rename(filepath, new_filepath)

# If the script is run directly...
if __name__ == "__main__":
    main()
