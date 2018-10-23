# cropclassification
This is a collection of scripts that can help to classify crops using Sentinel data.
test
## Installation manual

1. As google earth engine is used to get the Sentinel data for the classification, you will need a
google account that has google earth engine access enabled.
So go to https://earthengine.google.com/ and click "Sign up" at the top right if you haven't used
google earth engine yet.
Once you are signed up you should be able to visit https://code.earthengine.google.com/.

2. As the scripts are written in Python, you need to use a package manager to be able to install
the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and
python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

If you need some more installation instructions, have a look here:
https://conda.io/docs/user-guide/install/index.html

3. Once you have anaconda installed, you can open an anaconda terminal window and follow the
following steps:

      1. Create and activate a new conda environment
      ```
	  conda create --name cropclassification python=3.6
      conda activate cropclassification
	  ```
      2. Install the dependencies for the crop classification scripts:
      ```
	  conda install -c conda-forge earthengine-api
	  conda install -c conda-forge google-api-python-client
	  conda install scikit-learn
	  conda install geopandas
	  ```
      3. Reboot your computer
      4. Start the anaconda terminal window again and activate the environment
      ```
      conda activate cropclassification
      ```
      5. I use spyder to edit my scripts. After installing geopandas Spyder doesn't seem to work
      anymore, but installing it in the new environment explicitly and updating all
      packages in this environment solves this.
      ```
	  conda install spyder
      conda update -–all
      ```
4.  Run the following command to authenticate with google earth engine, and follow the procedure on the screen:
```
earthengine authenticate
```

5. Activate (read-only) access to your google drive
    1. Go to the Google APIs cloud resource manager: https://console.developers.google.com/cloud-resource-manager
    2. If the UI isn't in english, you might want to set it to english so following the next steps
    is easier. You can do this in the 'Three dots menu', then 'Preferences'.
    3. Create a project, eg. 'download-gee-data'.
    4. Search for 'Google Drive API', select the entry, and click 'Enable'.
    5. Select 'Credentials' from the left menu, click 'Create Credentials', select 'OAuth client ID'.
        1. Now, the product name and consent screen need to be set -> click 'Configure consent screen'
        2. Enter a name for the consent screen, eg. download-gee-data.
        3. Click 'Save'
        4. Select 'Application type' to be 'Other'.
        5. Enter a name, eg. download-gee-data.
        6. Click 'Create'
        7. Click 'Download JSON' on the right side of Client ID to download client_secret_<really long ID>.json.
    6. Copy 'client_secret_<really long ID>.json' to your python working directory and rename to
    client_secret_download-gee-data.json

6. You can now open the scripts in eg. spyder, and check out especially `marker_cropgroup.py` and
`global_settings.py`. You need to update those to match your environment, eg. set the paths as you
want them,...

7. Now run `marker_cropgroup.py` to start a crop classification...

## Sample data

Sample data can be downloaded from the following location. If you don't change the default paths in the scripts in needs to be put in the directory C:\temp\CropClassification\InputData
https://drive.google.com/open?id=1eN9cBcWyvM0msNMCD6nivcGuZfYyqV5q
