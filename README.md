# Crop classification
This is a collection of scripts that can help to classify crops using Sentinel data.

## Installation manual
1. Install conda
As the scripts are written in Python, you need to use a package manager to be able to install the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

If you need some more installation instructions, have a look here:
https://conda.io/docs/user-guide/install/index.html

1. Create new environment and install dependencies
Once you have anaconda installed, you can open an anaconda terminal window and follow the following steps:

      1. Create and activate a new conda environment
      ```
      conda create --name cropclassification python=3.6
      conda activate cropclassification
      ```
      2. Install the dependencies for the crop classification scripts:
      ```
      conda install --channel conda-forge earthengine-api google-api-python-client scikit-learn keras tensorflow rasterio rasterstats geopandas pyarrow psutil
      ```
      3. Possibly you need to restart your computer now, especially if it was the first time you installed anaconda/geopandas
      4. Start the anaconda terminal window again and activate the environment
      ```
      conda activate cropclassification
      ```
1. Update the configuration to your situation
You can now open the scripts in eg. visual studio code, and check out especially the .ini files in the config dir. You need to update those to match your environment, eg. set the paths as you want them,...

1. Run `calc_dias.py` to calculate the time series on a server that has access to the sentinel images

1. Run `run_job.py` to start a crop classification...

## Sample data

Sample data can be downloaded from the following location. If you don't change the default paths in the scripts in needs to be put in the directory C:\temp\CropClassification\InputData
https://drive.google.com/open?id=1eN9cBcWyvM0msNMCD6nivcGuZfYyqV5q
