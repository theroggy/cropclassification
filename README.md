# Crop classification
This is a collection of scripts that can help to classify crops using Sentinel data. 

Probably this documentation won't suffice to get you started, but you are free to reach out for more info.

## Installation manual
1. Install conda

As the scripts are written in Python, you need to use a package manager to be able to install the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

If you need some more installation instructions, have a look here:
https://conda.io/docs/user-guide/install/index.html

2. Create new environment and install dependencies

Once you have anaconda installed, you can open an anaconda terminal window and follow the following steps:

      1. Create and activate a new conda environment
      ```
      conda create --name cropclassification python=3.6
      conda activate cropclassification
      ```
      2. Install the dependencies for the crop classification scripts:
      ```
      conda install scikit-learn keras tensorflow rasterio rasterstats geopandas pyarrow psutil
      ```
      3. If it was the first time you installed anaconda/geopandas, you might have to restat your computer to proceed
      4. Start the anaconda terminal window again and activate the environment
      ```
      conda activate cropclassification
      ```
      5. Now install cropclassification with pip
      ```
      pip install cropclassification
      ```
4. Calculate time series 

To calculate time series, you need to run `cropclassification -t <tasks_dir>`, with a 'calc_timeseries' type of task in the tasks dir 
on a server that has access to sentinel CARD images.

5. Start a crop classification

Run `cropclassification -t <tasks_dir>`, with a 'calc_marker' type of task in the tasks dir.

## Sample data

Sample data can be downloaded from the following location. If you don't change the default paths in the scripts in needs to be put in the directory C:\temp\CropClassification\InputData
https://drive.google.com/open?id=1eN9cBcWyvM0msNMCD6nivcGuZfYyqV5q
