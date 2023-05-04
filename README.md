# Crop classification
This is a collection of scripts that can help to classify crops using Sentinel data. 

Probably this documentation won't suffice to get you started, but you are free to reach out for more info if you are really interested.

In general, you should also be aware that the package isn't really meant to be an easy-to-use solution for crop classification. It is rather a quite specialised solution for our specific needs. Nonetheless we want to make the code public so if anyone is interested, they can have a look and possibly get some inspiration.

## Installation manual
1. Install conda

As the scripts are written in Python, you need to use a package manager to be able to install the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

2. Create new environment and install dependencies

Once you have anaconda installed, you can open an anaconda terminal window and follow the following steps:

      1. Create and activate a new conda environment
      ```
      conda create --name cropclassification
      conda activate cropclassification
      conda config --env --add channels conda-forge
      conda config --env --set channel_priority strict
      ```
      2. Install the dependencies for the crop classification scripts:
      ```
      conda install python=3.9 geopandas geofileops "h5py<3" openeo psutil qgis rasterio rasterstats scikit-learn
      ```
      3. If it was the first time you installed anaconda/geopandas, you might have to restart your computer to proceed
      4. Start the anaconda terminal window again and activate the environment
      ```
      conda activate cropclassification
      ```
      5. Now install cropclassification and dependencies that need pip with pip
      ```
      pip install cropclassification
      ```
4. Calculate time series 

To calculate time series, you need to run `cropclassification -t <tasks_dir>`, with a 'calc_timeseries' type of task in the tasks dir 
on a server that has access to sentinel CARD images.

Mind: the sentinel CARD image structure as expected for timeseries calculation depends on the image type:
  * for Sentinel 2 images this is the standard S2 L2A format as available on the open acces copernicus hub.
  * for Sentinel 1 backscatter and sentinel 1 coherence data this is a non-standardized data structure as there isn't a standard format (yet) for level 2 processed sentinel 1 images (as far as I know). Check out the code to see the expected data structure ;-). 

5. Start a crop classification

Run `cropclassification -t <tasks_dir>`, with a 'calc_marker' type of task in the tasks dir.
