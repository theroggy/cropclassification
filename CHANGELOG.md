# CHANGELOG

## 0.3.0 (yyyy-mm-dd)

### Deprecations and compatibility notes

- Add feature "cross-prediction-models" to avoid markers being calculated on parcels
  using a model that used this parcel in its training (#142, #143)
- Consolidated some landcover pre-processing ignore codes (#120)
- Restructure + cleanup of configuration file, mainly to avoid duplicate keys and to
  improve parameter names:
    - Consolidate and rename `start_date_str`, `end_date_str` and
      `end_date_subtract_days` from sections `marker` and `calc_periodic_mosaic_params`
      to respectively `start_date`, `end_date` and `images_available_delay` in a new
      section `period`.
    - Consolidate `roi_bounds`, `roi_crs` and `roi_name` from
      sections `marker` and `calc_periodic_mosaic_params` to a new section `roi`.
    - Consolidate `marker.sensordata_to_use` and
      `calc_periodic_mosaic_params.imageprofiles_to_get` to `images.images`.
    - Move `marker.buffer` to section `timeseries`.
    - Move `marker.image_profiles_config_filepath` to
      `paths.image_profiles_config_filepath`
    - Move `marker.year` to `period.year`.
    - Drop `calc_periodic_mosaic_params.dest_image_data_dir` and use
      `dirs.images_periodic_dir` instead.
    - Rename section `dirs` to `paths`.
    - Remove `general.run_id`.

### Improvements

- Add some extra global accuracies (precision, recall, f1) to report (#119)
- Add option `images.on_missing_images` to be able to ignore errors in the
  calculation of images if needed (#125, #126, #138)
- Filter away rasterio logging for extrasamples (#127)
- Save openeo images with int16 bands as int16 locally as well (again) (#131, #135)
- Linting improvements: add isort, sync rules with geofileops (#133, #134)
- Add support for zonalstats calculation with ExactExctract (#139)

### Bugs fixed

- Fix `calc_periodic_mosaic_task` to read `roi_bounds` parameter as list of floats from
  config (#130)

## 0.2.0 (2024-06-17)

### Deprecations and compatibility notes

- To provide the possibility to specify any hyperparameter for sklearn classifiers, the
  parameters will now have to be specified as a json string instead of in individual
  parameters when such a classifier is used. Because of this, the following parameters
  become obsolete + the default values will become the default values in sklearn (#110):
    - randomforest_n_estimators: default 100 instead of 200
    - randomforest_max_depth: default None instead of 35
    - multilayer_perceptron_hidden_layer_sizes: default (100,) instead of (100, 100)
    - multilayer_perceptron_max_iter: default 200 instead of 1000
    - multilayer_perceptron_learning_rate_init
- For keras multilayer perceptron, some changes were applied to the default
  hyperparameters (#115)

### Improvements

- Add task/action to automate periodic download of images (#67)
- Add support to calculate indexes locally (#55)
- Improve config and handling of "weekly" and "biweekly" raster image periods (#78)
- Add possibility to configure any possible hyperparameter for the supported sklearn
  based classifiers (#110)
- Add support for HistGradientBoostingClassifier (#95)
- Improve configurability + defaults of keras mlp classifier (#115)
- Make image profiles to be used in a classification configurable in a config file (#56)
- Add option to overrule configuration parameters at runtime (#92)
- If image period is e.g. "weekly", align `start_date` of a marker to the next monday
  instead of the previous one to avoid using data outside the dates provided (#83, #84)
- Add method "best available pixel" on openeo for S2 (#70)
- Add utility script to recalculate reports for an existing run + make recalculation
  more robust for old runs (#91, #102, #103, #104, #106)
- Improve pixelcount calculation for parcels (#96, #105)
- Improve calculation of beta error in reporting (#97)
- Add "theta errors" to report + general reporting improvements (#114)
- Add whether a parcel has been used for training to output (#107)
- Run `bulk_zonal_stats` in low priority worker processes (#81)
- Use ruff instead of black and flake for formatting and linting (#57, #64, #65, #67)
- Updates to avoid warnings from (newer versions of) dependencies like pandas,
  geofileops (#88, #109)

### Bugs fixed

- Various fixes and improvements to `bulk_zonal_statistics` with engine="pyqgis"
  (#76, #80)
- Fix some group names being wrong/unclear in the classification reporting (#90)

## 0.1.1 (2023-08-08)

### Improvements

- Clip s2 and s1 timeseries values to one to avoid outliers > 1 (#47)
- Change default time_reducer to mean for both s1 and s2 (#48)

## 0.1.0
### Improvements

- Add support to use openeo for image retrieval/calculation (#36)
- Improve performance of zonal_stats_bulk (#38)
- Use black to comply to pep8 + minor general improvements (#13)
- Upgrade all dependencies (#12)
- Add support for pandas 2.0 (#21)
