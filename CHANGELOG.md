# CHANGELOG

## ???

### Improvements

- Add task/action to automate periodic download of images (#67)
- Add support to calculate indexes locally (#55)
- Make image profiles to be used in a classification configurable in a config file (#56)
- Run `bulk_zonal_stats` in low priority worker processes (#81)
- Use ruff instead of black and flake for formatting and linting (#57, #64, #65, #67)

### Bugs fixed

- Various fixes and improvements to `bulk_zonal_statistics` with engine="pyqgis"
  (#76, #80)

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
