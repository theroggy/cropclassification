# CHANGELOG

## ???

### Improvements

- Make image profiles to be used in a classification configurable in a config file (#56)
- Use ruff instead of black and flake for formatting and linting (#57, #64, #65, #67)
- Add task/action to automate periodic download of images (#67, #69)

## 0.1.1 (2023-08-08)

### Improvements

- Clip s2 and s1 timeseries values to one to avoid outliers > 1 (#47)
- Change default time_dimension_reducer to mean for both s1 and s2 (#48)

## 0.1.0
### Improvements

- Add support to use openeo for image retrieval/calculation (#36)
- Improve performance of zonal_stats_bulk (#38)
- Use black to comply to pep8 + minor general improvements (#13)
- Upgrade all dependencies (#12)
- Add support for pandas 2.0 (#21)
