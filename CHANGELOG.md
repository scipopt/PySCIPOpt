# CHANGELOG

## Unreleased
- remove Python 2 support

## 2.2.0 - 2019-08-28
### Added
- convenient access to solution data, via `solution[x]` syntax instead of `model.getVal(x)`
- store Variable objects in Model for faster access

### Removed
- `releaseVar()`: potentially harmful and not necessary anymore

## 2.1.9 - 2019-08-20
### Changed
- recommend `pip install .` over `python setup.py install` in INSTALL

## 2.1.8 - 2019-08-19
### Added
- `lpiGetIterations()` to get LP iterations of last solved LP relaxation
### Changed
- rework `setup.py` to support developer versions of SCIP libraries built with Makefiles

## 2.1.7 - 2019-08-09
### Fixed
- fix methods to get and set char and string parameters
