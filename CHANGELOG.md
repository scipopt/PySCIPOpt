# CHANGELOG

## Unreleased
- remove Python 2 support
- add Model.getParams that returns a dict mapping all parameter names to their values
- add Model.setParams to set multiple parameters at once using a dict
- Add Model.from_ptr and Model.to_ptr to interface with SCIP* managed outside of PySCIPOpt
- Node.getParent() returns None if the node has no parent
- add NULL pointer checks to all Python wrapper classes
- add Event.getRow() and Row.name
- expose domain changes, bound changes, branching decisions, added constraints for nodes
- define Python object identity based on underlying SCIP object pointers, so that e.g. rows and columns can be added to sets, and testing for equality is consistent over time.

## 2.2.3 - 2019-12-10
### Added
- expose even more SCIP functionality in `scip.pxd`

### Changed
- move the main Python class declarations into scip.pxd; this makes it possible to write custom `pyx` extensions.

## 2.2.2 - 2019-11-27
### Added
- support for node selector plugin
- getOpenNodes() to access all open nodes

### Changed
- new Benders functionality

## 2.2.1 - 2019-09-17
### Added
- evaluate polynomial expressions using the given solution values

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
