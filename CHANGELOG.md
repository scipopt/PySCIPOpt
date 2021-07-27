# CHANGELOG
## Unreleased
### Added
### Fixed
### Changed
### Removed


## 3.2.2 - 2021-06-21 
### Added
- add SCIP functions `getNSolsFound`, `getNSolsFound`, `getNLimSolsFound` and `getNBestSolsFound`

### Fixed
### Changed
- documentation build parameter EXTRACT_ALL = YES

### Removed

## 3.2.1 - 2021-06-21
### Added
- Continuous Integration: move from travis to github actions:
  - integration with scipoptsuite and
  - publishing documentation and packages.

### Fixed
### Changed
### Removed
- CI files

## 3.2.0 - 2021-06-07
### Added
- add convenience function `Model.addConss()` to add multiple constraints at once

## 3.1.5 - 2021-05-23
### Added
- add SCIP function `SCIPsetMessagehdlrLogfile`
### Fixed
- fix `Model.writeLP` method

## 3.1.4 - 2021-04-25
### Fixed
- check for correct stage when querying solution values (raise warning otherwise)

## 3.1.3 - 2021-04-23
### Fixed
- check for NULL pointers when creating Solution objects (will return None): [#494](https://github.com/scipopt/PySCIPOpt/pull/494)

## 3.1.2 - 2021-04-07
### Added
- add `Model.getNReaders` that returns the number of available readers

## 3.1.1 - 2021-03-10
### Added
- add evaluation of `Expr` in `Solution`.

## 3.1.0 - 2020-12-17
### Added
- add more SCIP functions: `getNSols`, `createPartialSol`

### Fixed
- consistent handling of filenames for reading/writing
- fix error when no SCIPOPTDIR env var is set on Windows

## 3.0.4 - 2020-10-30
### Added
- add more SCIP functions: `getNTotalNodes`, `getNIntVars`, `getNBinVars`

### Fixed
- `getTransformedVar` now uses `SCIPgetTransformedVar` instead of `SCIPtransformVar` which captures the variable

## 3.0.3 - 2020-09-05
### Added
- add parameter genericnames to Model.writeProblem() to allow for generic variable and constraint names

### Fixed
- strip quotes from SCIPOPTDIR path variable that might confuse Windows systems

## 3.0.2 - 2020-08-09
### Added
- allow creation of implicit integer variables
- make some more SCIP functionality available

### Fixed
- fix reference counters for Python constraints

## 3.0.1 - 2020-07-05
### Added
- expose even more SCIP functionality in `scip.pxd`

### Changed
- `Model.from_ptr` and `Model.to_ptr` use a `PyCapsule` to exchange the SCIP pointer
  rather than an integer.
- mark getDualMultiplier() as deprecated, only getDualSolLinear() is supposed to be used to get duals of constraints

### Removed
* removed `__div__` from Expr and GenExpr to make it compatible with cython 0.29.20

## 3.0.0 - 2020-04-11
### Added
- add Model.getParams that returns a dict mapping all parameter names to their values
- add Model.setParams to set multiple parameters at once using a dict
- Add Model.from_ptr and Model.to_ptr to interface with SCIP* managed outside of PySCIPOpt
- add NULL pointer checks to all Python wrapper classes
- add Event.getRow() and Row.name
- expose domain changes, bound changes, branching decisions, added constraints for nodes
- define Python object identity based on underlying SCIP object pointers, so that e.g. rows and columns can be added to sets, and testing for equality is consistent over time.
- add Row.isRemovable and Row.getOrigintype
- add Model.applyCutsProbing and Model.propagateProbing
- add Model.separateSol
- add methods to work with nonlinear rows
- adds new "threadsafe" parameter to the PyBendersCopy member function. Also, the "threadsafe" parameter can be passed
  when creating a Model instance
- adds the boolean return options of "infeasible" and "auxviol" to the Benders.benderspresubsolve function. "infeasible"
  indicates that the input solution induces an infeasible instance of at least one Benders' subproblems. "auxviol"
  indicates that the objective value of at least on Benders' subproblem is greater than the auxiliary variable value.
- adds chgVarUbProbing and chgVarLbProbing to change a variables upper or lower bound during probing mode.

### Changed
- Node.getParent() returns None if the node has no parent
- remove Python 2.7 support
- fix documentation errors
- setupBendersSubproblem now requires a checktype input. This input indicates the reason for solving the Benders'
  subproblems, either enforcing the LP, relaxation or pseudo solution (LP, RELAX or PSEUDO) or checking a candidate
  primal feasible solution (CHECK).

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
