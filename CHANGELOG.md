# CHANGELOG

## Unreleased
### Changed
- `Model.from_ptr` and `Model.to_ptr` use a `PyCapsule` to exchange the SCIP pointer
  rather than an integer.

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
