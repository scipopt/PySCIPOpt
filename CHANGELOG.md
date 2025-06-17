# CHANGELOG

## Unreleased
### Added
- More support for AND-Constraints
- Added support for knapsack constraints
- Added isPositive(), isNegative(), isFeasLE(), isFeasLT(), isFeasGE(), isFeasGT(), isHugeValue(), and tests
- Added SCIP_LOCKTYPE, addVarLocksType(), getNLocksDown(), getNLocksUp(), getNLocksDownType(), getNLocksUpType(), and tests
### Fixed
- Raised an error when an expression is used when a variable is required
### Changed
### Removed

## 5.5.0 - 2025.05.06
### Added
- Wrapped SCIPgetChildren and added getChildren and test (also test getOpenNodes) 
- Wrapped SCIPgetLeaves, SCIPgetNLeaves, and added getLeaves, getNLeaves and test
- Wrapped SCIPgetSiblings, SCIPgetNSiblings, and added getSiblings, getNSiblings and test
- Wrapped SCIPdeactivatePricer, SCIPsetConsModifiable, and added deactivatePricer, setModifiable and test
- Added getLinearConsIndicator
- Added SCIP_LPPARAM, setIntParam, setRealParam, getIntParam, getRealParam, isOptimal, getObjVal, getRedcost for lpi
- Added isFeasPositive
- Added SCIP function SCIProwGetDualsol and wrapper getDualsol
- Added SCIP function SCIProwGetDualfarkas and wrapper getDualfarkas
### Fixed
- Fixed bug when accessing matrix variable attributes
### Changed
- Stopped tests from running in draft PRs
### Removed

## 5.4.1 - 2024.02.24
### Added
- Added option to get Lhs, Rhs of nonlinear constraints
- Added cutoffNode and test
- Added getMajorVersion, getMinorVersion, and getTechVersion
- Added addMatrixVar and addMatriCons
- Added MatrixVariable, MatrixConstraint, MatrixExpr, and MatrixExprCons
### Fixed
- Warning at Model initialisation now uses new version calls
### Changed
### Removed
- Removed universal wheel type from setup.cfg (support for Python 2)

## 5.3.0 - 2025.02.07
### Added
- Added cdef type declaration of loop variables for slight speedup
- Added wrappers for setting and getting heuristic timing
- Added transformed option to getVarDict, updated test
- Added categorical data example
- Added printProblem to print problem to stdout
- Added stage checks to presolve, freereoptsolve, freetransform
- Added primal_dual_evolution recipe and a plot recipe
- Added python wrappers for usage of SCIPcopyLargeNeighborhoodSearch, SCIPtranslateSubSol and SCIPhashmapCreate
### Fixed
- Added default names to indicator constraints
### Changed
- GitHub actions using Mac now use precompiled SCIP from latest release
### Removed

## 5.2.1 - 2024.10.29
### Added
- Expanded Statistics class to more problems.
- Created Statistics class
- Added parser to read .stats file
- Release checklist in `RELEASE.md`
- Added Python definitions and wrappers for SCIPstartStrongbranch, SCIPendStrongbranch SCIPgetBranchScoreMultiple, 
  SCIPgetVarStrongbranchInt, SCIPupdateVarPseudocost, SCIPgetVarStrongbranchFrac, SCIPcolGetAge, 
  SCIPgetVarStrongbranchLast, SCIPgetVarStrongbranchNode, SCIPallColsInLP, SCIPcolGetAge
- Added getBipartiteGraphRepresentation
- Added helper functions that facilitate testing
- Added Python definitions and wrappers for SCIPgetNImplVars, SCIPgetNContVars, SCIPvarMayRoundUp,
  SCIPvarMayRoundDown, SCIPcreateLPSol, SCIPfeasFloor, SCIPfeasCeil, SCIPfeasRound, SCIPgetPrioChild,
  SCIPgetPrioSibling
- Added additional tests to test_nodesel, test_heur, and test_strong_branching
- Migrated documentation to Readthedocs
- `attachEventHandlerCallback` method to Model for a more ergonomic way to attach event handlers
- Added Model method: optimizeNogil
- Added Solution method: getOrigin, retransform, translate
- Added SCIP.pxd: SCIP_SOLORIGIN, SCIPcopyOrigVars, SCIPcopyOrigConss, SCIPsolve nogil, SCIPretransformSol, SCIPtranslateSubSol, SCIPsolGetOrigin, SCIPhashmapCreate, SCIPhashmapFree
- Added additional tests to test_multi_threads, test_solution, and test_copy
### Fixed
- Fixed too strict getObjVal, getVal check
### Changed
- Changed createSol to now have an option of initialising at the current LP solution
- Unified documentation style of scip.pxi to numpydocs
### Removed

## 5.1.1 - 2024-06-22
### Added
- Added SCIP_STATUS_DUALLIMIT and SCIP_STATUS_PRIMALLIMIT
- Added SCIPprintExternalCodes (retrieves version of linked symmetry, lp solver, nl solver etc)
- Added recipe with reformulation for detecting infeasible constraints
- Wrapped SCIPcreateOrigSol and added tests 
- Added verbose option for writeProblem and writeParams
- Expanded locale test
- Added methods for creating expression constraints without adding to problem
- Added methods for creating/adding/appending disjunction constraints
- Added check for pt_PT locale in test_model.py
- Added SCIPgetOrigConss and SCIPgetNOrigConss Cython bindings. 
- Added transformed=False option to getConss, getNConss, and getNVars
### Fixed
- Fixed locale errors in reading
### Changed
- Made readStatistics a standalone function
### Removed

## 5.0.1 - 2024-04-05
### Added
- Added recipe for nonlinear objective functions
- Added method for adding piecewise linear constraints
- Add SCIP function SCIPgetTreesizeEstimation and wrapper getTreesizeEstimation
- New test for model setLogFile
### Fixed
- Fixed locale fix
- Fixed model.setLogFile(None) error
- Add recipes sub-package
- Fixed "weakly-referenced object no longer exists" when calling dropEvent in test_customizedbenders
- Fixed incorrect writing/printing when user had a non-default locale
### Changed
### Removed

## 5.0.0 - 2024-03-05
### Added
- Added SCIP function addExprNonlinear
- Add support for Cython 3
- Added methods for getting the names of the current stage and of an event
- Add support for SCIP symmetry graph callbacks in constraint handlers
### Fixed
- Fixed README links 
- Fixed outdated time.clock call in gcp.py
### Changed
- Changed default installation option via pypi to package pre-build SCIP

## 4.4.0 - 2023-12-04
### Added
- Add getConshdlrName to class Constraint
- Added all event types and tests for checking them 
- Added SCIP functions SCIPconsGetNVars, SCIPconsGetVars
- Added SCIP functions SCIPchgCoefLinear, SCIPaddCoefLinear and SCIPdelCoefLinear
- Added SCIP function SCIPgetSolTime and wrapper getSolTime
- Added convenience methods relax and getVarDict
- Added SCIP functions hasPrimalRay, getPrimalRay, getPrimalRayVal
### Fixed
- Fixed mistake with outdated values for several enums
- Correctly set result, lowerbound in PyRelaxExec
- Fixed typo in documentation of chgRhs
- Pricer plugin fundamental callbacks now raise an error if not implemented
- Brachrule plugin fundamental callbacks now raise an error if not implemented
- Fixed segmentation fault when accessing the Solution class directly
- Changed getSols so that it prints solutions in terms of the original variables
- Fixed error message in _checkStage
### Changed
- Made it so SCIP macros are used directly, instead of being manually inputted. 
- Improved error message when using < or > instead of <= or >=
### Removed
- Removed double declaration of SCIPfindEventhdlr

## 4.3.0 - 2023-03-17
### Added
- Add SCIP function SCIprowGetOriginCons
- Add getConsOriginConshdlrtype to Row
- Add possibility to use sine and cosing
- Add ability to set priced variable score 

### Fixed
### Changed
### Removed
- Removed function rowGetNNonz

## 4.2.0 - 2022-03-21
### Added
- Interface to include custom reader plugins
- New test for reader plugin
### Fixed
- revert change from #543 to fix #570 (closing file descriptors)
- use correct offset value when updating the objective function
### Changed
### Removed

## 4.1.0 - 2022-02-22
### Added
- Interface to include custom cut selector plugins
- New test for cut selector plugin
- Add SCIP function SCIPgetCutLPSolCutoffDistance and wrapper getCutLPSolCutoffDistance
- Add SCIP function SCIPprintBestTransSol and wrapper writeBestTransSol
- Add SCIP function SCIPprintTransSol and wrapper writeTransSol
- Add SCIP function SCIPgetRowNumIntCols and wrapper getRowNumIntCols
- Add SCIP function SCIProwGetNNonz and wrapper rowGetNNonz
- Add SCIP function SCIPgetRowObjParallelism and wrapper getRowObjParallelism
- Add SCIP function SCIPgetNSepaRounds and wrapper getNSepaRounds
- Add SCIP function SCIPgetRowLinear and wrapper getRowLinear
- Add SCIP function SCIProwIsInGlobalCutpool and wrapper isInGlobalCutpool
- Add SCIP function SCIProwGetParallelism and wrapper getRowParallelism
- Add getObjCoeff call to Column
- Add isLocal call to Row
- Add getNorm call to Row
- Add getRowDualSol to Row
- Add getDualSolVal to Model
- added activeone parameter of addConsIndicator() allows to activate the constraint if the binary (indicator) variable is 1 or 0.
- added function getSlackVarIndicator(), returns the slack variable of the indicator constraint.
### Fixed
- cmake / make install works from build directory
### Changed
### Removed

## 4.0.0 - 2021-12-15
### Added
- many functions regarding the new cons expression logic and implementation
### Fixed
- fixed tests and github actions to fit the new SCIP version.
### Changed
- SCIP8 changes the way nonlinear constraints are handled inside SCIP. These changes have consequences for their respective PySCIPOpt wrappers and have changed regardingly. Please refer to the latest SCIP report for an in-depth explanation of these changes.
- small changes to the documentation.
### Removed
- some of the deprecated functions that could not be made backwards compatible

## 3.5.0 - 2021-12-07
### Added
### Fixed
- close file descriptors after file operation is finished
- fixed deletion of variable pointer from model when calling delVar
- fixed scip install for MAC integration test
- Fixing assert failure if scip is compiled using quadprecision
- fixed missing GIL by @AntoinePrv in #539
### Changed
- changed integration test to include scip TPI (tinycthreads)
### Removed
- removed Mac integration test until the segmentation fault in test_memory.py is fixed on Mac 

## 3.4.0 - 2021-10-30
### Added
- add support for concurrent optimization
  - note that SCIP needs to be linked to a TPI (task processing interface) to use this feature
- SCIPsolverConcurrent implementation from issue #229 by @TNonet in #535
- fix action to also run on external PRs by @mattmilten in #536
- fix concurrent solve test by @mattmilten in #537

## 3.3.0 - 2021-08-23
### Added
- add SCIP function `getPseudoBranchCands`

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
