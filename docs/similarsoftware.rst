#################
Similar Software
#################

.. contents:: Contents

Alternate MIP Solvers (with Python interfaces)
==============================================

In the following we will give a list of other mixed-integer optimizers with available python interfaces.
As each solver has its own set of problem classes that it can solve we will use a table with reference
keys to summarise these problem classes.

.. note:: This table is by no means complete.

.. note:: SCIP can solve of all the below problem classes (and many more).

.. list-table:: Label Summaries
  :widths: 25 25
  :align: center
  :header-rows: 1

  * - Key
    - Feature
  * - LP
    - Linear Programs
  * - MILP
    - Mixed-Integer Linear Programs
  * - QP
    - Quadratic Programs
  * - MIQP
    - Mixed-Integer Quadratic Programs
  * - QCP
    - Quadratically Constrained Programs
  * - MIQCP
    - Mixed-Integer Quadratically Constrained Programs
  * - MINLP
    - Mixed-Integer Nonlinear Programs
  * - PB
    - Pseudo-Boolean Problems
  * - SOCP
    - Second Order Cone Programming
  * - SDP
    - Semidefinite Programming
  * - MISDP
    - Mixed-Integer Semidefinite Programming

Open Source
***********

- `HiGHS <https://github.com/ERGO-Code/HiGHS>`_: O, LP, MILP, QP, MIQP
- `CLP / CBC <https://github.com/coin-or/CyLP>`_: O, LP, MILP
- `GLOP <https://github.com/google/or-tools>`_: O, LP

Closed Source
*************

- `CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_: LP, MILP, QP, MIQP, QCP, MIQCP, SOCP
- `Gurobi <https://www.gurobi.com/>`_: LP, MILP, QP, MIQP, QCP, MIQCP, MINLP, PB, SOCP
- `Xpress <https://www.fico.com/en/products/fico-xpress-optimization>`_: LP, MILP, QP, MIQP, QCP, MIQCP, MINLP, SOCP
- `COPT <https://www.copt.de/>`_: LP, MILP, QP, MIQP, QCP, MIQCP, SOCP, SDP
- `MOSEK <https://www.mosek.com/>`_: LP, MILP, QP, MIQP, QCP, MIQCP, SOCP, SDP, MISDP

General Modelling Frameworks (Solver Agnostic)
==============================================

This list will contain general modelling frameworks that allow you to use swap out SCIP for other
mixed-integer optimizers. While we recommend using PySCIPOpt for the many features it provides,
which are for the most part not available via general modelling frameworks,
if you want to simply use a mixed-integer optimizer then a general modelling framework
allows you to swap out the solver in a single line. This is a big advantage if you
are uncertain of the solver that you want to use, you believe the solver might change at some point,
or you want to compare the performance of different solvers.

- `Pyomo <https://github.com/Pyomo/pyomo>`_
- `CVXPy <https://github.com/cvxpy/cvxpy>`_
- `LINOPy <https://github.com/PyPSA/linopy>`_
- `PULP <https://github.com/coin-or/pulp>`_
- `PICOS <https://gitlab.com/picos-api/picos>`_


Software using PySCIPOpt
========================

This is software that is built on PySCIPOpt

- `GeCO <https://github.com/CharJon/GeCO>`_: Generators for Combinatorial Optimization
- `scip-routing <https://github.com/mmghannam/scip-routing>`_:  An exact VRPTW solver in Python
- `PySCIPOpt-ML <https://github.com/Opt-Mucca/PySCIPOpt-ML>`_:  Python interface to automatically formulate Machine Learning models into Mixed-Integer Programs
-  `SCIP Book <https://scipbook.readthedocs.io/en/latest/>`_: Mathematical Optimization: Solving Problems using SCIP and Python

Additional SCIP Resources
=========================

- `SCIP Website <https://scipopt.org/>`_
- `SCIP Documentation <https://scipopt.org/doc/html/>`_
- `SCIP GitHub <https://github.com/scipopt/scip>`_
