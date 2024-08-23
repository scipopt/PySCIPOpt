#####################
Read and Write Intro
#####################

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model

  scip = Model()

.. contents:: Contents

Model File Formats
=====================

SCIP has extensive support for a wide variety of file formats. The table below outlines
what formats those are and the model types they're associated with.

.. list-table:: Supported File Formats
  :widths: 25 25
  :align: center
  :header-rows: 1

  * - Extension
    - Model Type
  * - CIP
    - SCIP's constraint integer programming format
  * - CNF
    - DIMACS CNF (conjunctive normal form) format used for example for SAT problems
  * - DIFF
    - reading a new objective function for mixed-integer programs
  * - FZN
    - FlatZinc is a low-level solver input language that is the target language for MiniZinc
  * - GMS
    - mixed-integer nonlinear programs (GAMS) [reading requires compilation with GAMS=true and a working GAMS system]
  * - LP
    - mixed-integer (quadratically constrained quadratic) programs (CPLEX)
  * - MPS
    - mixed-integer (quadratically constrained quadratic) programs
  * - OPB
    - pseudo-Boolean optimization instances
  * - OSiL
    - mixed-integer nonlinear programs
  * - PIP
    - mixed-integer polynomial programming problems
  * - SOL
    - solutions; XML-format (read-only) or raw SCIP format
  * - WBO
    - weighted pseudo-Boolean optimization instances
  * - ZPL
    - ZIMPL models, i.e., mixed-integer linear and nonlinear programming problems [read only]


.. note:: In general we recommend sharing files using the ``.mps`` extension when possible.

  For a more human readable format for equivalent problems we then recommend the ``.lp`` extension.

  For general non-linearities that are to be shared with others we recommend the ``.osil`` extension.

  For general constraint types that will only be used by other SCIP users we recommend the ``.cip`` extension.

Write a Model
================

To write a SCIP Model to a file one simply needs to run the command:

.. code-block:: python

  from pyscipopt import Model
  scip = Model()
  scip.writeProblem(filename="example_file.mps", trans=False, genericnames=False)

.. note:: Both ``trans`` and ``genericnames`` are there as their default values. The ``trans``
  option is available if you want to print the transformed problem (post presolve) instead
  of the model originally created. The ``genericnames`` option is there if you want to overwrite
  the variable and constraint names provided.

Read a Model
===============

To read in a file to a SCIP model one simply needs to run the command:

.. code-block:: python

  from pyscipopt import Model
  scip = Model()
  scip.readProblem(filename="example_file.mps")

This will read in the file and you will now have a SCIP model that matches the file.
Variables and constraints can be queried, with their names matching those in the file.
