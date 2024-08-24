################
Variable Intro
################

In this overview of variables in PySCIPOpt we'll walk through best
practices for modelling them and the various information that they
can be extracted from them.

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, quicksum

  scip = Model()

.. note:: In general you want to keep track of your variable objects.
  They can always be obtained from the model after they are added, but then
  the responsibility falls to the user to match them, e.g. by name or constraints
  they feature in.

.. contents:: Contents

Dictionary of Variables
=========================

Here we will store PySCIPOpt variables in a standard Python dictionary

.. code-block:: python

  var_dict = {}
  n = 5
  m = 5
  for i in range(n):
      var_dict[i] = {}
      for j in range(m):
          var_dict[i][j] = scip.addVar(vtype='B', name=f"x_{i}_{j}")

  example_cons_dict = {}
  for i in range(n):
      example_cons[i] = scip.addCons(quicksum(var_dict[i][j] for j in range(m)) == 1, name=f"cons_{i}")

List of Variables
===================


Numpy array of Variables
=========================

Variable Types
=================

SCIP has four different types of variables

Variable Information
=======================
