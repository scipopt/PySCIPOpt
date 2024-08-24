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

  example_cons = {}
  for i in range(n):
      example_cons[i] = scip.addCons(quicksum(var_dict[i][j] for j in range(m)) == 1, name=f"cons_{i}")

List of Variables
===================

Here we will store PySCIPOpt variables in a standard Python list

.. code-block:: python

  n, m = 5, 5
  var_list = [[None for i in range(m)] for i in range(n)]
  for i in range(n):
      for j in range(m):
          var_list[i][j] = scip.addVar(vtype='B', name=f"x_{i}_{j}")

  example_cons = []
  for i in range(n):
      example_cons.append(scip.addCons(quicksum(var_list[i][j] for j in range(m)) == 1, name=f"cons_{i}"))


Numpy array of Variables
=========================

Here we will store PySCIPOpt variables in a numpy ndarray

.. code-block:: python

  import numpy as np
  n, m = 5, 5
  var_array = np.zeros((n, m), dtype=object) # dtype is object allows arbitrary storage
  for i in range(n):
      for j in range(m):
          var_array[i][j] = scip.addVar(vtype='B', name=f"x_{i}_{j}")

  example_cons = np.zeros((n,), dtype=object)
  for i in range(n):
      example_cons[i] = scip.addCons(quicksum(var_dict[i][j] for j in range(m)) == 1, name=f"cons_{i}")

.. note:: An advantage of using numpy array storage is that you can then use numpy operators on
  the array of variables, e.g. reshape and stacking functions. It also means that you
  can form PySCIPOpt expressions in bulk, similar to matrix variables in some other
  packages. That is something like:

  .. code-block:: python

    a = np.random.uniform(size=(n,m))
    c = a @ var_array


Variable Types
=================

SCIP has four different types of variables:

.. list-table:: Variable Types
  :widths: 25 25 25
  :align: center
  :header-rows: 1

  * - Variable Type
    - Abbreviation
    - Description
  * - Continuous
    - C
    - A continuous variable belonging to the reals with some lower and upper bound
  * - Integer
    - I
    - An integer variable unable to take fractional values in a solution with some lower and upper bound
  * - Binary
    - B
    - A variable restricted to the values 0 or 1.
  * - Implicit Integer
    - M
    - A variable that is continuous but can be inferred to be integer in any valid solution

The variable type can be queried from the Variable object.

.. code-block:: python

  x = scip.addVar(vtype='C', name='x')
  assert x.vtype() == "CONTINUOUS"

Variable Information
=======================

- get objective coefficient
- get LP sol
- egt AvgSol
- Explain that one should use getVal to get the solution value in the primal

What is a Column?
==================

- explain what a column is
- explain how it differs
