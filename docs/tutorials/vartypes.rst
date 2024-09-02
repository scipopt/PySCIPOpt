####################
Variable Information
####################

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

In this subsection we'll walk through some functionality that is possible with the variable
objects.

First, we can easily obtain the objective coefficient of a variable.

.. code-block:: python

  scip.setObjective(2 * x)
  assert x.getObj() == 2.0

Assuming we have a solution to our problem, we can obtain the variable solution value
in the current best solution with the command:

.. code-block:: python

  var_val = scip.getVal(x)

An alternate way to obtain the variable solution value (of whatever solution you wish) is
to query the solution object with the SCIP expression (potentially just the variable)

.. code-block:: python

  if scip.getNSols() >= 1:
      scip_sol = scip.getBestSol()
      var_val = scip_sol[x]

We can also obtain the LP solution of a variable. This would be used when you have included your own
plugin, and are querying specific information for a given LP relaxation at some node. This is not the
variable solution value in the final optimal solution!

.. code-block:: python

  lp_val = x.getLPSol()

The LP solution value brings up an interesting feature of SCIP. Is the variable even in the LP?
When you solve an optimization problm with SCIP, the problem is first transformed. This process is
called presolve, and is done to speed up the subsequent solving process. Therefore a variable
that was originally created may have been transformed entirely, or may have just been removed
from the transformed problem as it is redundant. The variable may also not exist because you
are currently doing some pricing, and the LP only contains a subset of the variables. The summary is:
It should not be taken for granted that your originally created variable is in an LP.
This can be checked with the following code:

.. code-block:: python

  is_in_lp = x.isInLP()
  if is_in_lp:
      print("Variable is in LP!")
  else:
      print("Variable is not in LP!")

Now to some additional confusion. When you're solving an LP do you actually want a variable object?
The variable object contains a lot of unnecessary information that is not needed to strictly
solve the LP. This information will also have to be sent to the LP solver because SCIP is a plugin
based solver and can use many different LP solvers. Therefore, if the variable is in the LP,
it is represented by a column. The column object is the object that is actually used when solving the LP.
The column for a variable can be found with the following code:

.. code-block:: python

  col = x.getCol()

Information that is LP specific can be queried by the column directly. This includes the
objective value coefficient, the LP solution value, lower and upper bounds,
and of course the variable that it represents.

.. code-block::

  obj_coeff = col.getObjCoeff()
  lp_val = col.getPrimsol()
  lb = col.getLb()
  ub = col.getUb()
  x = col.getVar()
