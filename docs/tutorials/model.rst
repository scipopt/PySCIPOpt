####################
Model Introduction
####################


The ``Model`` object is the central Python object that you will interact with. To use the ``Model`` object
simply import it from the package directly.

.. code-block:: python

  from pyscipopt import Model

  scip = Model()

.. contents:: Contents


Create a Model via Variables and Constraints
==============================================

While an empty Model is still something, we ultimately want a non-empty optimization problem. Let's
consider the basic optimization problem:

.. math::

  &\text{min} & &2x + 3y -5z \\
  &\text{s.t.} & &x + y \leq 5\\
  & & &x+z \geq 3\\
  & & &y + z = 4\\
  & & &(x,y,z) \in \mathbb{R}_{\geq 0}

We can construct the optimization problem as follows:

.. code-block:: python

  scip = Model()
  x = scip.addVar(vtype='C', lb=0, ub=None, name='x')
  y = scip.addVar(vtype='C', lb=0, ub=None, name='y')
  z = scip.addVar(vtype='C', lb=0, ub=None, name='z')
  cons_1 = scip.addCons(x + y <= 5, name="cons_1")
  cons_1 = scip.addCons(y + z >= 3, name="cons_2")
  cons_1 = scip.addCons(x + y == 5, name="cons_3")
  scip.setObjective(2 * x + 3 * y - 5 * z, sense="minimize")
  scip.optimize()

That's it! We've built the optimization problem defined above and run it.

.. note:: ``vtype='C'`` here refers to a continuous variables.
  Providing the lb, ub was not necessary as they default to (0, None) for continuous variables.
  Providing the name attribute is not necessary but is good practice.
  Providing the objective sense was also not necessary as it defaults to "minimize".


- create a model
- read a model
- optimize a problem (then query information)
- copy the model (I never do this)
- discuss freetransform



