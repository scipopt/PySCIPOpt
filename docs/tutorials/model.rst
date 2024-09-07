#####################################################################
Introduction (Model Object, Solution Information, Parameter Settings)
#####################################################################


The ``Model`` object is the central Python object that you will interact with. To use the ``Model`` object
simply import it from the package directly.

.. code-block:: python

  from pyscipopt import Model

  scip = Model()

.. contents:: Contents


Create a Model, Variables, and Constraints
==============================================

While an empty Model is still something, we ultimately want a non-empty optimization problem. Let's
consider the basic optimization problem:

.. math::

  &\text{min} & \quad &2x + 3y -5z \\
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

That's it! We've built the optimization problem defined above and we've optimized it.
For how to read a Model from file see :doc:`this page </tutorials/readwrite>` and for best practices
on how to create more variables see :doc:`this page </tutorials/vartypes>`.

.. note:: ``vtype='C'`` here refers to a continuous variables.
  Providing the lb, ub was not necessary as they default to (0, None) for continuous variables.
  Providing the name attribute is not necessary either but is good practice.
  Providing the objective sense was also not necessary as it defaults to "minimize".

.. note:: An advantage of SCIP is that it can handle general non-linearities. See
  :doc:`this page </tutorials/expressions>` for more information on this.

Query the Model for Solution Information
=========================================

Now that we have successfully optimized our model, let's see some examples
of what information we can query. For example, the solving time, number of nodes,
optimal objective value, and the variable solution values in the optimal solution.

.. code-block:: python

  solve_time = scip.getSolvingTime()
  num_nodes = scip.getNTotalNodes() # Note that getNNodes() is only the number of nodes for the current run (resets at restart)
  obj_val = scip.getObjVal()
  for scip_var in [x, y, z]:
      print(f"Variable {scip_var.name} has value {scip.getVal(scip_var)})

Set / Get a Parameter
=====================

SCIP has an absolutely giant amount of parameters (see `here <https://www.scipopt.org/doc/html/PARAMETERS.php>`_).
There is one easily accessible function for setting individual parameters. For example,
if we want to set a time limit of 20s on the solving process then we would execute the following code:

.. code-block:: python

  scip.setParam("limits/time", 20)

To get the value of a parameter there is also one easily accessible function. For instance, we could
now check if the time limit has been set correctly with the following code.

.. code-block:: python

  time_limit = scip.getParam("limits/time")

A user can set multiple parameters at once by creating a dictionary with keys corresponding to the
parameter names and values corresponding to the desired parameter values.

.. code-block:: python

  param_dict = {"limits/time": 20}
  scip.setParams(param_dict)

To get the values of all parameters in a dictionary use the following command:

.. code-block:: python

  param_dict = scip.getParams()

Finally, if you have a ``.set`` file (common for using SCIP via the command-line) that contains
all the parameter values that you wish to set, then one can use the command:

.. code-block:: python

  scip.readParams(path_to_file)

Copy a SCIP Model
==================

A SCIP Model can also be copied. This can be done with the following logic:

.. code-block:: python

  scip_alternate_model = Model(sourceModel=scip) # Assuming scip is a pyscipopt Model

This model is completely independent from the source model. The data has been duplicated.
That is, calling ``scip.optimize()`` at this point will have no effect on ``scip_alternate_model``.

.. note:: After optimizing users often struggle with reoptimization. To make changes to an
  already optimized model, one must first fo the following:

  .. code-block:: python

    scip.freeTransform()

  Without calling this function the Model can only be queried in its post optimized state.
  This is because the transformed problem and all the previous solving information
  is not automatically deleted, and thus stops a new optimization call.

.. note:: To completely remove the SCIP model from memory use the following command:

  .. code-block:: python

    scip.freeProb()

  This command is potentially useful if there are memory concerns and one is creating a large amount
  of different SCIP models.



