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
  cons_2 = scip.addCons(x + z >= 3, name="cons_2")
  cons_3 = scip.addCons(y + z == 4, name="cons_3")
  scip.setObjective(2 * x + 3 * y - 5 * z, sense="minimize")
  scip.optimize()

That's it! We've built the optimization problem defined above and we've optimized it.
For how to read a Model from file see :doc:`this page </tutorials/readwrite>` and for best practices
on how to create more variables see :doc:`this page </tutorials/vartypes>`.

.. note:: ``vtype='C'`` here refers to a continuous variable.
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
      print(f"Variable {scip_var.name} has value {scip.getVal(scip_var)}")

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

Set Plugin-wide Parameters (Aggressiveness)
===========================================

We can influence the behavior of some of SCIP's plugins using ``SCIP_PARAMSETTING``. This can be applied 
to the heuristics, to the presolvers, and to the separators (respectively with ``setHeuristics``, 
``setPresolve``, and ``setSeparating``).

.. code-block:: python
  
  from pyscipopt import Model, SCIP_PARAMSETTING

  scip = Model()
  scip.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE) 

There are four parameter settings:

.. list-table:: A list of the different options and the result
  :widths: 25 25
  :align: center
  :header-rows: 1

  * - Option 
    - Result
  * - ``DEFAULT``
    - set to the default values of all the plugin's parameters
  * - ``FAST``
    - the time spend for the plugin is decreased
  * - ``AGGRESSIVE``
    - such that the plugin is called more aggressively
  * - ``OFF``
    - turn off the plugin

.. note:: This is important to get dual information, as it's necessary to disable presolving and heuristics. 
  For more information, see the tutorial on getting :doc:`constraint information.</tutorials/constypes/>`


Set Solver Emphasis
===================

One can also instruct SCIP to focus on different aspects of the search process. To do this, import 
``SCIP_PARAMEMPHASIS`` from ``pyscipopt`` and set the appropriate value. For example, 
if the goal is just to find a feasible solution, then we can do the following:

.. code-block:: python

    from pyscipopt import Model, SCIP_PARAMEMPHASIS

    scip = Model()
    scip.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)

You can find below a list of the available options, alongside their meaning.

.. list-table:: Parameter emphasis summary
    :widths: 25 25
    :align: center
    :header-rows: 1

    * - Setting 
      - Meaning
    * - ``PARAMEMPHASIS.DEFAULT`` 
      - to use default values
    * - ``PARAMEMPHASIS.COUNTER``
      - to get feasible and "fast" counting process
    * - ``PARAMEMPHASIS.CPSOLVER`` 
      - to get CP-like search (e.g. no LP relaxation)
    * - ``PARAMEMPHASIS.EASYCIP``
      - to solve easy problems fast
    * - ``PARAMEMPHASIS.FEASIBILITY`` 
      - to detect feasibility fast
    * - ``PARAMEMPHASIS.HARDLP``
      - to be capable to handle hard LPs
    * - ``PARAMEMPHASIS.OPTIMALITY``
      - to prove optimality fast
    * - ``PARAMEMPHASIS.PHASEFEAS``
      - to find feasible solutions during a 3 phase solution process
    * - ``PARAMEMPHASIS.PHASEIMPROVE``
      - to find improved solutions during a 3 phase solution process
    * - ``PARAMEMPHASIS.PHASEPROOF``
      - to proof optimality during a 3 phase solution process
    * - ``PARAMEMPHASIS.NUMERICS``
      - to solve problems which cause numerical issues

Copy a SCIP Model
==================

A SCIP Model can also be copied. This can be done with the following logic:

.. code-block:: python

  scip_alternate_model = Model(sourceModel=scip) # Assuming scip is a pyscipopt Model

This model is completely independent of the source model. The data has been duplicated.
That is, calling ``scip.optimize()`` at this point will have no effect on ``scip_alternate_model``.

.. note:: After optimizing users often struggle with reoptimization. To make changes to an
  already optimized model, one must first do the following:

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



