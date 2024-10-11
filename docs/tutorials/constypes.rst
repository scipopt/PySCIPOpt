###################
Constraints in SCIP
###################

In this overview of constraints in PySCIPOpt we'll walk through best
practices for modelling them and the various information that they
can be extracted from them.

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, quicksum

  scip = Model()

.. note:: In general you want to keep track of your constraint objects.
  They can always be obtained from the model after they are added, but then
  the responsibility falls to the user to match them, e.g. by name.

.. contents:: Contents

What is a Constraint?
========================

A constraint in SCIP is likely much more broad in definition than you are familiar with.
For more information we recommend reading :doc:`this page </whyscip>`.

To create a standard linear or non-linear constraint use the command:

.. code-block:: python

  x = scip.addVar(vtype='B', name='x')
  y = scip.addVar(vtype='B', name='y')
  z = scip.addVar(vtype='B', name='z')
  # Linear constraint
  linear_cons = scip.addCons(x + y + z == 1, name="lin_cons")
  # Non-linear constraint
  nonlinear_cons = scip.addCons(x * y + z == 1, name="nonlinear_cons")


Quicksum
========

It is very common that when constructing constraints one wants to use the inbuilt ``sum`` function
in Python. For example, consider the common scenario where we have a set of binary variables.

.. code-block:: python

  x = [scip.addVar(vtype='B', name=f"x_{i}") for i in range(1000)]

A standard constraint in this example may be that exactly one binary variable can be active.
To sum these varaibles we recommend using the custom ``quicksum`` function, as it avoids
intermediate data structure and adds terms inplace. For example:

.. code-block:: python

  scip.addCons(quicksum(x[i] for i in range(1000)) == 1, name="sum_cons")

.. note:: While this is often unnecessary for smaller models, for larger models it can have a substantial
  improvement on time spent in model construction.

.. note:: For ``prod`` there also exists an equivalent ``quickprod`` function.

Constraint Information
======================

The Constraint object can be queried like any other object. Some of the information a Constraint
object contains is the name of the constraint handler responsible for the constraint,
and many boolean properties of the constraint, e.g., is it linear.

.. code-block:: python

  linear_conshdlr_name = linear_cons.getConshdlrName()
  assert linear_cons.isLinear()

As constraints are broader than the standard linear constraints most users are familiar with,
many of the functions that obtain constraint information are callable from the Model object.
These include the activity of the constraint, the slack of the constraint,
and adding or deleting coefficients.

.. code-block:: python

  if scip.getNSols() >= 1:
      scip_sol = scip.getBestSol()
      activity = scip.getActivity(linear_cons, scip_sol)
      slack = scip.getSlack(linear_cons, scip_sol)
  # Check current coefficients with scip.getValsLinear(linear_cons)
  scip.chgCoefLinear(linear_cons, x, 7) # Change the coefficient to 7

Currently not mentioned w.r.t. the constraints and rows is the dual information.
This is frustratingly complicated. SCIP has a plugin based LP solver, which offers many
choices for LP solvers, but makes getting information from them more complicated. Getting
dual values from constraints or rows will work, but to be confident that they are returning
the correct information we encourage doing three different things:

- Disable presolving and propagation to ensure that the LP solver
  - which is providing the dual information - actually solves the unmodified problem.
- Disable heuristics to avoid that the problem is solved before the LP solver is called.
- Ensure there are no bound constraints, i.e., constraints with only one variable.

To accomplish this one can apply the following settings to the Model.

.. code-block:: python

  from pyscipopt import SCIP_PARAMSETTING
  scip.setPresolve(SCIP_PARAMSETTING.OFF)
  scip.setHeuristics(SCIP_PARAMSETTING.OFF)
  scip.disablePropagation()

We stress again that when accessing such values you should be confident that you know which
LP is being referenced. This information for instance is unclear or difficult
to derive a meaningful interpretation from when the solution process has ended.
The dual value of a constraint can be obtained with the following code:

.. code-block:: python

  dual_sol = scip.getDualsolLinear(linear_cons)

Constraint Types
==================

In the above we presented examples of only linear constraints and a non-linear
constraint. SCIP however can handle many different types of constraints. Some of these that are
likely familiar are SOS constraints, Indicator constraints, and AND / OR / XOR constraints.
These constraint handlers have custom methods for improving the solving process of
optimization problems that they feature in. To add such a constraint, e.g., an SOS and indicator
constraint, you'd use the code:

.. code-block:: python

  sos_cons = scip.addConsSOS1([x, y, z], name="example_sos")
  indicator_cons = scip.addConsIndicator(x + y <= 1, binvar=z, name="example_indicator")

SCIP also allows the creation of custom constraint handlers. These could be empty and just
there to record data, there to provide custom handling of some user defined function, or they could be there to
enforce a constraint that is  incredibly inefficient to enforce via linear constraints.
An example of such a constraint handler
is presented in the lazy constraint tutorial for modelling the subtour elimination
constraints :doc:`here </tutorials/lazycons>`

What is a Row?
================

In a similar fashion to Variables with columns, see :doc:`this page </tutorials/vartypes>`,
constraints bring up an interesting feature of SCIP when used in the context of an LP.
The context of an LP here means that we are after the LP relaxation of the optimization problem
at some node. Is the constraint even in the LP?
When you solve an optimization problm with SCIP, the problem is first transformed. This process is
called presolve, and is done to accelerate the subsequent solving process. Therefore a constraint
that was originally created may have been transformed entirely, as the original variables that
featured in the constraint have also been changed. Additionally, maybe the constraint was found to be redundant,
i.e., trivially true, and was removed. The constraint is also much more general
than necessary, containing information that is not strictly necessary for solving the LP,
and may not even be representable by linear constraints.
Therefore, when representing a constraint in an LP, we use Row objects.
Be warned however, that this is not necessarily a simple one-to-one matching. Some more complicated
constraints may either have no Row representation in the LP or have multiple such rows
necessary to best represent it in the LP. For a standard linear constraint the Row
that represents the constraint in the LP can be found with the code:

.. code-block:: python

  row = scip.getRowLinear(linear_cons)

.. note:: Remember that such a Row representation refers only to the latest LP, and is
  best queried when access to the current LP is clear, e.g. when branching.

From a Row object one can easily obtain information about the current LP. Some quick examples are
the lhs, rhs, constant shift, the columns with non-zero coefficient values, the matching
coefficient values, and the constraint handler that created the Row.

.. code-block:: python

  lhs = row.getLhs()
  rhs = row.getRhs()
  constant = row.getConstant()
  cols = row.getCols()
  vals = row.getVals()
  origin_cons_name = row.getConsOriginConshdlrtype()