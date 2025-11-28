###########
Presolvers
###########

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, Presol, SCIP_RESULT, SCIP_PRESOLTIMING

  scip = Model()

.. contents:: Contents
----------------------


What is Presolving?
===================

Presolving simplifies a problem before the actual search starts. Typical
transformations include:

- tightening bounds,
- removing redundant variables/constraints,
- aggregating variables,
- detecting infeasibility early.

This can reduce numerical issues and simplify constraints and objective
expressions without changing the solution space.


The Presol Plugin Interface (Python)
====================================

A presolver in PySCIPOpt is a subclass of ``pyscipopt.Presol`` that implements the method:

- ``presolexec(self, nrounds, presoltiming)``

and is registered on a ``pyscipopt.Model`` via
the class method ``pyscipopt.Model.includePresol``.

Here is a high-level flow:

1. Subclass ``MyPresolver`` and capture any parameters in ``__init__``.
2. Implement ``presolexec``: inspect variables, compute transformations, call SCIP aggregation APIs, and return a result code.
3. Register your presolver using ``includePresol`` with a priority, maximal rounds, and timing.
4. Solve the model, e.g. by calling ``presolve`` or ``optimize``.


A Minimal Skeleton
------------------

.. code-block:: python

   from pyscipopt import Presol, SCIP_RESULT

   class MyPresolver(Presol):
       def __init__(self, someparam=123):
           self.someparam = someparam

       def presolexec(self, nrounds, presoltiming):
           scip = self.model

           # ... inspect model, change bounds, aggregate variables, etc. ...

           return {"result": SCIP_RESULT.SUCCESS}  # or DIDNOTFIND, DIDNOTRUN, CUTOFF


Example: Writing a Custom Presolver
===================================

This tutorial shows how to write a presolver entirely in Python using
PySCIPOpt's ``Presol`` plugin interface. We will implement a small
presolver that shifts variable bounds from ``[a, b]`` to ``[0, b - a]``
and optionally flips signs to reduce constant offsets.

For educational purposes, we keep our example as close as possible to SCIP's implementation, which can be found `here <https://scipopt.org/doc-5.0.1/html/presol__boundshift_8c_source.php>`__. However, one may implement Boundshift differently as SCIP's logic does not translate perfectly to Python. To avoid any confusion with the already implemented version of Boundshift, we will call our custom presolver *Shiftbound*.

A complete working example can be found in the directory:

- ``examples/finished/shiftbound.py``


Implementing Shiftbound
-----------------------

Below we walk through the important parts to illustrate design decisions to translate the Boundshift presolver to PySCIPOpt.

We want to provide parameters to control the presolver's behaviour:

- ``maxshift``: maximum length of interval ``b - a`` we are willing to shift,
- ``flipping``: allow sign flips for better numerics,
- ``integer``: only shift integer-ranged variables if true.

We will put these parameters into the ``__init__`` method to help us initialise the attributes of the presolver class. Then, in ``presolexec``, we implement the algorithm our custom presolver must follow.

.. code-block:: python

   import math
   from pyscipopt import SCIP_RESULT, Presol

   class ShiftboundPresolver(Presol):
       def __init__(self, maxshift=float("inf"), flipping=True, integer=True):
           self.maxshift = maxshift
           self.flipping = flipping
           self.integer = integer

       def presolexec(self, nrounds, presoltiming):
           scip = self.model

           # Utility replacements for a few SCIP helpers which are not exposed to PySCIPOpt
           # Emulate SCIP's absolute real value
           def REALABS(x): return math.fabs(x)

           # Emulate SCIP's "is integral" using the model's epsilon value
           def SCIPisIntegral(val):
               return val - math.floor(val + scip.epsilon()) <= scip.epsilon()

           # Emulate adjusted bound rounding for integral variables
           def SCIPadjustedVarBound(var, val):
               if val < 0 and -val >= scip.infinity():
                   return -scip.infinity()
               if val > 0 and val >= scip.infinity():
                   return scip.infinity()
               if var.vtype() != "CONTINUOUS":
                   return scip.feasCeil(val)
               if REALABS(val) <= scip.epsilon():
                   return 0.0
               return val

           # Respect global presolve switches (here, if aggregation disabled)
           if scip.getParam("presolving/donotaggr"):
               return {"result": SCIP_RESULT.DIDNOTRUN}

           # We want to operate on non-binary active variables only
           scipvars = scip.getVars()
           nbin = scip.getNBinVars()
           vars = scipvars[nbin:]  # SCIP orders by type: binaries first

           result = SCIP_RESULT.DIDNOTFIND

           for var in reversed(vars):
               if var.vtype() == "BINARY":
                   continue
               if not var.isActive():
                   continue

               lb = var.getLbGlobal()
               ub = var.getUbGlobal()

               # For integral types: round to feasible integers to avoid noise
               if var.vtype() != "CONTINUOUS":
                   assert SCIPisIntegral(lb)
                   assert SCIPisIntegral(ub)
                   lb = SCIPadjustedVarBound(var, lb)
                   ub = SCIPadjustedVarBound(var, ub)

               # Is the variable already fixed?
               if scip.isEQ(lb, ub):
                   continue

               # If demanded by the parameters, restrict to integral-length intervals
               if self.integer and not SCIPisIntegral(ub - lb):
                   continue

               # Only shift "reasonable" finite bounds
               MAXABSBOUND = 1000.0
               shiftable = all((
                   not scip.isEQ(lb, 0.0),
                   scip.isLT(ub, scip.infinity()),
                   scip.isGT(lb, -scip.infinity()),
                   scip.isLT(ub - lb, self.maxshift),
                   scip.isLE(REALABS(lb), MAXABSBOUND),
                   scip.isLE(REALABS(ub), MAXABSBOUND),
               ))
               if not shiftable:
                   continue

               # Create a new variable y with bounds [0, ub-lb], and same type
               newvar = scip.addVar(
                   name=f"{var.name}_shift",
                   vtype=var.vtype(),
                   lb=0.0,
                   ub=(ub - lb),
                   obj=0.0,
               )

               # Aggregate old variable with new variable:
               #   1.0 * var + 1.0 * newvar = ub        (flip), whichever yields smaller |offset|, or
               #   1.0 * var + (-1.0) * newvar = lb     (no flip)
               if self.flipping and (REALABS(ub) < REALABS(lb)):
                   infeasible, redundant, aggregated = scip.aggregateVars(var, newvar, 1.0,  1.0, ub)
               else:
                   infeasible, redundant, aggregated = scip.aggregateVars(var, newvar, 1.0, -1.0, lb)

               # Has the problem become infeasible? 
               if infeasible:
                   return {"result": SCIP_RESULT.CUTOFF}

               # Aggregation succeeded; SCIP marks var as redundant and keeps newvar for further search
               assert redundant
               assert aggregated
               result = SCIP_RESULT.SUCCESS

           return {"result": result}

Registering the Presolver
-------------------------

After having initialised our ``model``, we instantiate an object based on our ``ShiftboundPresolver`` including the parameters we wish our presolver's behaviour to be set to.
Lastly, we register the custom presolver by including ``presolver``, followed by a name and a description, as well as specifying its priority, maximum rounds to be called (where ``-1`` specifies no limit), and timing mode.

.. code-block:: python

   from pyscipopt import Model, SCIP_PRESOLTIMING, SCIP_PARAMSETTING

   model = Model()

   presolver = ShiftboundPresolver(maxshift=float("inf"), flipping=True, integer=True)
   model.includePresol(
       presolver,
       "shiftbound",
       "converts variables with domain [a,b] to variables with domain [0,b-a]",
       priority=7900000,
       maxrounds=-1,
       timing=SCIP_PRESOLTIMING.FAST,
   )
