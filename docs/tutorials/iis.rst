#######################################
Irreducible Infeasible Subsystems (IIS)
#######################################

For the following, let us assume that a Model object is available, which is created as follows:

.. code-block:: python

    from pyscipopt import Model, IISfinder, SCIP_RESULT
    model = Model()

.. contents:: Contents

What is an IIS?
===============

It is a common issue for integer programming practitioners to (unexpectedly) encounter infeasible problems.
Often it is desirable to better understand exactly why the problem is infeasible.
Was it an error in the input data, was the underlying formulation incorrect, or was the model simply infeasible by construction?

A common tool for helping diagnose the reason for infeasibility is an **Irreducible Infeasible Subsystem (IIS)**.
An IIS is a subset of constraints and variable bounds from the original problem that:

1. Remains infeasible when considered together
2. Cannot be further reduced without the subsystem becoming feasible

Practitioners can use IIS finders to narrow their focus onto a smaller, more manageable problem.
Note, however, that there are potentially many different irreducible subsystems for a given infeasible problem, and that IIS finders may not provide a guarantee of an IIS of minimum size.

Generating an IIS
=================
Let us create a simple infeasible model and then generate an IIS for it.

.. code-block:: python

x1 = model.addVar("x1", vtype="B")
x2 = model.addVar("x2", vtype="B")
x3 = model.addVar("x3", vtype="B")

# These four constraints cannot be satisfied simultaneously
model.addCons(x1 + x2 == 1, name="c1")
model.addCons(x2 + x3 == 1, name="c2")
model.addCons(x1 + x3 == 1, name="c3")
model.addCons(x1 + x2 + x3 <= 0, name="c4")

model.optimize()
iis = model.generateIIS()

When you run this code, SCIP will output a log showing the progress of finding the IIS:

.. code-block:: text

    presolving:
    presolving (1 rounds: 1 fast, 0 medium, 0 exhaustive):
    2 deleted vars, 2 deleted constraints, 0 added constraints, 0 tightened bounds, 0 added holes, 0 changed sides, 0 changed coefficients
    0 implications, 0 cliques, 0 implied integral variables (0 bin, 0 int, 0 cont)
    presolving detected infeasibility
    Presolving Time: 0.00

    SCIP Status        : problem is solved [infeasible]
    Solving Time (sec) : 0.00
    Solving Nodes      : 0
    Primal Bound       : +1.00000000000000e+20 (0 solutions)
    Dual Bound         : +1.00000000000000e+20
    Gap                : 0.00 %
    time(s)| node  | cons  | vars  | bounds| infeasible
        0.0|      0|      1|      3|      6|         no
        0.0|      0|      2|      3|      6|         no
        0.0|      0|      4|      3|      6|         no
        0.0|      0|      3|      3|      6|        yes
        0.0|      0|      2|      3|      6|        yes
        0.0|      0|      2|      3|      6|        yes

    IIS Status            : irreducible infeasible subsystem (IIS) found
    IIS irreducible       : yes
    Generation Time (sec) : 0.01
    Generation Nodes      : 0
    Num. Cons. in IIS     : 2
    Num. Vars. in IIS     : 3
    Num. Bounds in IIS    : 6

After SCIP finds that the model is infeasible, see that SCIP's IIS finders alternate between including constraints to make the problem feasible, and removing constraints to make the problem as small as possible.
You see in the final statistics that the IIS is indeed irreducible, with 3 variables and 2 constraints.

.. note:: 
   While an already optimized infeasible model is not required to use the IIS functionality, it is
   encouraged to call this functionality only after ``model.optimize()``. Otherwise, SCIP will naturally optimize
   the base problem first to ensure that it is actually infeasible.

The IIS Object
==============

The ``IIS`` object returned by ``generateIIS()`` can be queried to access the following information:

- **time**: The CPU time spent finding the IIS
- **irreducible**: Boolean indicating if the IIS is irreducible
- **nodes**: Number of nodes explored during IIS generation
- **subscip**: A ``Model`` object containing the subscip with the IIS constraints

You can interact with the subscip to examine which constraints and variables are part of the IIS:

.. code-block:: python

  iis = model.generateIIS()
  subscip = iis.getSubscip()
  
  # Get constraints in the IIS
  for cons in subscip.getConss():
      print(f"Constraint: {cons.name}")
  
  # Get variables in the IIS
  for var in subscip.getVars():
      print(f"Variable: {var.name}")

Creating a Custom IIS Finder
=============================

You may want to implement your own algorithm to find an IIS.
PySCIPOpt supports this through the ``IISfinder`` class, which allows you to define custom logic
for identifying infeasible subsystems.

Basic Structure
---------------

To create a custom IIS finder, inherit from the ``IISfinder`` class and implement the ``iisfinderexec`` method.
This IISfinder just directly removes two constraints in the example above to yield an IIS: 

.. code-block:: python

    from pyscipopt import IISfinder, SCIP_RESULT

    class SimpleIISFinder(IISfinder):
        """
        Minimal working example: keep a known infeasible subset (by name)
        and mark it as the IIS.
        """
        
        def iisfinderexec(self):
            subscip = self.iis.getSubscip()

            # keep only constraints {c2, c4} and delete others
            keep = {"c2", "c4"}
            for cons in list(subscip.getConss()):
                if cons.name not in keep:
                    subscip.delCons(cons)

            # Tell SCIP that our sub-SCIP represents an (irreducible) IIS
            self.iis.setSubscipInfeasible(True)
            self.iis.setSubscipIrreducible(True)
            return {"result": SCIP_RESULT.SUCCESS}

Including Your Custom IIS Finder
---------------------------------

To use your custom IIS finder, include it in the model before calling ``generateIIS()``:

.. code-block:: python

  # Create model
  model = Model()
  
  # Add variables and constraints (infeasible problem)
  x1 = model.addVar("x1", vtype="B")
  x2 = model.addVar("x2", vtype="B")
  x3 = model.addVar("x3", vtype="B")
  
  model.addCons(x1 + x2 == 1, name="c1")
  model.addCons(x2 + x3 == 1, name="c2")
  model.addCons(x1 + x3 == 1, name="c3")
  model.addCons(x1 + x2 + x3 <= 0, name="c4")
  
  # Create and include the custom IIS finder
  simple_iis = SimpleIISFinder()
  model.includeIISfinder(
      simple_iis, 
      name="simpleiis",
      desc="Simple greedy IIS finder",
      priority=1000000  # Higher priority means it will be used first
  )
  
  # Solve to verify infeasibility
  model.optimize()
  
  # Generate IIS using our custom finder
  iis = model.generateIIS()
  
  # Examine the result
  print(f"\nIIS Information:")
  print(f"  Time: {iis.getTime():.2f} seconds")
  print(f"  Nodes: {iis.getNNodes()}")
  print(f"  Irreducible: {iis.isSubscipIrreducible()}")
  print(f"  Number of constraints: {iis.getSubscip().getNConss()}")

Key Methods in IISfinder
-------------------------

When implementing a custom IIS finder, you have access to several important methods:

- ``subscip=self.iis.getSubscip()``: Get the sub-problem (Model) containing the candidate IIS
- ``subscip.getConss()``: Get all constraints in the subscip
- ``subscip.delCons(cons)``: Remove a constraint from the subscip
- ``subscip.addCons(cons)``: Add a constraint back to the subscip
- ``subscip.optimize()``: Solve the subscip
- ``subscip.getStatus()``: Check if the subscip is infeasible
