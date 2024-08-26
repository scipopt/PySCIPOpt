######################
Branching Rule Intro
######################

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, quicksum

  scip = Model()

.. contents:: Contents

What is Branching
===================

Branching is when an optimization problem is split into smaller subproblems.
Traditionally this is done on an integer variable with a fractional LP solution, with
two child nodes being created with constraints ``x >= ceil(frac)`` and ``x <= floor(frac)``.
In SCIP, arbitrary amount of children nodes can be created, and the constraints added the
created nodes can also be arbitrary. This is not going to be used in the examples below, but this
should be kept in mind when considering your application of your branching rule.

Example Branching Rule
=======================

Here we will program a most infeasible branching rule

How to Include a Branching Rule
=================================

Strong Branching Information
=============================
