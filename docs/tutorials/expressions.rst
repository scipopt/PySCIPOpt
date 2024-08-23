############################
Non-Linear Expressions Intro
############################


One of the big advantages of SCIP is that it handles arbitrary constraints.
Arbitrary here is not an exaggeration, see :doc:`the constraint tutorial </tutorials/constypes>`.
An advantage of this generality is that it has inbuilt support for many non-linear functions.
These non-linear expressions can be arbitrarily composed and SCIP will still find a globally
optimal solution within tolerances of the entire constraint. Below we will outline many of the
supported non-linear expressions.

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model

  scip = Model()

.. contents:: Contents

Polynomials
============


Absolute (Abs)
===============

Exponential (exp) and Log
==========================

Square Root (sqrt)
===================

Sin and Cosine (cos)
======================


