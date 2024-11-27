############################
Frequently Asked Questions
############################

In this page we will provide some answers to commonly asked questions by users of PySCIPOpt.

How do I set parameter values?
==============================

See the appropriate section on :doc:`this page <tutorials/model>`.

How do I know the order that the different plug-ins will be called?
===================================================================

Each rule for a given plug-in is called in order of their priority.
To ensure that your custom rule is called first it must have a higher
priority than all other rules of that plug-in.

Problems with dual values?
==========================

See the appropriate section on :doc:`this page <tutorials/constypes>`. Short answer: SCIP cannot
guarantee accurate precise dual values without certain parameter settings.

Constraints with both LHS and RHS (ranged constraints)
======================================================

A ranged constraint takes the form:

.. code-block:: python

    lhs <= expression <= rhs

While these are supported, the Python syntax for chained comparisons can't be hijacked with operator overloading.
Instead, parenthesis must be used when adding your own ranged constraints, e.g.,

.. code-block:: python

    lhs <= (expression <= rhs)

Alternatively, you may call ``scip.chgRhs(cons, newrhs)`` or ``scip.chgLhs(cons, newlhs)`` after the single-sided
constraint has been created.

.. note:: Be careful of constant expressions being rearranged by Python when handling ranged consraints.

My model crashes when I make changes to it after optimizing
===========================================================

When dealing with an already optimized model, and you want to make changes, e.g., add a new
constraint or change the objective, please use the following command:

.. code-block:: python

    scip.freeTransform()

Without calling this function the Model can only be queried in its post optimized state.
This is because the transformed problem and all the previous solving information
is not automatically deleted, and thus stops a new optimization call.

Why can I not add a non-linear objective?
=========================================

SCIP does not support non-linear objectives, however, an equivalent optimization
problem can easily be constructed by introducing a single new variable and a constraint.
Please see :doc:`this page <tutorials/expressions>` for a guide.