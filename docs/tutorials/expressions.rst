#######################
Non-Linear Expressions
#######################


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

Non-Linear Objectives
======================

While SCIP supports general non-linearities, it only supports linear objective functions.
With some basic reformulation this is not a restriction however. Let's consider the general
optimization problem:

.. math::

  &\text{min} & \quad &f(x) \\
  &\text{s.t.} & & g(x) \leq b \\
  & & & x \in \mathbb{Z}^{|\mathcal{J}|} \times \mathbb{R}^{[n] / \mathcal{J}}, \quad \mathcal{J} \subseteq [n] \\
  & & & f : \mathbb{R}^{n} \rightarrow \mathbb{R}, g : \mathbb{R}^{n} \rightarrow \mathbb{R}

Let's consider the case where ``f(x)`` is a non-linear function. This problem can be equivalently
reformulated as:

.. math::

  &\text{min} & \quad &y \\
  &\text{s.t.} & & g(x) \leq b \\
  & & & y \geq f(x) \\
  & & & x \in \mathbb{Z}^{|\mathcal{J}|} \times \mathbb{R}^{[n] / \mathcal{J}}, \quad \mathcal{J} \subseteq [n] \\
  & & & y \in \mathbb{R} \\
  & & & f : \mathbb{R}^{n} \rightarrow \mathbb{R}, g : \mathbb{R}^{n} \rightarrow \mathbb{R}

We've now obtained an equivalent problem with a linear objective function!
The same process can be performed with a maximization problem, albeit by introducing
a ``<=`` constraint for the introduced variable.

Let's see an example of how this would work when programming. Consider the simple problem:

.. math::

  &\text{min} & \quad &x^{2} + y \\
  &\text{s.t.} & & x + y \geq 5 \\
  & & & x + 1.3 y \leq 10 \\
  & & & (x,y) \in \mathbb{Z}^{2}

One can program an equivalent optimization problem with linear objective function as follows:

.. code-block:: python

  x = scip.addVar(vtype='I', name='x')
  y = scip.addVar(vtype='I', name='y')
  z = scip.addVar(vtype='I', name='z') # This will be our replacement objective variable
  cons_1 = scip.addCons(x + y >= 5, name="cons_1")
  cons_2 = scip.addCons(x + 1.3 * y <= 10, name="cons_2")
  cons_3 = scip.addCons(z >= x * x + y, name="cons_3")
  scip.setObjective(z)


Polynomials
============

Polynomials can be constructed directly from using Python operators on created variables.
Let's see an example of constructing the following constraint:

.. math::

  \frac{3x^{2} + y^{3}z^{2} + (2x + 3z)^{2}}{2(xz)} \leq xyz

The code for the following constraint can be written as follows:

.. code-block:: python

  x = scip.addVar(vtype='C', name='x')
  y = scip.addVar(vtype='C', name='y')
  z = scip.addVar(vtype='C', name='z')
  # Build the expression slowly (or do it all in the addCons call)
  lhs = 3 * (x ** 2) + ((y ** 3) * (z ** 2)) + ((2 * x) + (3 * z)) ** 2
  lhs = lhs / (2 * x * z)
  cons_1 = scip.addCons(lhs <= x * y * z, name="poly_cons")

Square Root (sqrt)
===================

There is native support for the square root function. Let's see an example for
constructing the following constraint:

.. math::

  \sqrt{x} \leq y

The code for the following constraint can be written as follows:

.. code-block:: python

  from pyscipopt import sqrt
  x = scip.addVar(vtype='C', name='x')
  y = scip.addVar(vtype='C', name='y')
  cons_1 = scip.addCons(sqrt(x) <= y, name="sqrt_cons")


Absolute (Abs)
===============

Absolute values of expressions is supported by overloading how ``__abs__`` function of
SCIP expression objects. Therefore one does not need to import any functions.
Let's see an example for constructing the following constraint:

.. math::

  |x| \leq y + 5

The code for the following constraint can be written as follows:

.. code-block:: python

  x = scip.addVar(vtype='C', lb=None, name='x')
  y = scip.addVar(vtype='C', name='y')
  cons_1 = scip.addCons(abs(x) <= y + 5, name="abs_cons")

.. note:: In general many constraints containing ``abs`` functions can be reformulated
  to linear constraints with the introduction of some binary variables. We recommend
  reformulating when it is easily possible, as it will in general improve solver performance.

Exponential (exp) and Log
==========================

There is native support for the exp and log functions. Let's see an example for
constructing the following constraints:

.. math::

  \frac{1}{1 + e^{-x}} &= y \\
  & \\
  \log (x) &\leq z

The code for the following constraint can be written as follows:

.. code-block:: python

  from pyscipopt import exp, log
  x = scip.addVar(vtype='C', name='x')
  y = scip.addVar(vtype='C', name='y')
  z = scip.addVar(vtype='C', name='z')
  cons_1 = scip.addCons( (1 / (1 + exp(-x))) == y, name="exp_cons")
  cons_2 = scip.addCons(log(x) <= z, name="log_cons)


Sin and Cosine (cos)
======================

There is native support for the sin and cos functions. Let's see an example for
constructing the following constraints:

.. math::

  sin(x) &= y \\
  & \\
  cos(y) & \leq 0.5 \\


The code for the following constraint can be written as follows:

.. code-block:: python

  from pyscipopt import cos, sin
  x = scip.addVar(vtype='C', name='x')
  y = scip.addVar(vtype='C', name='y')
  cons_1 = scip.addCons(sin(x) == y, name="sin_cons")
  cons_2 = scip.addCons(cos(y) <= 0.5, name="cos_cons")


