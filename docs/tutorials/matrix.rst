##############
Matrix API
##############

In this overview of the matrix variable and constraint API in PySCIPOpt
we'll walk through best practices for modelling them and the various information that
can be extracted from them.

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, quicksum

  scip = Model()


This tutorial should only be read after having an understanding of both the ``Variable``
object (see :doc:`the constraint tutorial </tutorials/vartypes>`) and the ``Constraint``
object (see :doc:`the constraint tutorial </tutorials/constypes>`).

.. note::

    The matrix API is built heavily on `numpy <https://numpy.org/>`_. This means that users can
    use all standard ``numpy`` operations that they are familiar with when handling matrix
    variables and expressions. For example, using the ``@``, ``matmul``, ``*``,
    ``+``, ``hstack``, ``vstack``, and ``**`` operations work exactly as they do
    when handling any standard ``numpy`` array.

.. contents:: Contents

What is a Matrix API?
======================

The standard approach explained in the variable and constraint tutorials, is to
build each variable yourself (storing them in some data structure, e.g., a list or dict,
with some loop), and to construct each constraint in a similar manner. That means building
up each constraint yourself term by term. This approach is flexible, and still remains the standard,
but an increasingly common trend is to view the modelling approach from a vector, matrix,
and tensor perspective. That is, directly operate on larger sets of variables and expressions,
letting python handle the interaction for each term. For such cases, it is encouraged
that users now use the new matrix API!

Matrix Variables
=================

Matrix variables are added via a single function call. It is important beforehand
to know the ``shape`` of the new set of variables you want to create, where ``shape``
is some ``tuple`` or ``int``. Below is an example for creating a 2x2 matrix variable
of type continuous with an ub of 8.

.. code-block:: python

    x = scip.addMatrixVar(shape, vtype='C', name='x', ub=8)

.. note::

    The ``name`` of each variable in the example above becomes ``x_(indices)``

In the case of each ``kwarg``, e.g., ``vtype`` and ``ub``, a ``np.array`` of explicit
values can be passed. In the example above, this means that each variable within the
matrix variable can have its own custom information. For example:

.. code-block:: python

    x = scip.addMatrixVar(shape, vtype='C', name='x', ub=np.array([[5, 6], [2, 8]]))

Matrix Constraints
===================

Matrix constraints follow the same logic as matrix variables. They can be constructed quickly
and added all at once. Some examples are provided below (these examples are nonsensical,
and there to purely understand the API):

.. code-block:: python

    x = scip.addMatrixVar(shape=(2, 2), vtype="B", name="x")
    y = scip.addMatrixVar(shape=(2, 2), vtype="C", name="y", ub=5)
    z = scip.addVar(vtype="C", name="z", ub=7)

    scip.addMatrixCons(x + y <= z)
    scip.addMatrixCons(exp(x) + sin(sqrt(y)) == z + y)
    scip.addMatrixCons(y <= x @ y <= x)
    scip.addMatrixCons(x.sum() <= 2)

.. note::

    When creating constraints, one can mix standard variables and values in the same
    expressions. ``numpy`` will then handle this, and broadcast the correct operations.
    In general this can be viewed as creating an imaginary ``np.array`` of the appropriate
    shape and populating it with the variable / value.

Class Properties
=================

A ``MatrixVariable`` and ``MatrixConstraint`` object have all the same getter
functions that are in general available for the standard equivalent. An example
is provided below for ``vtype``.

.. code-block:: python

    x = scip.addVar()
    matrix_x = scip.addMatrixVar(shape=(2,2))

    x.vtype()
    matrix_x.vtype()

The objects are not interchangeable however, when being passed into functions
derived from the ``Model`` class. That is, there is currently no global support,
that the following code runs:

.. code-block:: python

    scip.imaginary_function(x) # will always work
    scip.imaginary_function(matrix_x) # may have to access each variable manually

Accessing Variables and Constraints
===================================

After creating the matrix variables and matrix constraints,
one can always access the individual variables or constraints via their index.

.. code-block:: python

    x = scip.addMatrixVar(shape=(2, 2))
    assert(isinstance(x, MatrixVariable))
    assert(isinstance(x[0][0], Variable))
    cons = x <= 2
    assert(isinstance(cons, MatrixConstraint))
    assert(isinstance(cons[0][0]), Constraint)



