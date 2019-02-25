=========
PySCIPOpt
=========

This project provides an interface from Python to the `SCIP Optimization Suite <http://scip.zib.de>`__.

|Gitter| |PyPI version| |Travis Status| |AppVeyor Status| |Coverage| |Health|


Installation
============

See `INSTALL.rst <INSTALL.rst>`__ for instructions.

Building and solving a model
============================

There are several `examples <examples/finished>`__ and `tutorials <examples/tutorials>`__.
These display some functionality of the interface and can serve as an entry
point for writing more complex code. You might also want to have a look
at this article about PySCIPOpt:
https://link.springer.com/chapter/10.1007%2F978-3-319-42432-3_37.

Minimal usage example:

1) Import the main class of the module:

.. code:: python

   from pyscipopt import Model

2) Create a solver instance:

.. code:: python

   model = Model()

3) Construct a model and solve it:

.. code:: python

   x = model.addVar("x")
   y = model.addVar("y", vtype="INTEGER")
   model.setObjective(x + y)
   model.addCons(2*x - y*y >= 0)
   model.optimize()

Writing new plugins
===================

PySCIPOpt can be used to define custom plugins to extend the
functionality of SCIP. You may write a pricer, heuristic or even
constraint handler using pure Python code and SCIP can call their
methods using the callback system. Every available plugin has a base
class that you need to extend, overwriting the predefined but empty
callbacks. Please see `test_pricer.py <tests/test_pricer.py>`__ and
`test_heur.py <tests/test_heur.py>`__ for two simple examples.

Please notice that in most cases one needs to use a ``dictionary`` to
specify the return values needed by SCIP.

Extending the interface
=======================

PySCIPOpt already covers many of the SCIP callable
library methods. You may also extend it to increase the
functionality of this interface. The following will provide some
directions on how this can be achieved:

The two most important files in PySCIPOpt are `scip.pxd <src/pyscipopt/scip.pxd>`__
and `scip.pyx <src/pyscipopt/scip.pxd>`__. These two files specify the
public functions of SCIP that can be accessed from PySCIPOpt and how this access
is to be performed.

To make PySCIPOpt aware of a public SCIP function you must add the declaration to
`scip.pxd <src/pyscipopt/scip.pxd>`__, including any missing ``enum``\ s,
``struct``\ s, SCIP variable types, etc.:

.. code:: python

  int SCIPgetNVars(SCIP* scip)

Then you can make the new function callable from Python by adding a new
wrapper in ``scip.pyx``:
   
.. code:: python
   
  def getNVars(self):
    """Retrieve number of variables in the problems"""
    return SCIPgetNVars(self._scip)

We are always happy to accept pull requests containing patches or extensions!

Please have a look at our `contribution guidelines <CONTRIBUTING.rst>`__.

How to cite
===========

Please refer to the `citing guidelines for SCIP <https://scip.zib.de/index.php#cite>`__
and add a reference to this article whenever you are using PySCIPOpt in a publication:
  
- Maher S., Miltenberger M., Pedroso J.P., Rehfeldt D., Schwarz R., Serrano F. (2016) PySCIPOpt: Mathematical Programming in Python with the SCIP Optimization Suite. In: Greuel GM., Koch T., Paule P., Sommese A. (eds) Mathematical Software – ICMS 2016. ICMS 2016. Lecture Notes in Computer Science, vol 9725. Springer, Cham

  https://link.springer.com/chapter/10.1007%2F978-3-319-42432-3_37


Gotchas
=======

Ranged constraints
------------------

While ranged constraints of the form

.. code::

    lhs <= expression <= rhs

are supported, the Python syntax for `chained
comparisons <https://docs.python.org/3.5/reference/expressions.html#comparisons>`__
can't be hijacked with operator overloading. Instead, parenthesis must
be used, e.g.,

.. code::

    lhs <= (expression <= rhs)

Alternatively, you may call ``model.chgRhs(cons, newrhs)`` or ``model.chgLhs(cons, newlhs)`` after the single-sided constraint has been created.

Variable objects
----------------

You can't use ``Variable`` objects as elements of ``set``\ s or as keys
of ``dict``\ s. They are not hashable and comparable. The issue is that
comparisons such as ``x == y`` will be interpreted as linear
constraints, since ``Variable``\ s are also ``Expr`` objects.

Dual values
-----------

While PySCIPOpt supports access to the dual values of a solution, there are some limitations involved:

- Can only be used when presolving and propagation is disabled to ensure that the LP solver - which is providing the dual information - actually solves the unmodified problem.
- Heuristics should also be disabled to avoid that the problem is solved before the LP solver is called.
- There should be no bound constraints, i.e., constraints with only one variable. This can cause incorrect values as explained in `#136 <https://github.com/SCIP-Interfaces/PySCIPOpt/issues/136>`__

Therefore, you should use the following settings when trying to work with dual information:

.. code:: python

   model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
   model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
   model.disablePropagation()

.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Gitter
   :target: https://gitter.im/PySCIPOpt/Lobby

.. |Travis Status| image:: https://travis-ci.org/SCIP-Interfaces/PySCIPOpt.svg?branch=master
   :alt: TravisCI Status
   :target: https://travis-ci.org/SCIP-Interfaces/PySCIPOpt

.. |Coverage| image:: https://img.shields.io/codecov/c/github/SCIP-Interfaces/PySCIPOpt/master.svg
   :alt: TravisCI Test Coverage
   :target: https://codecov.io/gh/SCIP-Interfaces/PySCIPOpt

.. |AppVeyor Status| image:: https://ci.appveyor.com/api/projects/status/fsa896vkl8be79j9?svg=true
   :alt: AppVeyor Status
   :target: https://ci.appveyor.com/project/mattmilten/pyscipopt

.. |PyPI version| image:: https://img.shields.io/pypi/v/pyscipopt.svg
   :alt: PySCIPOpt on PyPI
   :target: https://pypi.python.org/pypi/pyscipopt

.. |Health| image:: https://landscape.io/github/SCIP-Interfaces/PySCIPOpt/master/landscape.svg?style=flat
   :alt: Code Health
   :target: https://landscape.io/github/SCIP-Interfaces/PySCIPOpt/master
