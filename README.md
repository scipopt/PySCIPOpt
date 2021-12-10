PySCIPOpt
=========

This project provides an interface from Python to the [SCIP Optimization
Suite](https://www.scipopt.org/).

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/PySCIPOpt/Lobby)
[![PySCIPOpt on PyPI](https://img.shields.io/pypi/v/pyscipopt.svg)](https://pypi.python.org/pypi/pyscipopt)
[![Integration test](https://github.com/scipopt/PySCIPOpt/actions/workflows/integration-test.yml/badge.svg)](https://github.com/scipopt/PySCIPOpt/actions/workflows/integration-test.yml)
[![AppVeyor Status](https://ci.appveyor.com/api/projects/status/fsa896vkl8be79j9/branch/master?svg=true)](https://ci.appveyor.com/project/mattmilten/pyscipopt/branch/master)


Documentation
-------------

Please consult the [online documentation](https://scipopt.github.io/PySCIPOpt/docs/html) or use the `help()` function directly in Python or `?` in IPython/Jupyter.

See [CHANGELOG.md](CHANGELOG.md) for added, removed or fixed functionality.

Installation
------------

**Using Conda**

[![Conda version](https://img.shields.io/conda/vn/conda-forge/pyscipopt?logo=conda-forge)](https://anaconda.org/conda-forge/pyscipopt)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/pyscipopt?logo=conda-forge)](https://anaconda.org/conda-forge/pyscipopt)

Conda will install SCIP automatically, hence everything can be installed in a single command:
```bash
conda install --channel conda-forge pyscipopt
```

**Using PyPI and from Source**

See [INSTALL.md](INSTALL.md) for instructions.
Please note that the latest PySCIPOpt version is usually only compatible with the latest major release of the SCIP Optimization Suite.
The following table summarizes which version of PySCIPOpt is required for a given SCIP version:

|SCIP| PySCIPOpt |
|----|----|
8.0 | 4.x
7.0 | 3.x
6.0 | 2.x
5.0 | 1.4, 1.3
4.0 | 1.2, 1.1
3.2 | 1.0

Information which version of PySCIPOpt is required for a given SCIP version can also be found in [INSTALL.md](INSTALL.md).

Building and solving a model
----------------------------

There are several [examples](examples/finished) and
[tutorials](examples/tutorial). These display some functionality of the
interface and can serve as an entry point for writing more complex code.
You might also want to have a look at this article about PySCIPOpt:
<https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/6045>. The
following steps are always required when using the interface:

1)  It is necessary to import python-scip in your code. This is achieved
    by including the line

``` {.sourceCode .python}
from pyscipopt import Model
```

2)  Create a solver instance.

``` {.sourceCode .python}
model = Model("Example")  # model name is optional
```

3)  Access the methods in the `scip.pyx` file using the solver/model
    instance `model`, e.g.:

``` {.sourceCode .python}
x = model.addVar("x")
y = model.addVar("y", vtype="INTEGER")
model.setObjective(x + y)
model.addCons(2*x - y*y >= 0)
model.optimize()
sol = model.getBestSol()
print("x: {}".format(sol[x]))
print("y: {}".format(sol[y]))
```

Writing new plugins
-------------------

The Python interface can be used to define custom plugins to extend the
functionality of SCIP. You may write a pricer, heuristic or even
constraint handler using pure Python code and SCIP can call their
methods using the callback system. Every available plugin has a base
class that you need to extend, overwriting the predefined but empty
callbacks. Please see `test_pricer.py` and `test_heur.py` for two simple
examples.

Please notice that in most cases one needs to use a `dictionary` to
specify the return values needed by SCIP.

Extending the interface
-----------------------

PySCIPOpt already covers many of the SCIP callable library methods. You
may also extend it to increase the functionality of this interface. The
following will provide some directions on how this can be achieved:

The two most important files in PySCIPOpt are the `scip.pxd` and
`scip.pyx`. These two files specify the public functions of SCIP that
can be accessed from your python code.

To make PySCIPOpt aware of the public functions you would like to
access, you must add them to `scip.pxd`. There are two things that must
be done in order to properly add the functions:

1)  Ensure any `enum`s, `struct`s or SCIP variable types are included in
    `scip.pxd` <br>
2)  Add the prototype of the public function you wish to access to
    `scip.pxd`

After following the previous two steps, it is then possible to create
functions in python that reference the SCIP public functions included in
`scip.pxd`. This is achieved by modifying the `scip.pyx` file to add the
functionality you require.

We are always happy to accept pull request containing patches or
extensions!

Please have a look at our [contribution guidelines](CONTRIBUTING.md).

Gotchas
-------

### Ranged constraints

While ranged constraints of the form

``` {.sourceCode .}
lhs <= expression <= rhs
```

are supported, the Python syntax for [chained
comparisons](https://docs.python.org/3.5/reference/expressions.html#comparisons)
can't be hijacked with operator overloading. Instead, parenthesis must
be used, e.g.,

``` {.sourceCode .}
lhs <= (expression <= rhs)
```

Alternatively, you may call `model.chgRhs(cons, newrhs)` or
`model.chgLhs(cons, newlhs)` after the single-sided constraint has been
created.

### Variable objects

You can't use `Variable` objects as elements of `set`s or as keys of
`dict`s. They are not hashable and comparable. The issue is that
comparisons such as `x == y` will be interpreted as linear constraints,
since `Variable`s are also `Expr` objects.

### Dual values

While PySCIPOpt supports access to the dual values of a solution, there
are some limitations involved:

-   Can only be used when presolving and propagation is disabled to
    ensure that the LP solver - which is providing the dual
    information - actually solves the unmodified problem.
-   Heuristics should also be disabled to avoid that the problem is
    solved before the LP solver is called.
-   There should be no bound constraints, i.e., constraints with only
    one variable. This can cause incorrect values as explained in
    [\#136](https://github.com/scipopt/PySCIPOpt/issues/136)

Therefore, you should use the following settings when trying to work
with dual information:

``` {.sourceCode .python}
model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
model.disablePropagation()
```

Citing PySCIPOpt
----------------

Please cite [this paper](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/6045)
```
@incollection{MaherMiltenbergerPedrosoRehfeldtSchwarzSerrano2016,
  author = {Stephen Maher and Matthias Miltenberger and Jo{\~{a}}o Pedro Pedroso and Daniel Rehfeldt and Robert Schwarz and Felipe Serrano},
  title = {{PySCIPOpt}: Mathematical Programming in Python with the {SCIP} Optimization Suite},
  booktitle = {Mathematical Software {\textendash} {ICMS} 2016},
  publisher = {Springer International Publishing},
  pages = {301--307},
  year = {2016},
  doi = {10.1007/978-3-319-42432-3_37},
}
```
as well as the corresponding [SCIP Optimization Suite report](https://scip.zib.de/index.php#cite) when you use this tool for a publication or other scientific work.
