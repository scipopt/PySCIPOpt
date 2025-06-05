PySCIPOpt
=========

This project provides an interface from Python to the [SCIP Optimization Suite](https://www.scipopt.org/). Starting from v8.0.3, SCIP uses the [Apache2.0](https://www.apache.org/licenses/LICENSE-2.0) license. If you plan to use an earlier version of SCIP, please review [SCIP's license restrictions](https://scipopt.org/index.php#license).

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/PySCIPOpt/Lobby)
[![PySCIPOpt on PyPI](https://img.shields.io/pypi/v/pyscipopt.svg)](https://pypi.python.org/pypi/pyscipopt)
[![Integration test](https://github.com/scipopt/PySCIPOpt/actions/workflows/integration-test.yml/badge.svg)](https://github.com/scipopt/PySCIPOpt/actions/workflows/integration-test.yml)
[![coverage](https://img.shields.io/codecov/c/github/scipopt/pyscipopt)](https://app.codecov.io/gh/scipopt/pyscipopt/)
[![AppVeyor Status](https://ci.appveyor.com/api/projects/status/fsa896vkl8be79j9/branch/master?svg=true)](https://ci.appveyor.com/project/mattmilten/pyscipopt/branch/master)


Documentation
-------------

Please consult the [online documentation](https://pyscipopt.readthedocs.io/en/latest/) or use the `help()` function directly in Python or `?` in IPython/Jupyter.

The old documentation, which we are in the process of migrating from,
is still more complete w.r.t. the API, and can be found [here](https://scipopt.github.io/PySCIPOpt/docs/html/index.html)

See [CHANGELOG.md](https://github.com/scipopt/PySCIPOpt/blob/master/CHANGELOG.md) for added, removed or fixed functionality.

Installation
------------

The recommended installation method is via [PyPI](https://pypi.org/project/PySCIPOpt/):

```bash
pip install pyscipopt
```

To avoid interfering with system packages, it's best to use a [virtual environment](https://docs.python.org/3/library/venv.html):

```bash
python3 -m venv venv     # creates a virtual environment called venv
source venv/bin/activate # activates the environment. On Windows use: venv\Scripts\activate
pip install pyscipopt
```
Remember to activate the environment (`source venv/bin/activate` or equivalent) in each terminal session where you use PySCIPOpt.
Note that some configurations require the use of virtual environments.

For information on specific versions, installation via Conda, and guides for building from source,
please see the [online documentation](https://pyscipopt.readthedocs.io/en/latest/install.html).

Building and solving a model
----------------------------

There are several [examples](https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished) and
[tutorials](https://github.com/scipopt/PySCIPOpt/blob/master/examples/tutorial). These display some functionality of the
interface and can serve as an entry point for writing more complex code. Some of the common usecases are also available in the [recipes](https://github.com/scipopt/PySCIPOpt/blob/master/src/pyscipopt/recipes) sub-package.
You might also want to have a look at this article about PySCIPOpt:
<https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/6045>. The
following steps are always required when using the interface:

1)  It is necessary to import python-scip in your code. This is achieved
    by including the line

```python
from pyscipopt import Model
```

2)  Create a solver instance.

```python
model = Model("Example")  # model name is optional
```

3)  Access the methods in the `scip.pxi` file using the solver/model
    instance `model`, e.g.:

```python
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

Using PySCIPOpt?
----------------

If your project or company is using PySCIPOpt, consider letting us know at scip@zib.de. We are always interested
in knowing how PySCIPOpt is being used, and, given permission, would also appreciate adding your company's logo 
to our website.  

If you are creating models with some degree of complexity which don't take too long to solve, also consider
sharing them with us. We might want to add them to [`tests/helpers/utils.py`](tests/helpers/utils.py) to help make our tests more robust, or add them to our examples.

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
