#####################
Building From Source
#####################

When building PySCIPOpt from source, one must have their own installation of `SCIP <https://scipopt.org/>`_.
To download SCIP please either use the pre-built SCIP Optimization Suite available
`here <https://scipopt.org/index.php#download>`__ (recommended) or download SCIP and build from source itself from
`here <https://github.com/scipopt/scip>`__. A more minimal and experimental pre-built SCIP is available
`here <https://github.com/scipopt/scipoptsuite-deploy/releases>`__.

.. contents:: Contents

.. note:: The latest PySCIPOpt version is usually only compatible with the latest major release of the
  SCIP Optimization Suite. The following table summarizes which version of PySCIPOpt is required for a
  given SCIP version:

  .. list-table:: Supported SCIP versions for each PySCIPOpt version
    :widths: 25 25
    :align: center
    :header-rows: 1

    * - SCIP
      - PySCIPOpt
    * - 10.0.0
      - 6.0
    * - 9.2
      - 5.3, 5.4, 5.5, 5.6, 5.7 
    * - 9.1
      - 5.1, 5.2.x
    * - 9.0
      - 5.0.x
    * - 8.0
      - 4.x
    * - 7.0
      - 3.x
    * - 6.0
      - 2.x
    * - 5.0
      - 1.4, 1.3
    * - 4.0
      - 1.2, 1.1
    * - 3.2
      - 1.0

.. note:: If you install SCIP yourself and are not using the pre-built packages,
  you need to install the SCIP Optimization Suite using CMake.
  The Makefile system is not compatible with PySCIPOpt!

Download Source Code
======================

To download the source code for PySCIPOpt we recommend cloning the repository using Git. The two methods
for doing so are via SSH (recommended) and HTTPS.

.. code-block:: bash

  git clone git@github.com:scipopt/PySCIPOpt.git

.. code-block:: bash

  git clone https://github.com/scipopt/PySCIPOpt.git

One can also download the repository itself from GitHub `here <https://github.com/scipopt/PySCIPOpt>`__.

Requirements
==============

When building from source you must have the packages ``setuptools`` and ``Cython`` installed in your Python
environment. These can be installed via PyPI:

.. code-block:: bash

  pip install setuptools
  pip install Cython

.. note:: Since the introduction of Cython 3 we recommend building using ``Cython>=3``.

Furthermore, you must have the Python development files installed on your system.
Not having these files will produce an error similar to: ``(error message "Python.h not found")``.
To install these development files on Linux use the following command (change according to your distributions
package manager):

.. code-block:: bash

  sudo apt-get install python3-dev # Linux

.. note:: For other operating systems this may not be necessary as it comes with many Python installations.


Environment Variables
========================

When installing PySCIPOpt from source, Python must be able to find your installation of SCIP.
If SCIP is installed globally then this is not an issue, although we still encourage users to explicitly use
such an environment variable. If SCIP is not installed globally, then the user must set the appropriate
environment variable that points to the installation location of SCIP. The environment variable that must
be set is ``SCIPOPTDIR``.

For Linux and MacOS systems set the variable with the following command:

.. code-block:: bash

  export SCIPOPTDIR=<path_to_install_dir>

.. note::

  For macOS users, to ensure that the SCIP dynamic library can be found at runtime by PySCIPOpt,
  you should add your SCIP installation path to the ``DYLD_LIBRARY_PATH`` environment variable by running:

  .. code-block::

    export DYLD_LIBRARY_PATH="<path_to_install_dir>/lib:$DYLD_LIBRARY_PATH"

For Windows use the following command:

.. code-block:: bash

  set SCIPOPTDIR=<path_to_install_dir> # This is done for command line interfaces (cmd, Cmder, WSL)
  $Env:SCIPOPTDIR = "<path_to_install_dir>" # This is done for command line interfaces (powershell)

``SCIPOPTDIR`` should be a directory. It needs to have a subdirectory lib that contains the
library, e.g. libscip.so (for Linux) and a subdirectory include that contains the corresponding header files:

.. code-block:: RST

  SCIPOPTDIR
    > lib
      > libscip.so ...
    > include
      > scip
      > lpi
      > ...

.. note:: It is always recommended to use virtual environments for Python, see `here <https://virtualenv.pypa.io/en/latest/>`_.

  A virtual environment allows one to have multiple environments with different packages installed in each.
  To install a virtual environment simply run the command:

  .. code-block::

     python -m venv <venv_name e.g. venv>


Build Instructions
===================

After setting up the environment variables ``SCIPOPTDIR`` (see above) and installing all requirements
(see above), you can now install PySCIPOpt from source. To do so run the following command from the
main directory of PySCIPOpt (one with ``setup.py``, ``pyproject.toml`` and ``README.md``):

.. code-block:: bash

  # Set environment variable SCIPOPTDIR if not yet done
  python -m pip install .

For recompiling the source in the current directory use the command:

.. code-block:: bash

  python -m pip install --compile .

.. note:: Building PySCIPOpt from source can be slow. This is normal.

  If you want to build it quickly and unoptimised, which will affect performance
  (highly discouraged if running any meaningful time dependent experiments),
  you can set the environment variable ``export CFLAGS="-O0 -ggdb"`` (Linux example command)

Build with Debug
==================
To use debug information in PySCIPOpt you need to build it with the following command:

.. code-block::
  export PYSCIPOPT_DEBUG=True # With Windows CMD: set PYSCIPOPT_DEBUG=True
  python -m pip install .

.. note:: Be aware that you will need the debug library of the SCIP Optimization Suite for this to work
  (cmake .. -DCMAKE_BUILD_TYPE=Debug).

Testing the Installation
==========================

To test your brand-new installation of PySCIPOpt you need to
install some dependencies.

.. code-block:: bash

  pip install -r requirements/test.txt

Tests can be run in the PySCIPOpt directory with the commands:

.. code-block:: bash

  pytest # Will run all the available tests
  pytest tests/test_name.py # Will run a specific tests/test_name.py (Unix)
  pytest -nauto # Will run tests in parallel using all available cores

Ideally, the status of your tests must be passed or skipped.
Running tests with pytest creates the __pycache__ directory in tests and, occasionally,
a model file in the working directory. They can be removed harmlessly.

Building Documentation Locally
===============================

You can build the documentation locally with the command:

.. code-block:: bash

  pip install -r docs/requirements.txt
  sphinx-build docs docs/_build
