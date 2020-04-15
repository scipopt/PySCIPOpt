Requirements
============

PySCIPOpt requires a working installation of the [SCIP Optimization
Suite](http://scip.zib.de/). Please, make sure that your SCIP installation works!

Note that the latest PySCIPOpt version is usually only compatible with the latest major release of the SCIP Optimization Suite.
The following table summarizes which version of PySCIPOpt is required for a given SCIP version:

|SCIP| PySCIPOpt |
|----|----|
7.0 | 3.0
6.0 | 2.2, 2.1, 2.0
5.0 | 1.4, 1.3
4.0 | 1.2, 1.1
3.2 | 1.0

If SCIP is not installed in the global path
you need to specify the install location using the environment variable
`SCIPOPTDIR`:

-   on Linux and OS X:\
    `export SCIPOPTDIR=<path_to_install_dir>`
-   on Windows:\
    `set SCIPOPTDIR=<path_to_install_dir>` (**cmd**, **Cmder**, **WSL**)\
    `$Env:SCIPOPTDIR = "<path_to_install_dir>"` (**powershell**)

`SCIPOPTDIR` needs to have a subdirectory `lib` that contains the
library, e.g. `libscip.so` (for Linux) and a subdirectory `include` that
contains the corresponding header files:

    SCIPOPTDIR
      > lib
        > libscip.so ...
      > include
        > scip
        > lpi
        > nlpi
        > ...

If you are not using the installer packages, you need to [install the
SCIP Optimization Suite using CMake](http://scip.zib.de/doc/html/CMAKE.php).
The Makefile system is not compatible with PySCIPOpt!

On Windows it is highly recommended to use the [Anaconda Python
Platform](https://www.anaconda.com/).

Installation from PyPI
======================

`pip install pyscipopt`

On Windows you may need to ensure that the `scip` library can be found
at runtime by adjusting your `PATH` environment variable:

-   on Windows: `set PATH=%PATH%;%SCIPOPTDIR%\bin`

On Linux and OS X this is encoded in the generated PySCIPOpt library and
therefore not necessary.

Building everything from source
===============================

PySCIPOpt requires [Cython](http://cython.org/), at least version 0.21
(`pip install cython`). Furthermore, you need to have the Python
development files installed on your system (error message "Python.h not
found"):

    sudo apt-get install python-dev   # for Python 2, on Linux
    sudo apt-get install python3-dev  # for Python 3, on Linux

After setting up `SCIPOPTDIR` as specified above, please run

    pip install [-e] .

Building with debug information
===============================

To use debug information in PySCIPOpt you need to build it like this:

    pip install [-e] --install-option="--debug" .

Be aware that you will need the **debug library** of the SCIP
Optimization Suite for this to work
(`cmake .. -DCMAKE_BUILD_TYPE=Debug`).

Testing new installation
========================

To test your brand-new installation of PySCIPOpt you need
[pytest](https://pytest.org/) on your system. Here is the [installation
procedure](https://docs.pytest.org/en/latest/getting-started.html).

Tests can be run in the `PySCIPOpt` directory with: :

    py.test # all the available tests
    py.test tests/test_name.py # a specific tests/test_name.py (Unix)

Ideally, the status of your tests must be passed or skipped. Running
tests with pytest creates the `__pycache__` directory in `tests` and,
occasionally, a `model` file in the working directory. They can be
removed harmlessly.

Common errors
=============

-   readline: `libreadline.so.6: undefined symbol: PC` This is a
    readline/ncurses compatibility issue that can be fixed like this
    (when using `conda`):

        conda install -c conda-forge readline=6.2


