Requirements
============

From version 5.0.0 SCIP is automatically shipped when using PyPI for the following systems:

- CPython 3.8 / 3.9 / 3.10 / 3.11 / 3.12 for  Linux (manylinux2014)
- CPython 3.8 / 3.9 / 3.10 / 3.11 / 3.12 for MacOS for x86_64 / ARM64
- CPython 3.8 / 3.9 / 3.10 / 3.11 / 3.12 for Windows 64bit

This work is currently ongoing, and is planned to encompass all Python and OS combinations. 

Please note that to use any of these Python / OS combinations without the default SCIP, one must build PySCIPOpt from source and
link against your own installation of SCIP. Such a scenario is common if the LP plugin should be Gurobi / Xpress / CPLEX instead of Soplex.

When installing from source or using PyPI with a python version and operating system combination that is not mentioned above,
PySCIPOpt requires a working installation of the [SCIP Optimization
Suite](https://www.scipopt.org/). Please, make sure that your SCIP installation works!

**Note that the latest PySCIPOpt version is usually only compatible with the latest major release of the SCIP Optimization Suite. See the table on the README.md page for details.**

If you install SCIP yourself and are not using the installer packages, you need to [install the
SCIP Optimization Suite using CMake](https://www.scipopt.org/doc/html/md_INSTALL.php#CMAKE).
The Makefile system is not compatible with PySCIPOpt!

If installing SCIP from source or using PyPI with a python and operating system that is not mentioned above, and SCIP is not installed in the global path,
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

Please note that some Mac configurations require adding the library installation path to `DYLD_LIBRARY_PATH` when using a locally installed version of SCIP.


When building SCIP from source using Windows it is highly recommended to use the [Anaconda Python
Platform](https://www.anaconda.com/).

Installation from PyPI
======================

    python -m pip install pyscipopt

To avoid interfering with system packages, it's best to use a [virtual environment](https://docs.python.org/3/library/venv.html).<br>
<span style="color:orange">**Warning!**</span> This is mandatory in some newer configurations.

```bash
python3 -m venv venv source venv/bin/activate  # On Windows use: venv\Scripts\activate pip install pyscipopt
pip install pyscipopt
```
Remember to activate the environment (`source venv/bin/activate`) in each terminal session where you use PySCIPOpt.

Please note that if your Python version and OS version are in the combinations at the start of this INSTALL file then
pip now automatically installs a pre-built version of SCIP. For these combinations, to use your own installation of SCIP,
please see the section on building from source. For unavailable combinations this pip command will automatically
search your global installs or custom set paths as above.

On Windows for combinations not listed at the start of this file, you may need to ensure that the `scip` library can be found
at runtime by adjusting your `PATH` environment variable:

-   on Windows: `set PATH=%PATH%;%SCIPOPTDIR%\bin`

On Linux and OS X this is encoded in the generated PySCIPOpt library and
therefore not necessary.

Building everything from source
===============================

Recommended is to install in a virtual environment (e.g. `python3 -m venv <DIR_PATH>`).
Please note that a globally installed version of PySCIPOpt on your machine might lead to problems.

Furthermore, you need to have the Python
development files installed on your system (error message "Python.h not
found"):

    sudo apt-get install python-dev   # for Python 2, on Linux
    sudo apt-get install python3-dev  # for Python 3, on Linux

After setting up `SCIPOPTDIR` as specified above install pyscipopt

    export SCIPOPTDIR=/path/to/scip/install/dir
    python -m pip install .

For recompiling the source in the current directory `.` use

    python -m pip install --compile .

Building with debug information
===============================

To use debug information in PySCIPOpt you need to build it like this:

    python -m pip install --install-option="--debug" .

Be aware that you will need the **debug library** of the SCIP
Optimization Suite for this to work
(`cmake .. -DCMAKE_BUILD_TYPE=Debug`).

Testing new installation
========================

To test your brand-new installation of PySCIPOpt you need
[pytest](https://pytest.org/) on your system.

    pip install pytest

Here is the complete [installation
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


