Requirements
============

PySCIPOpt uses the shared library of the [SCIP Optimization Suite](http://scip.zib.de/).
You need to have the library corresponding to your platform (Linux, Windows, OS X) available on your system.
If the library is not installed in the global path you need to specify its location using the environment variable `SCIPOPTDIR`:

 - on Linux and OS X:
    export SCIPOPTDIR=<absolut_path_to_directory>

 - on Windows:
    set SCIPOPTDIR=<absolut_path_to_directory>

`SCIPOPTDIR` needs to have a subdirectory `lib` that contains the library.

Additionally, if you're building PySCIPOpt from source, i.e. not using the precompiled egg or wheel, you also need to place all SCIP header files into a directory `include` next to `lib`:

    SCIPOPTDIR
    |- lib
        |- libscipopt.so ...
    |- include
        |- scip
        |- lpi
        |- nlpi
        |- ...



Installation from PyPI
======================

`pip install pyscipopt`

To be able to use PySCIPOpt with a library that is not located in the standard search path of your machine (`~/lib`, `/usr/local/lib`, etc.) you need to set it accordingly:

 - on Linux:
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCIPOPTDIR/lib

 - on Windows:
    set PATH=%PATH%;%SCIPOPTDIR\lib

 - on OS X:
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$SCIPOPTDIR/lib


Building everything form source
===============================

Please refer to installation instructions of the SCIP Optimization Suite for information on how to generate a shared library or download a precompiled one.

PySCIPOpt requires [Cython](http://cython.org/) to be installed in your system. If the Cython compilation fails, upgrade your Cython version (confirmed that version 0.20.1 fails). Furthermore, you need to have the Python development files installed on your system (error message "Python.h not found"):

    sudo apt-get install python-dev   # for Python 2
    sudo apt-get install python3-dev  # for Python 3

After setting up your environment variables as specified above, please execute the following command:

    python setup.py install

You may use the additional options `--user` or `--prefix=<custom-python-path>`, to build the interface locally.


Note:
-----

You cannot use the interface module from within the `PySCIPOpt` main directory. This is because Python will try to import the `pyscipopt` package from the local directory instead of using the installed one.
