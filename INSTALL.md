Requirements
============

PySCIPOpt uses the shared library of the [SCIP Optimization Suite](http://scip.zib.de/).
You need to have the library corresponding to your platform (Linux, Windows, OS X) available on your system.
If the library is not installed in the global path you need to specify its location using the environment variable `SCIPOPTDIR`:

 - on Linux and OS X:  
    `export SCIPOPTDIR=<path_to_install_dir>`

 - on Windows:  
    `set SCIPOPTDIR=<path_to_install_dir>`

`SCIPOPTDIR` needs to have a subdirectory `lib` that contains the library.

Additionally, if you're building PySCIPOpt from source, i.e. not using the precompiled egg or wheel, you also need to place all SCIP header files into a directory `include` next to `lib` (this is done automatically by `make install INSTALLDIR=$SCIPOPTDIR SHARED=true` of the SCIP Optimization Suite):

    SCIPOPTDIR
      > lib
        > libscipopt.so ...
      > include
        > scip
        > lpi
        > nlpi
        > ...



Installation from PyPI
======================

`pip install pyscipopt`

On Windows you need to make sure that the `scipopt` library can be found at runtime by adjusting your `PATH` environment variable:

 - on Windows:  
    `set PATH=%PATH%;%SCIPOPTDIR%\lib`

On Linux and OS X this is encoded in the generated PySCIPOpt library and therefore not necessary.



Building everything form source
===============================

Please refer to [installation instructions](http://scip.zib.de/doc/html/MAKE.php) of the SCIP Optimization Suite for information on how to generate a shared library or download a precompiled one.

PySCIPOpt requires [Cython](http://cython.org/) to be installed in your system. If the Cython compilation fails, upgrade your Cython version (confirmed that version 0.20.1 fails). Furthermore, you need to have the Python development files installed on your system (error message "Python.h not found"):

    sudo apt-get install python-dev   # for Python 2
    sudo apt-get install python3-dev  # for Python 3

After setting up your environment variables as specified above, please execute the following command:

    python setup.py install

You may use the additional options `--user` or `--prefix=<custom-python-path>`, to build the interface locally.
