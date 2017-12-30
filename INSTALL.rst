Requirements
============

PySCIPOpt uses the shared library of the `SCIP Optimization
Suite <http://scip.zib.de/>`__. You need to have the library
corresponding to your platform (Linux, Windows, OS X) available on your
system. If the library is not installed in the global path you need to
specify its location using the environment variable ``SCIPOPTDIR``:

-  | on Linux and OS X:
   | ``export SCIPOPTDIR=<path_to_install_dir>``

-  | on Windows:
   | ``set SCIPOPTDIR=<path_to_install_dir>``

``SCIPOPTDIR`` needs to have a subdirectory ``lib`` that contains the
library ``libscip.so`` (for Linux).

Please `install the SCIP Optimization Suite using CMake <http://scip.zib.de/doc/html/CMAKE.php>`__ if you're building
it from source.

::

    SCIPOPTDIR
      > lib
        > libscip.so ...
      > include
        > scip
        > lpi
        > nlpi
        > ...

Installation from PyPI
======================

``pip install pyscipopt``

On Windows you may need to ensure that the ``scip`` library can be
found at runtime by adjusting your ``PATH`` environment variable:

-  on Windows:
   ``set PATH=%PATH%;%SCIPOPTDIR%\bin``

On Linux and OS X this is encoded in the generated PySCIPOpt library and
therefore not necessary.

Building everything form source
===============================

Please refer to `installation
instructions <http://scip.zib.de/doc/html/CMAKE.php>`__ of the SCIP
Optimization Suite for information on how to generate a shared library
or download a precompiled one.

PySCIPOpt requires `Cython <http://cython.org/>`__ to be installed in
your system. If the Cython compilation fails, upgrade your Cython
version (confirmed that version 0.20.1 fails). Furthermore, you need to
have the Python development files installed on your system (error
message "Python.h not found"):

::

    sudo apt-get install python-dev   # for Python 2
    sudo apt-get install python3-dev  # for Python 3

After setting up your environment variables as specified above, please
execute the following command:

::

    python setup.py install

You may use the additional options ``--user`` or
``--prefix=<custom-python-path>``, to build the interface locally.

Building with debug information
===============================

To use debug mode in PySCIPOpt you need to build it like this:

::

    python setup.py install --debug

Be aware that you will need the *debug library* of the SCIP Optimization
Suite for this to work.
