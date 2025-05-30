##################
Installation Guide
##################

This page will detail all methods for installing PySCIPOpt via package managers,
which come with their own versions of SCIP. For building PySCIPOpt against your
own custom version of SCIP, or for building PySCIPOpt from source, visit :doc:`this page </build>`.

.. contents:: Contents


PyPI (pip)
============

Pre-built binary wheels are uploaded to PyPI (Python Package Index) for each release.
Supported platforms are Linux (x86_64), Windows (x86_64) and MacOS (x86_64, Apple Silicon).

To install PySCIPOpt simply run the command:

.. code-block:: bash

  pip install pyscipopt

To avoid interfering with system packages, it's best to use a `virtual environment <https://docs.python.org/3/library/venv.html>`.

.. warning::

  Using a virtual environment is **mandatory** in some newer Python configurations
  to avoid permission and package conflicts.

.. code-block:: bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install pyscipopt

Remember to activate the environment (``source venv/bin/activate``) in each terminal session where you use PySCIPOpt.

.. note:: For Linux users: PySCIPOpt versions newer than 5.1.1 installed via PyPI now require glibc 2.28+

  For our build infrastructure we use `manylinux <https://github.com/pypa/manylinux>`_.
  As CentOS 7 is no longer supported, we have migrated from ``manylinux2014`` to ``manylinux_2_28``.

  TLDR: Older linux distributions may not work for newer versions of PySCIPOpt installed via pip.

.. note:: For Mac users: PySCIPOpt versions newer than 5.1.1 installed via PyPI now only support
  MACOSX 13+ for users running x86_64 architecture, and MACOSX 14+ for users running newer Apple silicon.

.. note:: For versions older than 4.4.0 installed via PyPI SCIP is not automatically installed.
  This means that SCIP must be installed yourself. If it is not installed globally,
  then the ``SCIPOPTDIR`` environment flag must be set, see :doc:`this page </build>` for more details.

.. note:: Some Mac configurations require adding the library installation path to `DYLD_LIBRARY_PATH` when
  using a locally installed version of SCIP.

Conda
=====

It is also possible to use the Conda package manager to install PySCIPOpt.
Conda will install SCIP automatically, hence everything can be installed in a single command:

.. code-block:: bash

  conda install --channel conda-forge pyscipopt

.. note:: Do not use the Conda base environment to install PySCIPOpt.

