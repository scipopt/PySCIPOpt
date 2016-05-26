#!/bin/bash

# we need to prepare the links to the dependency of `scipoptlib`
mkdir lib

# create symbolic link to shared library
ln -s ${PREFIX}/lib/libscipopt.so lib/libscipopt.so

# create symbolic links to headers
ln -s ${PREFIX}/include lib/scip-src

# using conda-specific setup.py to get rid of some dependencies
${PYTHON} conda_setup.py install || exit 1;
