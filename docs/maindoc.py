##@file maindoc.py 
#@brief Main documentation page

## @mainpage Overview
#
# This project provides an interface from Python to the [SCIP Optimization Suite](http://scip.zib.de). <br>
#
# See the [web site] (https://github.com/SCIP-Interfaces/PySCIPOpt) to download PySCIPOpt.
#
# @section Changelog
# See [CHANGELOG.md](CHANGELOG.md) for added, removed or fixed functionality.
#
# @section Installation
# See [INSTALL.md](INSTALL.md) for instructions.
#
# @section TABLEOFCONTENTS Structure of this manual
# 
# This documentation gives an introduction to the functionality of the Python interface of the SCIP code in the following chapters
#
# - \ref pyscipopt::scip::Model "Model" Class with the most fundamental functions to create and solve a problem
# - \ref examples/tutorial "Tutorials" and \ref examples/finished "Examples" to display some functionality of the interface
# - @subpage EXTEND Explanations on extending the PySCIPOpt interface
# 
# For a more detailed description on how to create a model and how to extend the interface, please have a look at the [README.md] (README.md).
#

##@page EXTEND Extending the interface
# PySCIPOpt already covers many of the SCIP callable library methods. You
#may also extend it to increase the functionality of this interface. The
#following will provide some directions on how this can be achieved:
#
#The two most important files in PySCIPOpt are the `scip.pxd` and
#`scip.pyx`. These two files specify the public functions of SCIP that
#can be accessed from your python code.
#
#To make PySCIPOpt aware of the public functions you would like to
#access, you must add them to `scip.pxd`. There are two things that must
#be done in order to properly add the functions:
#
# -# Ensure any `enum`s, `struct`s or SCIP variable types are included in
#    `scip.pxd`
# -# Add the prototype of the public function you wish to access to
#    `scip.pxd`
#
#After following the previous two steps, it is then possible to create
#functions in python that reference the SCIP public functions included in
#`scip.pxd`. This is achieved by modifying the `scip.pyx` file to add the
#functionality you require.
#
#We are always happy to accept pull request containing patches or
#extensions!
#
#Please have a look at our [contribution guidelines](CONTRIBUTING.md).
