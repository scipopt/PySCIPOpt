#!/bin/bash

# generate html documentation in docs/html
doxygen docs/doxy

# fix broken links to SCIP online documentation
sed -i "s/\.php\.html/\.php/g" docs/html/*.html
