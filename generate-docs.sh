#!/bin/bash

#get SCIP TAGFILE
wget -O docs/scip.tag https://scip.zib.de/doc/scip.tag

# generate html documentation in docs/html
doxygen docs/doxy

# fix broken links to SCIP online documentation
sed -i "s/\.php\.html/\.php/g" docs/html/*.html
