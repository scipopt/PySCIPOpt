#!/bin/bash

# get some 3rd party code
cd ThirdParty/Metis && ./get.Metis && cd ../../
cd ThirdParty/Mumps && ./get.Mumps && cd ../../

./configure --prefix=${PREFIX} \
            --with-blas="-L${PREFIX} -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl" \
            --with-lapack="-L${PREFIX} -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl"
# no HSL!

make
make install
