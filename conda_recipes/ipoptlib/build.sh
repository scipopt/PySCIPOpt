#!/bin/bash

# get some 3rd party code
THIRDPARTY=(Blas Lapack Metis Mumps)
for TP in ${THIRDPARTY[@]}
do
    cd ThirdParty/${TP} && ./get.${TP} && cd ../../
done

./configure --prefix=${PREFIX}
# no HSL!

make
make install
