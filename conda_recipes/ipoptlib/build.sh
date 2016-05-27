#!/bin/bash

# TODO: get any 3rd party code?

./configure --prefix=${PREFIX}
# BLAS from mkl?
# LAPACK from mkl?
# HSL?
# other libraries?

make
make install

