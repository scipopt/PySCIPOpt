#!/bin/bash

VERSION=3.2.1
OSTYPE=linux
ARCH=x86_64
COMP=gnu
IPOPTOPT=opt

SCIPDIR=scip-$VERSION

# manually create symlinks to IPOPT "installation directory"
tar xf ${SCIPDIR}.tgz
mkdir -p ${SCIPDIR}/lib
ln -s ${PREFIX} ${SCIPDIR}/lib/ipopt.${OSTYPE}.${ARCH}.${COMP}.${IPOPTOPT}

# build the shared library only
make scipoptlib SHARED=true IPOPT=true GMP=false READLINE=false

# "install" the shared library
cp lib/libscipopt-*.so ${PREFIX}/lib/libscipopt.so

# "install" the C headers of SCIP
DIRECTORIES=(blockmemshell lpi nlpi scip)
for D in ${DIRECTORIES[@]}
do
    mkdir -p ${PREFIX}/include/${D}
    cp ${SCIPDIR}/src/${D}/*.h ${PREFIX}/include/${D}
done
