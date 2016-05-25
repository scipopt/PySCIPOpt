VERSION=3.2.1
SCIPDIR=scip-$VERSION

# build the shared library only
make scipoptlib SHARED=true GMP=false READLINE=false

# "install" the shared library
cp lib/libscipopt-*.so ${PREFIX}/lib/libscipopt.so

# "install" the C headers of SCIP
DIRECTORIES=(blockmemshell lpi nlpi scip)
for D in ${DIRECTORIES[@]}
do
    mkdir -p ${PREFIX}/include/${D}
    cp ${SCIPDIR}/src/${D}/*.h ${PREFIX}/include/${D}
done
