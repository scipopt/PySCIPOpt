from pyscipopt import Model
import pytest
import os

# This test requires a directory link in tests/ to check/ in the main SCIP directory.

testset = []
primalsolutions = {}
dualsolutions = {}
tolerance = 1e-5
infinity = 1e20

testsetpath = 'check/testset/short.test'
solufilepath = 'check/testset/short.solu'

if not all(os.path.isfile(fn) for fn in [testsetpath, solufilepath]):
    if pytest.__version__ < "3.0.0":
        pytest.skip("Files for testset `short` not found (symlink missing?)")
    else:
        pytestmark = pytest.mark.skip

else:
    with open(testsetpath, 'r') as f:
        for line in f.readlines():
            testset.append('check/' + line.rstrip('\n'))

    with open(solufilepath, 'r') as f:
        for line in f.readlines():

            if len(line.split()) == 2:
                [s, name] = line.split()
            else:
                [s, name, value] = line.split()

            if   s == '=opt=':
                primalsolutions[name] = float(value)
                dualsolutions[name] = float(value)
            elif s == '=inf=':
                primalsolutions[name] = infinity
                dualsolutions[name] = infinity
            elif s == '=best=':
                primalsolutions[name] = float(value)
            elif s == '=best dual=':
                dualsolutions[name] = float(value)
            # status =unkn= needs no data

def relGE(v1, v2, tol = tolerance):
    if v1 is None or v2 is None:
        return True
    else:
        reltol = tol * max(abs(v1), abs(v2), 1.0)
        return (v1 - v2) >= -reltol

def relLE(v1, v2, tol = tolerance):
    if v1 is None or v2 is None:
        return True
    else:
        reltol = tol * max(abs(v1), abs(v2), 1.0)
        return (v1 - v2) <= reltol


@pytest.mark.parametrize('instance', testset)
def test_instance(instance):
    s = Model()
    s.hideOutput()
    s.readProblem(instance)
    s.optimize()
    name = os.path.split(instance)[1]
    if name.rsplit('.',1)[1].lower() == 'gz':
        name = name.rsplit('.',2)[0]
    else:
        name = name.rsplit('.',1)[0]

    # we do not need the solution status
    primalbound = s.getObjVal()
    dualbound = s.getDualbound()

    # get solution data from solu file
    primalsolu = primalsolutions.get(name, None)
    dualsolu = dualsolutions.get(name, None)

    if s.getObjectiveSense() == 'minimize':
        assert relGE(primalbound, dualsolu)
        assert relLE(dualbound, primalsolu)
    else:
        if( primalsolu == infinity ): primalsolu = -infinity
        if( dualsolu == infinity ): dualsolu = -infinity
        assert relLE(primalbound, dualsolu)
        assert relGE(dualbound, primalsolu)
