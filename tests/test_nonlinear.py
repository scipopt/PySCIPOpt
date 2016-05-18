import pytest

from pyscipopt import Model, quicksum
from pyscipopt.scip import Expr, ExprCons

def test_string():
    PI = 3.141592653589793238462643
    NWIRES = 11
    DIAMETERS = [0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500]
    PRELOAD = 300.0
    MAXWORKLOAD = 1000.0
    MAXDEFLECT = 6.0
    DEFLECTPRELOAD = 1.25
    MAXFREELEN = 14.0
    MAXCOILDIAM = 3.0
    MAXSHEARSTRESS = 189000.0
    SHEARMOD = 11500000.0

    m = Model()
    coil = m.addVar('coildiam')
    wire = m.addVar('wirediam')
    defl = m.addVar('deflection', lb=DEFLECTPRELOAD / (MAXWORKLOAD - PRELOAD), ub=MAXDEFLECT / PRELOAD)
    ncoils = m.addVar('ncoils', vtype='I')
    const1 = m.addVar('const1')
    const2 = m.addVar('const2')
    volume = m.addVar('volume')
    y = [m.addVar('wire%d' % i, vtype='B') for i in range(NWIRES)]

    obj = 1.0 * volume
    m.setObjective(obj, 'minimize')

    m.addCons(PI/2*(ncoils + 2)*coil*wire**2 - volume == 0, name='voldef')

    # defconst1: coil / wire - const1 == 0.0
    m.addCons(coil - const1*wire == 0, name='defconst1')

    # defconst2: (4.0*const1 - 1.0) / (4.0*const1 - 4.0) + 0.615 / const1 - const2 == 0.0
    d1 = (4.0*const1 - 4.0)
    d2 = const1
    m.addCons((4.0*const1 - 1.0)*d2 + 0.615*d1 - const2*d1*d2 == 0, name='defconst2')

    m.addCons(8.0*MAXWORKLOAD/PI*const1*const2 - MAXSHEARSTRESS*wire**2 <= 0.0, name='shear')

    # defdefl: 8.0/shearmod * ncoils * const1^3 / wire - defl == 0.0
    m.addCons(8.0/SHEARMOD*ncoils*const1**3 - defl*wire == 0.0, name="defdefl")

    m.addCons(MAXWORKLOAD*defl + 1.05*ncoils*wire + 2.1*wire <= MAXFREELEN, name='freel')

    m.addCons(coil + wire <= MAXCOILDIAM, name='coilwidth')

    m.addCons(quicksum(c*v for (c,v) in zip(DIAMETERS, y)) - wire == 0, name='defwire')

    m.addCons(quicksum(y) == 1, name='selectwire')

    m.optimize()

if __name__ == '__main__':
    test_string()
