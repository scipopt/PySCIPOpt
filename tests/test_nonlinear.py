import pytest
import random

from pyscipopt import Model, quicksum, sqrt, exp, log, sin

# test string with polynomial formulation (uses only Expr)
def test_string_poly():
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

    assert abs(m.getPrimalbound() - 1.6924910128) < 1.0e-3

# test string with original formulation (uses GenExpr)
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

    m.addCons(coil / wire - const1 == 0, name='defconst1')

    m.addCons((4.0*const1 - 1.0) / (4.0*const1 - 4.0) + 0.615 / const1 - const2 == 0, name='defconst2')

    m.addCons(8.0*MAXWORKLOAD/PI*const1*const2 - MAXSHEARSTRESS*wire**2 <= 0.0, name='shear')

    m.addCons(8.0/SHEARMOD*ncoils*const1**3 / wire - defl == 0.0, name="defdefl")

    m.addCons(MAXWORKLOAD*defl + 1.05*ncoils*wire + 2.1*wire <= MAXFREELEN, name='freel')

    m.addCons(coil + wire <= MAXCOILDIAM, name='coilwidth')

    m.addCons(quicksum(c*v for (c,v) in zip(DIAMETERS, y)) - wire == 0, name='defwire')

    m.addCons(quicksum(y) == 1, name='selectwire')

    m.optimize()

    assert abs(m.getPrimalbound() - 1.6924910128) < 1.0e-3

@pytest.mark.skip(reason="Test fails on CPython3.6 for MacOS with x86_64")
# test circle: find circle of smallest radius that encloses the given points
def test_circle():
    points =[
            (2.802686, 1.398947),
            (4.719673, 4.792101),
            (1.407758, 7.769566),
            (2.253320, 2.373641),
            (8.583144, 9.769102),
            (3.022725, 5.470335),
            (5.791380, 1.214782),
            (8.304504, 8.196392),
            (9.812677, 5.284600),
            (9.445761, 9.541600)]

    m = Model()
    a = m.addVar('a', lb=None)
    b = m.addVar('b', ub=None)
    r = m.addVar('r')

    # minimize radius
    m.setObjective(r, 'minimize')

    for i,p in enumerate(points):
        # NOTE: SCIP will not identify this as SOC constraints!
        m.addCons( sqrt((a - p[0])**2 + (b - p[1])**2) <= r, name = 'point_%d'%i)

    m.optimize()

    bestsol = m.getBestSol()
    assert abs(m.getSolVal(bestsol, r) - 5.2543) < 1.0e-2
    assert abs(m.getSolVal(bestsol, a) - 6.1230) < 1.0e-2
    assert abs(m.getSolVal(bestsol, b) - 5.4713) < 1.0e-2

# test gastrans: see example in <scip path>/examples/CallableLibrary/src/gastrans.c
# of course there is a more pythonic/elegant way of implementing this, probably
# starting by using a proper graph structure
def test_gastrans():
    GASTEMP = 281.15
    RUGOSITY = 0.05
    DENSITY = 0.616
    COMPRESSIBILITY = 0.8
    nodes = [
            #   name          supplylo   supplyup pressurelo pressureup   cost
            ("Anderlues",          0.0,       1.2,       0.0,      66.2,   0.0  ),  #  0
            ("Antwerpen",         None,    -4.034,      30.0,      80.0,   0.0  ),  #  1
            ("Arlon",             None,    -0.222,       0.0,      66.2,   0.0  ),  #  2
            ("Berneau",            0.0,       0.0,       0.0,      66.2,   0.0  ),  #  3
            ("Blaregnies",        None,   -15.616,      50.0,      66.2,   0.0  ),  #  4
            ("Brugge",            None,    -3.918,      30.0,      80.0,   0.0  ),  #  5
            ("Dudzele",            0.0,       8.4,       0.0,      77.0,   2.28 ),  #  6
            ("Gent",              None,    -5.256,      30.0,      80.0,   0.0  ),  #  7
            ("Liege",             None,    -6.385,      30.0,      66.2,   0.0  ),  #  8
            ("Loenhout",           0.0,       4.8,       0.0,      77.0,   2.28 ),  #  9
            ("Mons",              None,    -6.848,       0.0,      66.2,   0.0  ),  # 10
            ("Namur",             None,    -2.120,       0.0,      66.2,   0.0  ),  # 11
            ("Petange",           None,    -1.919,      25.0,      66.2,   0.0  ),  # 12
            ("Peronnes",           0.0,      0.96,       0.0,      66.2,   1.68 ),  # 13
            ("Sinsin",             0.0,       0.0,       0.0,      63.0,   0.0  ),  # 14
            ("Voeren",          20.344,    22.012,      50.0,      66.2,   1.68 ),  # 15
            ("Wanze",              0.0,       0.0,       0.0,      66.2,   0.0  ),  # 16
            ("Warnand",            0.0,       0.0,       0.0,      66.2,   0.0  ),  # 17
            ("Zeebrugge",         8.87,    11.594,       0.0,      77.0,   2.28 ),  # 18
            ("Zomergem",           0.0,       0.0,       0.0,      80.0,   0.0  )   # 19
            ]
    arcs = [
            # node1  node2  diameter length active */
            (   18,     6,    890.0,   4.0, False ),
            (   18,     6,    890.0,   4.0, False ),
            (    6,     5,    890.0,   6.0, False ),
            (    6,     5,    890.0,   6.0, False ),
            (    5,    19,    890.0,  26.0, False ),
            (    9,     1,    590.1,  43.0, False ),
            (    1,     7,    590.1,  29.0, False ),
            (    7,    19,    590.1,  19.0, False ),
            (   19,    13,    890.0,  55.0, False ),
            (   15,     3,    890.0,   5.0,  True ),
            (   15,     3,    395.0,   5.0,  True ),
            (    3,     8,    890.0,  20.0, False ),
            (    3,     8,    395.0,  20.0, False ),
            (    8,    17,    890.0,  25.0, False ),
            (    8,    17,    395.0,  25.0, False ),
            (   17,    11,    890.0,  42.0, False ),
            (   11,     0,    890.0,  40.0, False ),
            (    0,    13,    890.0,   5.0, False ),
            (   13,    10,    890.0,  10.0, False ),
            (   10,     4,    890.0,  25.0, False ),
            (   17,    16,    395.5,  10.5, False ),
            (   16,    14,    315.5,  26.0,  True ),
            (   14,     2,    315.5,  98.0, False ),
            (    2,    12,    315.5,   6.0, False )
            ]

    scip = Model()

    # create flow variables
    flow = {}
    for arc in arcs:
        flow[arc] = scip.addVar("flow_%s_%s"%(nodes[arc[0]][0],nodes[arc[1]][0]), # names of nodes in arc
                lb = 0.0 if arc[4] else None) # no lower bound if not active

    # pressure difference variables
    pressurediff = {}
    for arc in arcs:
        pressurediff[arc] = scip.addVar("pressurediff_%s_%s"%(nodes[arc[0]][0],nodes[arc[1]][0]), # names of nodes in arc
                lb = None)

    # supply variables
    supply = {}
    for node in nodes:
        supply[node] = scip.addVar("supply_%s"%(node[0]), lb = node[1], ub = node[2], obj = node[5])

    # square pressure variables
    pressure = {}
    for node in nodes:
        pressure[node] = scip.addVar("pressure_%s"%(node[0]), lb = node[3]**2, ub = node[4]**2)


    # node balance constrains, for each node i: outflows - inflows = supply
    for nid, node in enumerate(nodes):
        # find arcs that go or end at this node
        flowbalance = 0
        for arc in arcs:
            if arc[0] == nid: # arc is outgoing
                flowbalance += flow[arc]
            elif arc[1] == nid: # arc is incoming
                flowbalance -= flow[arc]
            else:
                continue

        scip.addCons(flowbalance == supply[node], name="flowbalance%s"%node[0])

    # pressure difference constraints: pressurediff[node1 to node2] = pressure[node1] - pressure[node2]
    for arc in arcs:
        scip.addCons(pressurediff[arc] == pressure[nodes[arc[0]]] - pressure[nodes[arc[1]]], "pressurediffcons_%s_%s"%(nodes[arc[0]][0],nodes[arc[1]][0]))

    # pressure loss constraints:
    # active arc: flow[arc]^2 + coef * pressurediff[arc] <= 0.0
    # regular pipes: flow[arc] * abs(flow[arc]) - coef * pressurediff[arc] == 0.0
    # coef = 96.074830e-15*diameter(i)^5/(lambda*compressibility*temperatur*length(i)*density)
    # lambda = (2*log10(3.7*diameter(i)/rugosity))^(-2)
    from math import log10
    for arc in arcs:
        coef = 96.074830e-15 * arc[2]**5 * (2.0*log10(3.7*arc[2]/RUGOSITY))**2 / COMPRESSIBILITY / GASTEMP / arc[3] / DENSITY
        if arc[4]: # active
            scip.addCons(flow[arc]**2 + coef * pressurediff[arc] <= 0.0, "pressureloss_%s_%s"%(nodes[arc[0]][0],nodes[arc[1]][0]))
        else:
            scip.addCons(flow[arc]*abs(flow[arc]) - coef * pressurediff[arc] == 0.0, "pressureloss_%s_%s"%(nodes[arc[0]][0],nodes[arc[1]][0]))

    scip.setRealParam('limits/time', 5)
    scip.optimize()

    if scip.getStatus() == 'timelimit':
        pytest.skip()

    assert abs(scip.getPrimalbound() - 89.08584) < 1.0e-5

def test_quad_coeffs():
    """test coefficient access method for quadratic constraints"""
    scip = Model()
    x = scip.addVar()
    y = scip.addVar()
    z = scip.addVar()

    c = scip.addCons(2*x*y + 0.5*x**2 + 4*z >= 10)
    assert c.isNonlinear()
    assert scip.checkQuadraticNonlinear(c)

    bilinterms, quadterms, linterms = scip.getTermsQuadratic(c)

    assert bilinterms[0][0].name == x.name
    assert bilinterms[0][1].name == y.name
    assert bilinterms[0][2] == 2

    assert quadterms[0][0].name == x.name
    assert quadterms[0][1] == 0.5

    assert linterms[0][0].name == z.name
    assert linterms[0][1] == 4

def test_addExprNonLinear():
    m = Model()
    x = m.addVar("x", lb=0, ub=1, obj=10)
    y = m.addVar("y", obj=1)
    z = m.addVar("z", obj=1)

    c = m.addCons(x**2 >= 9)
    c1 = m.addCons(x**3 >= 4)
    m.addExprNonlinear(c, y**2, 2)
    m.addExprNonlinear(c1, z**(1/3), 1)

    m.setParam("numerics/epsilon", 10**(-5)) # bigger eps due to nonlinearities
    m.optimize()

    assert m.getNSols() > 0
    assert m.isEQ(m.getVal(y), 2)
    assert m.isEQ(m.getVal(z), 27)