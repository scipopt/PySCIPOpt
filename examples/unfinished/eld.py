"""
eld.py: economic load dispatching in electricity generation

Approach: use an SOS2 constraints for modeling non-linear functions.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict
import math
import random

from piecewise import convex_comb_sos

def cost(a,b,c,e,f,p_min,p):
    """cost: fuel cost based on "standard" parameters
    (with valve-point loading effect)
    """
    return a + b*p + c*p*p + abs(e*math.sin(f*(p_min-p)))

def lower_brkpts(a,b,c,e,f,p_min,p_max,n):
    """lower_brkpts: lower approximation of the cost function
    Parameters:
        - a,...,p_max: cost parameters
        - n: number of breakpoints' intervals to insert between valve points
    Returns: list of breakpoints in the form [(x0,y0),...,(xK,yK)]
    """
    EPS = 1.e-12        # for avoiding round-off errors
    if f == 0: f = math.pi/(p_max-p_min)
    brk = []
    nvalve = int(math.ceil(f*(p_max-p_min)/math.pi))
    for i in range(nvalve+1):
        p0 = p_min + i*math.pi/f
        if p0 >= p_max-EPS:
            brk.append((p_max,cost(a,b,c,e,f,p_min,p_max)))
            break
        for j in range(n):
            p = p0 + j*math.pi/f/n
            if p >= p_max:
                break
            brk.append((p,cost(a,b,c,e,f,p_min,p)))
    return brk


def eld_complete(U,p_min,p_max,d,brk):
    """eld -- economic load dispatching in electricity generation
    Parameters:
        - U: set of generators (units)
        - p_min[u]: minimum operating power for unit u
        - p_max[u]: maximum operating power for unit u
        - d: demand
        - brk[k]: (x,y) coordinates of breakpoint k, k=0,...,K
    Returns a model, ready to be solved.
    """

    model = Model("Economic load dispatching")

    p,F = {},{}
    for u in U:
        p[u] = model.addVar(lb=p_min[u], ub=p_max[u], name="p(%s)"%u)    # capacity
        F[u] = model.addVar(lb=0,name="fuel(%s)"%u)

    # set fuel costs based on piecewise linear approximation
    for u in U:
        abrk = [X for (X,Y) in brk[u]]
        bbrk = [Y for (X,Y) in brk[u]]

        # convex combination part:
        K = len(brk[u])-1
        z = {}
        for k in range(K+1):
            z[k] = model.addVar(ub=1) # do not name variables for avoiding clash

        model.addCons(p[u] == quicksum(abrk[k]*z[k] for k in range(K+1)))
        model.addCons(F[u] == quicksum(bbrk[k]*z[k] for k in range(K+1)))
        model.addCons(quicksum(z[k] for k in range(K+1)) == 1)
        model.addConsSOS2([z[k] for k in range(K+1)])

    # demand satisfaction
    model.addCons(quicksum(p[u] for u in U) == d, "demand")

    # objective
    model.setObjective(quicksum(F[u] for u in U), "minimize")

    model.data = p
    return model


def eld_another(U,p_min,p_max,d,brk):
    """eld -- economic load dispatching in electricity generation
    Parameters:
        - U: set of generators (units)
        - p_min[u]: minimum operating power for unit u
        - p_max[u]: maximum operating power for unit u
        - d: demand
        - brk[u][k]: (x,y) coordinates of breakpoint k, k=0,...,K for unit u
    Returns a model, ready to be solved.
    """
    model = Model("Economic load dispatching")

    # set objective based on piecewise linear approximation
    p,F,z = {},{},{}
    for u in U:
        abrk = [X for (X,Y) in brk[u]]
        bbrk = [Y for (X,Y) in brk[u]]
        p[u],F[u],z[u] = convex_comb_sos(model,abrk,bbrk)
        p[u].lb = p_min[u]
        p[u].ub = p_max[u]

    # demand satisfaction
    model.addCons(quicksum(p[u] for u in U) == d, "demand")

    # objective
    model.setObjective(quicksum(F[u] for u in U), "minimize")

    model.data = p
    return model


def eld13():
    U,      a,          b,      c,      e,      f,      p_min,  p_max = multidict({
    1   : [ 550,    8.1,    0.00028,    300,    0.035,  0,      680 ],
    2   : [ 309,    8.1,    0.00056,    200,    0.042,  0,      360 ],
    3   : [ 307,    8.1,    0.00056,    200,    0.042,  0,      360 ],
    4   : [ 240,    7.74,   0.00324,    150,    0.063,  60,     180 ],
    5   : [ 240,    7.74,   0.00324,    150,    0.063,  60,     180 ],
    6   : [ 240,    7.74,   0.00324,    150,    0.063,  60,     180 ],
    7   : [ 240,    7.74,   0.00324,    150,    0.063,  60,     180 ],
    8   : [ 240,    7.74,   0.00324,    150,    0.063,  60,     180 ],
    9   : [ 240,    7.74,   0.00324,    150,    0.063,  60,     180 ],
    10  : [ 126,    8.6,    0.00284,    100,    0.084,  40,     120 ],
    11  : [ 126,    8.6,    0.00284,    100,    0.084,  40,     120 ],
    12  : [ 126,    8.6,    0.00284,    100,    0.084,  55,     120 ],
    13  : [ 126,    8.6,    0.00284,    100,    0.084,  55,     120 ],
    })
    return U, a, b, c, e, f, p_min, p_max


def eld40():
    U,      a,          b,      c,      e,      f,      p_min,  p_max = multidict({
    1  : [ 94.705,  6.73,  0.00690, 100,    0.084,      36,       114],
    2  : [ 94.705,  6.73,  0.00690, 100,    0.084,      36,       114],
    3  : [ 309.54,  7.07,  0.02028, 100,    0.084,      60,       120],
    4  : [ 369.03,  8.18,  0.00942, 150,    0.063,      80,       190],
    5  : [ 148.89,  5.35,  0.01140, 120,    0.077,      47,       97],
    6  : [ 222.33,  8.05,  0.01142, 100,    0.084,      68,       140],
    7  : [ 287.71,  8.03,  0.00357, 200,    0.042,      110,      300],
    8  : [ 391.98,  6.99,  0.00492, 200,    0.042,      135,      300],
    9  : [ 455.76,  6.60,  0.00573, 200,    0.042,      135,      300],
    10 : [ 722.82,  12.9,  0.00605, 200,    0.042,      130,      300],
    11 : [ 635.20,  12.9,  0.00515, 200,    0.042,      94,       375],
    12 : [ 654.69,  12.8,  0.00569, 200,    0.042,      94,       375],
    13 : [ 913.40,  12.5,  0.00421, 300,    0.035,      125,      500],
    14 : [ 1760.4,  8.84,  0.00752, 300,    0.035,      125,      500],
    15 : [ 1728.3,  9.15,  0.00708, 300,    0.035,      125,      500],
    16 : [ 1728.3,  9.15,  0.00708, 300,    0.035,      125,      500],
    17 : [ 647.85,  7.97,  0.00313, 300,    0.035,      220,      500],
    18 : [ 649.69,  7.95,  0.00313, 300,    0.035,      220,      500],
    19 : [ 647.83,  7.97,  0.00313, 300,    0.035,      242,      550],
    20 : [ 647.81,  7.97,  0.00313, 300,    0.035,      242,      550],
    21 : [ 785.96,  6.63,  0.00298, 300,    0.035,      254,      550],
    22 : [ 785.96,  6.63,  0.00298, 300,    0.035,      254,      550],
    23 : [ 794.53,  6.66,  0.00284, 300,    0.035,      254,      550],
    24 : [ 794.53,  6.66,  0.00284, 300,    0.035,      254,      550],
    25 : [ 801.32,  7.10,  0.00277, 300,    0.035,      254,      550],
    26 : [ 801.32,  7.10,  0.00277, 300,    0.035,      254,      550],
    27 : [ 1055.1,  3.33,  0.52124, 120,    0.077,      10,       150],
    28 : [ 1055.1,  3.33,  0.52124, 120,    0.077,      10,       150],
    29 : [ 1055.1,  3.33,  0.52124, 120,    0.077,      10,       150],
    30 : [ 148.89,  5.35,  0.01140, 120,    0.077,      47,       97],
    31 : [ 222.92,  6.43,  0.00160, 150,    0.063,      60,       190],
    32 : [ 222.92,  6.43,  0.00160, 150,    0.063,      60,       190],
    33 : [ 222.92,  6.43,  0.00160, 150,    0.063,      60,       190],
    34 : [ 107.87,  8.95,  0.00010, 200,    0.042,      90,       200],
    35 : [ 116.58,  8.62,  0.00010, 200,    0.042,      90,       200],
    36 : [ 116.58,  8.62,  0.00010, 200,    0.042,      90,       200],
    37 : [ 307.45,  5.88,  0.01610, 80,     0.098,      25,       110],
    38 : [ 307.45,  5.88,  0.01610, 80,     0.098,      25,       110],
    39 : [ 307.45,  5.88,  0.01610, 80,     0.098,      25,       110],
    40 : [ 647.83,  7.97,  0.00313, 300,    0.035,      242,      550],
    })

    U = list(a.keys())
    return U,a,b,c,e,f,p_min,p_max,d



if __name__ == "__main__":
    # U,a,b,c,e,f,p_min,p_max = eld13(); d=1800
    U,a,b,c,e,f,p_min,p_max = eld13(); d=2520
    # U,a,b,c,e,f,p_min,p_max = eld40(); d=10500
    n = 100     # number of breakpoints between valve points
    brk = {}
    for u in U:
        brk[u] = lower_brkpts(a[u],b[u],c[u],e[u],f[u],p_min[u],p_max[u],n)

    lower = eld_complete(U,p_min,p_max,d,brk)
    # lower = eld_another(U,p_min,p_max,d,brk)

    lower.setRealParam("limits/gap", 1e-12)
    lower.setRealParam("limits/absgap", 1e-12)
    lower.setRealParam("numerics/feastol", 1e-9)

    lower.optimize()
    p = lower.data
    print("Lower bound:",lower.ObjBound)
    UB = sum(cost(a[u],b[u],c[u],e[u],f[u],p_min[u],lower.getVal(p[u])) for u in U)
    print("Upper bound:",UB)
    print("Solution:")
    for u in p:
        print(u,lower.getVal(p[u]))
