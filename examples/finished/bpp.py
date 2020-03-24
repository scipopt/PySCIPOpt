##@file bpp.py 
#@brief use SCIP for solving the bin packing problem.
"""
The instance of the bin packing problem is represented by the two
lists of n items of sizes and quantity s=(s_i).
The bin size is B.

We use Martello and Toth (1990) formulation, and suggest
extensions with tie-breaking and SOS constraints.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from pyscipopt import Model, quicksum

def FFD(s,B):
    """First Fit Decreasing heuristics for the Bin Packing Problem.
    Parameters:
        - s: list with item widths
        - B: bin capacity
    Returns a list of lists with bin compositions.
    """
    remain = [B]        # keep list of empty space per bin
    sol = [[]]          # a list ot items (i.e., sizes) on each used bin
    for item in sorted(s,reverse=True):
        for (j,free) in enumerate(remain):
            if free >= item:
                remain[j] -= item
                sol[j].append(item)
                break
        else: #does not fit in any bin
            sol.append([item])
            remain.append(B-item)
    return sol


def bpp(s,B):
    """bpp: Martello and Toth's model to solve the bin packing problem.
    Parameters:
        - s: list with item widths
        - B: bin capacity
    Returns a model, ready to be solved.
    """
    n = len(s)
    U = len(FFD(s,B)) # upper bound of the number of bins
    model = Model("bpp")
    # setParam("MIPFocus",1)
    x,y = {},{}
    for i in range(n):
        for j in range(U):
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
    for j in range(U):
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)

    # assignment constraints
    for i in range(n):
        model.addCons(quicksum(x[i,j] for j in range(U)) == 1, "Assign(%s)"%i)

    # bin capacity constraints
    for j in range(U):
        model.addCons(quicksum(s[i]*x[i,j] for i in range(n)) <= B*y[j], "Capac(%s)"%j)

    # tighten assignment constraints
    for j in range(U):
        for i in range(n):
            model.addCons(x[i,j] <= y[j], "Strong(%s,%s)"%(i,j))

    # tie breaking constraints
    for j in range(U-1):
        model.addCons(y[j] >= y[j+1],"TieBrk(%s)"%j)

    # SOS constraints
    for i in range(n):
        model.addConsSOS1([x[i,j] for j in range(U)])

    model.setObjective(quicksum(y[j] for j in range(U)), "minimize")
    model.data = x,y

    return model


def solveBinPacking(s,B):
    """solveBinPacking: use an IP model to solve the in Packing Problem.

    Parameters:
        - s: list with item widths
        - B: bin capacity

    Returns a solution: list of lists, each of which with the items in a roll.
    """
    n = len(s)
    U = len(FFD(s,B)) # upper bound of the number of bins
    model = bpp(s,B)
    x,y = model.data

    model.optimize()

    bins = [[] for i in range(U)]
    for (i,j) in x:
        if model.getVal(x[i,j]) > .5:
            bins[j].append(s[i])
    for i in range(bins.count([])):
        bins.remove([])
    for b in bins:
        b.sort()
    bins.sort()
    return bins


import random
def DiscreteUniform(n=10,LB=1,UB=99,B=100):
    """DiscreteUniform: create random, uniform instance for the bin packing problem."""
    B = 100
    s = [0]*n
    for i in range(n):
        s[i] = random.randint(LB,UB)
    return s,B


if __name__ == "__main__":
    random.seed(256)
    s,B = DiscreteUniform()
    print("items:", s)
    print("bin size:", B)

    ffd = FFD(s,B)
    print("\n\n\n FFD heuristic:")
    print("Solution:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\n IP formulation:")
    bins = solveBinPacking(s,B)
    print("Solution:")
    print(bins)
    print(len(bins), "bins")
