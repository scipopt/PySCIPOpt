#todo relax function needed
"""
cutstock.py:  use SCIP for solving the cutting stock problem.

The instance of the cutting stock problem is represented by the two
lists of m items of size w=(w_i) and and quantity q=(q_i).

The roll size is B.

Given packing patterns t_1, ...,t_k,...t_K where t_k is a vector of
the numbers of items cut from a roll, the problem is reduced to the
following LP:

    minimize   sum_{k} x_k
    subject to sum_{k} t_k(i) x_k >= q_i    for all i
               x_k >=0                      for all k.

We apply a column generation approch (Gilmore-Gomory approach) in
which we generate cutting patterns by solving a knapsack sub-problem.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from pyscipopt import Model, quicksum, multidict

LOG = True
EPS = 1.e-6

def solveCuttingStock(w,q,B):
    """solveCuttingStock: use column generation (Gilmore-Gomory approach).
    Parameters:
        - w: list of item's widths
        - q: number of items of a width
        - B: bin/roll capacity
    Returns a solution: list of lists, each of which with the cuts of a roll.
    """
    t = []      # patterns
    m = len(w)

    # Generate initial patterns with one size for each item width
    for (i,width) in enumerate(w):
        pat = [0]*m  # vector of number of orders to be packed into one roll (bin)
        pat[i] = int(B/width)
        t.append(pat)

    # if LOG:
    #     print "sizes of orders=",w
    #     print "quantities of orders=",q
    #     print "roll size=",B
    #     print "initial patterns",t

    K = len(t)
    master = Model("master LP") # master LP problem
    x = {}

    for k in range(K):
        x[k] = master.addVar(vtype="I", name="x(%s)"%k)

    orders = {}

    for i in range(m):
        orders[i] = master.addCons(
            quicksum(t[k][i]*x[k] for k in range(K) if t[k][i] > 0) >= q[i], "Order(%s)"%i)

    master.setObjective(quicksum(x[k] for k in range(K)), "minimize")

    # master.Params.OutputFlag = 0 # silent mode

    # iter = 0
    while True:
        # print "current patterns:"
        # for ti in t:
        #     print ti
        # print

        # iter += 1
        relax = master.relax()
        relax.optimize()
        pi = [relax.getDualsolLinear(c) for c in relax.getConss()] # keep dual variables

        knapsack = Model("KP")     # knapsack sub-problem
        knapsack.setMaximize       # maximize
        y = {}

        for i in range(m):
            y[i] = knapsack.addVar(lb=0, ub=q[i], vtype="I", name="y(%s)"%i)

        knapsack.addCons(quicksum(w[i]*y[i] for i in range(m)) <= B, "Width")

        knapsack.setObjective(quicksum(pi[i]*y[i] for i in range(m)), "maximize")

        knapsack.hideOutput() # silent mode
        knapsack.optimize()
        # if LOG:
        #     print "objective of knapsack problem:", knapsack.ObjVal
        if knapsack.getObjVal() < 1+EPS: # break if no more columns
            break

        pat = [int(y[i].X+0.5) for i in y]      # new pattern
        t.append(pat)
        # if LOG:
        #     print "shadow prices and new pattern:"
        #     for (i,d) in enumerate(pi):
        #         print "\t%5s%12s%7s" % (i,d,pat[i])
        #     print

        # add new column to the master problem
        col = Column()
        for i in range(m):
            if t[K][i] > 0:
                col.addTerms(t[K][i], orders[i])
        x[K] = master.addVar(obj=1, vtype="I", name="x(%s)"%K, column=col)

        # master.write("MP" + str(iter) + ".lp")
        K += 1


    # Finally, solve the IP
    # if LOG:
    #     master.Params.OutputFlag = 1 # verbose mode
    master.optimize()

    # if LOG:
    #     print
    #     print "final solution (integer master problem):  objective =", master.ObjVal
    #     print "patterns:"
    #     for k in x:
    #         if x[k].X > EPS:
    #             print "pattern",k,
    #             print "\tsizes:",
    #             print [w[i] for i in range(m) if t[k][i]>0 for j in range(t[k][i]) ],
    #             print "--> %s rolls" % int(x[k].X+.5)

    rolls = []
    for k in x:
        for j in range(int(x[k].X + .5)):
            rolls.append(sorted([w[i] for i in range(m) if t[k][i]>0 for j in range(t[k][i])]))
    rolls.sort()
    return rolls



def CuttingStockExample1():
    """CuttingStockExample1: create toy instance for the cutting stock problem."""
    B = 110            # roll width (bin size)
    w = [20,45,50,55,75]  # width (size) of orders (items)
    q = [48,35,24,10,8]  # quantitiy of orders
    return w,q,B

def CuttingStockExample2():
    """CuttingStockExample2: create toy instance for the cutting stock problem."""
    B = 9            # roll width (bin size)
    w = [2,3,4,5,6,7,8]   # width (size) of orders (items)
    q = [4,2,6,6,2,2,2]  # quantitiy of orders
    return w,q,B


def mkCuttingStock(s):
    """mkCuttingStock: convert a bin packing instance into cutting stock format"""
    w,q = [],[]   # list of different widths (sizes) of items, their quantities
    for item in sorted(s):
        if w == [] or item != w[-1]:
            w.append(item)
            q.append(1)
        else:
            q[-1] += 1
    return w,q


def mkBinPacking(w,q):
    """mkBinPacking: convert a cutting stock instance into bin packing format"""
    s = []
    for j in range(len(w)):
        for i in range(q[j]):
            s.append(w[j])
    return s


if __name__ == "__main__":
    from bpp import FFD,solveBinPacking

    w,q,B = CuttingStockExample1()
    # w,q,B = CuttingStockExample2()
    # n = 500
    # B = 100
    # s,B = DiscreteUniform(n,18,50,B)

    s = mkBinPacking(w,q)
    ffd = FFD(s,B)
    print("\n\n\nSolution of FFD:")
    print(ffd)
    print(len(ffd), "bins")

    print("\n\n\nCutting stock problem, column generation:")
    rolls = solveCuttingStock(w,q,B)
    print(len(rolls), "rolls:")
    print(rolls)

    print("\n\n\nBin packing problem:")
    bins = solveBinPacking(s,B)
    print(len(bins), "bins:")
    print(bins)
