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

from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

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
    P = []      # patterns
    m = len(w)

    # Generate initial patterns with one size for each item width
    for (i,width) in enumerate(w):
        pat = [0]*m  # vector of number of orders to be packed into one roll (bin)
        pat[i] = int(B/width)
        P.append(pat)
    

    K = len(P)
    master = Model("master LP") # master LP problem
    x = {}

    for k in range(K):
        x[k] = master.addVar(vtype="I", lb = 0, name="x(%s)"%k)

    y = master.addVar(ub=0, lb=0, name="y") # This variable is required due to issues with the way SCIP deals with dual variables in bounded constraints. See https://github.com/scipopt/PySCIPOpt#dual-values

    orders = {}

    for i in range(m):
        orders[i] = master.addCons(quicksum(P[k][i]*x[k] + y for k in range(K)) >= q[i], "Order(%s)"%i)

    master.setObjective(quicksum(x[k] for k in range(K)), "minimize")
    master.hideOutput() # Silent mode
    
    while True:

        relax = Model(sourceModel = master) # Linear relxation of RMP
        for var in relax.getVars():
            relax.chgVarType(var, "CONTINUOUS")

        relax.setPresolve(SCIP_PARAMSETTING.OFF)
        relax.setHeuristics(SCIP_PARAMSETTING.OFF)
        relax.disablePropagation()        
        relax.optimize()

        pi = [relax.getDualsolLinear(c) for c in relax.getConss()] # keep dual variables

        knapsack = Model("KP")     # knapsack sub-problem
        knapsack.setMaximize       # maximize
        y = {}

        for i in range(m):
            y[i] = knapsack.addVar(lb=0, ub=q[i], vtype="INTEGER", name="y(%s)"%i)

        knapsack.addCons(quicksum(w[i]*y[i] for i in range(m)) <= B, "Width")


        knapsack.setObjective(quicksum(pi[i]*y[i] for i in range(m)), "maximize")

        knapsack.hideOutput() # silent mode
        knapsack.optimize()

        if knapsack.getObjVal() < 1 + EPS: # break if no more columns
            break

        pat = [int(knapsack.getVal(y[i]) + 0.5) for i in y]      # new pattern
        P.append(pat)

        x[K] = master.addVar(obj=1, vtype="I", name="x(%s)"%K)

        # add new column to the master problem
        for i, con in enumerate(master.getConss()):
            master.addConsCoeff(con, x[K], P[K-1][i])

        K += 1

    master.optimize()

    rolls = []
    for k in x:
        for j in range(int(master.getVal(x[k]) + .5)):
            rolls.append(sorted([w[i] for i in range(m) if P[k][i]>0 for j in range(P[k][i])]))
    rolls.sort()
    return rolls


def cuttingStockKantorovich(w, q, B):
    """
    Direct formulation of the Cutting Stock problem.
    """

    model = Model("Naive Cutting Stock")
    m = max(w)*max(q) # m rolls
    n = len(q) # n orders  
    y = {}
    for j in range(m):
        y[j] = model.addVar(name = "y[%s]" % j, vtype="BINARY")
    
    x = {}
    for j in range(m):
        for i in range(n):
            x[i,j] = model.addVar(name = "x[%s,%s]" %(i,j), lb = 0, vtype="INTEGER")
            model.addCons(x[i,j] <= q[i]*y[j])

    for i in range(n):
        model.addCons(quicksum(x[i,j] for j in range(m)) == q[i])

    for j in range(m):
        model.addCons((quicksum(w[i]*x[i,j] for i in range(n)) <= B))

    model.setObjective(quicksum(y[j] for j in range(m)), "minimize")
    model.hideOutput()
    model.optimize()

    return model.getObjVal()


def generateCuttingStockExample():
    # Generates small/medium sized instances
    from random import randint
    B = randint(30,70)
    n_orders = randint(2,5)
    w = [randint(10,B) for _ in range(n_orders)]
    q = [randint(1,10) for _ in range(n_orders)]
    return w,q,B


if __name__ == "__main__":
    from random import seed
    seed(42)
    for i in range(10):
        w, q, B = generateCuttingStockExample() # Getting a new instance
        naive_obj = cuttingStockKantorovich(w,q,B) # Solving it with original MIP formulation
        column_generation_obj = solveCuttingStock(w,q,B) # Solving it with delayed column generation
        
        assert abs(naive_obj - len(column_generation_obj)) < EPS, "Different objectives" # Checking that the heuristic works for these instances
        print("Test %i OKAY." %(i+1), "Item widths: ", w, "Item demands ", q, "Roll Capcity ", B)
    
    w, q, B = [13,16], [3,3], 46 # <- this is an example where the relaxation does not lead to the optimal solution. Branching is required.
    naive_obj = cuttingStockKantorovich(w,q,B)
    column_generation_obj = solveCuttingStock(w,q,B)
    print("Kantorovich Cutting Stock: %i bins | Column Generation Heuristic: %i bins" % (naive_obj, len(column_generation_obj)))