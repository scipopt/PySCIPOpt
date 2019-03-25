##@file atsp.py
#@brief solve the asymmetric traveling salesman problem

"""

formulations implemented:
    - mtz -- Miller-Tucker-Zemlin's potential formulation
    - mtz_strong -- Miller-Tucker-Zemlin's potential formulation with stronger constraint
    - scf -- single-commodity flow formulation
    - mcf -- multi-commodity flow formulation

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

def mtz(n,c):
    """mtz: Miller-Tucker-Zemlin's model for the (asymmetric) traveling salesman problem
    (potential formulation)
    Parameters:
        - n: number of nodes
        - c[i,j]: cost for traversing arc (i,j)
    Returns a model, ready to be solved.
    """

    model = Model("atsp - mtz")

    x,u = {},{}
    for i in range(1,n+1):
        u[i] = model.addVar(lb=0, ub=n-1, vtype="C", name="u(%s)"%i)
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

    for i in range(1,n+1):
        for j in range(2,n+1):
            if i != j:
                model.addCons(u[i] - u[j] + (n-1)*x[i,j] <= n-2, "MTZ(%s,%s)"%(i,j))

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")
    
    model.data = x,u
    return model



def mtz_strong(n,c):
    """mtz_strong: Miller-Tucker-Zemlin's model for the (asymmetric) traveling salesman problem
    (potential formulation, adding stronger constraints)
    Parameters:
        n - number of nodes
        c[i,j] - cost for traversing arc (i,j)
    Returns a model, ready to be solved.
    """

    model = Model("atsp - mtz-strong")
    
    x,u = {},{}
    for i in range(1,n+1):
        u[i] = model.addVar(lb=0, ub=n-1, vtype="C", name="u(%s)"%i)
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

    for i in range(1,n+1):
        for j in range(2,n+1):
            if i != j:
                model.addCons(u[i] - u[j] + (n-1)*x[i,j] + (n-3)*x[j,i] <= n-2, "LiftedMTZ(%s,%s)"%(i,j))

    for i in range(2,n+1):
        model.addCons(-x[1,i] - u[i] + (n-3)*x[i,1] <= -2, name="LiftedLB(%s)"%i)
        model.addCons(-x[i,1] + u[i] + (n-3)*x[1,i] <= n-2, name="LiftedUB(%s)"%i)

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")
    
    model.data = x,u
    return model


def scf(n,c):
    """scf: single-commodity flow formulation for the (asymmetric) traveling salesman problem
    Parameters:
        - n: number of nodes
        - c[i,j]: cost for traversing arc (i,j)
    Returns a model, ready to be solved.
    """
    model = Model("atsp - scf")

    x,f = {},{}
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
                if i==1:
                    f[i,j] = model.addVar(lb=0, ub=n-1, vtype="C", name="f(%s,%s)"%(i,j))
                else:
                    f[i,j] = model.addVar(lb=0, ub=n-2, vtype="C", name="f(%s,%s)"%(i,j))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

    model.addCons(quicksum(f[1,j] for j in range(2,n+1)) == n-1, "FlowOut")

    for i in range(2,n+1):
        model.addCons(quicksum(f[j,i] for j in range(1,n+1) if j != i) - \
                        quicksum(f[i,j] for j in range(1,n+1) if j != i) == 1, "FlowCons(%s)"%i)

    for j in range(2,n+1):
        model.addCons(f[1,j] <= (n-1)*x[1,j], "FlowUB(%s,%s)"%(1,j))
        for i in range(2,n+1):
            if i != j:
                model.addCons(f[i,j] <= (n-2)*x[i,j], "FlowUB(%s,%s)"%(i,j))

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")

    model.data = x,f
    return model



def mcf(n,c):
    """mcf: multi-commodity flow formulation for the (asymmetric) traveling salesman problem
    Parameters:
        - n: number of nodes
        - c[i,j]: cost for traversing arc (i,j)
    Returns a model, ready to be solved.
    """
    model = Model("mcf")

    x,f = {},{}
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
            if i != j and j != 1:
                for k in range(2,n+1):
                    if i != k:
                        f[i,j,k] = model.addVar(ub=1, vtype="C", name="f(%s,%s,%s)"%(i,j,k))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

    for k in range(2,n+1):
        model.addCons(quicksum(f[1,i,k] for i in range(2,n+1) if (1,i,k) in f) == 1, "FlowOut(%s)"%k)
        model.addCons(quicksum(f[i,k,k] for i in range(1,n+1) if (i,k,k) in f) == 1, "FlowIn(%s)"%k)

        for i in range(2,n+1):
            if i != k:
                model.addCons(quicksum(f[j,i,k] for j in range(1,n+1) if (j,i,k) in f) == \
                                quicksum(f[i,j,k] for j in range(1,n+1) if (i,j,k) in f),
                                "FlowCons(%s,%s)"%(i,k))

    for (i,j,k) in f:
        model.addCons(f[i,j,k] <= x[i,j], "FlowUB(%s,%s,%s)"%(i,j,k))

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")

    model.data = x,f
    return model



def sequence(arcs):
    """sequence: make a list of cities to visit, from set of arcs"""
    succ = {}
    for (i,j) in arcs:
        succ[i] = j
    curr = 1    # first node being visited
    sol = [curr]
    for i in range(len(arcs)-2):
        curr = succ[curr]
        sol.append(curr)
    return sol


if __name__ == "__main__":
    n = 5
    c = { (1,1):0,  (1,2):1989,  (1,3):102, (1,4):102, (1,5):103,
          (2,1):104, (2,2):0,  (2,3):11,  (2,4):104, (2,5):108,
          (3,1):107, (3,2):108, (3,3):0,  (3,4):19,  (3,5):102,
          (4,1):109, (4,2):102, (4,3):107, (4,4):0,  (4,5):15,
          (5,1):13,  (5,2):103, (5,3):104, (5,4):101, (5,5):0,
         }

    model = mtz(n,c)
    model.hideOutput() # silent mode
    model.optimize()
    cost = model.getObjVal()
    print()
    print("Miller-Tucker-Zemlin's model:")
    print("Optimal value:", cost)
    #model.printAttr("X")
    for v in model.getVars():
        if model.getVal(v) > 0.001:
            print(v.name, "=", model.getVal(v))

    x,u = model.data
    sol = [i for (p,i) in sorted([(int(model.getVal(u[i])+.5),i) for i in range(1,n+1)])]
    print(sol)
    arcs = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > .5]
    sol = sequence(arcs)
    print(sol)
    # assert cost == 5

    model = mtz_strong(n,c)
    model.hideOutput() # silent mode
    model.optimize()
    cost = model.getObjVal()
    print()
    print("Miller-Tucker-Zemlin's model with stronger constraints:")
    print("Optimal value:",cost)
    #model.printAttr("X")
    for v in model.getVars():
        if model.getVal(v) > 0.001:
            print(v.name, "=", model.getVal(v))

    x,u = model.data
    sol = [i for (p,i) in sorted([(int(model.getVal(u[i])+.5),i) for i in range(1,n+1)])]
    print(sol)
    arcs = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > .5]
    sol = sequence(arcs)
    print(sol)
    # assert cost == 5

    model = scf(n,c)
    model.hideOutput() # silent mode
    model.optimize()
    cost = model.getObjVal()
    print()
    print("Single-commodity flow formulation:")
    print("Optimal value:",cost)
    #model.printAttr("X")
    for v in model.getVars():
        if model.getVal(v) > 0.001:
            print(v.name, "=", model.getVal(v))

    x,f = model.data
    arcs = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > .5]
    sol = sequence(arcs)
    print(sol)
    # assert cost == 5

    model = mcf(n,c)
    model.hideOutput() # silent mode
    model.optimize()
    cost = model.getObjVal()
    print()
    print("Multi-commodity flow formulation:")
    print("Optimal value:",cost)
    #model.printAttr("X")
    for v in model.getVars():
        if model.getVal(v)>0.001:
            print(v.name, "=", model.getVal(v))

    x,f = model.data
    arcs = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > .5]
    sol = sequence(arcs)
    print(sol)
    # assert cost == 5
