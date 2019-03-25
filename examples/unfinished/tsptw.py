"""
tsptw.py: solve the asymmetric traveling salesman problem with time window constraints

minimize the travel cost for visiting n customers exactly once;
each customer has a time window within which the salesman must visit him

formulations based on Miller-Tucker-Zemlin's formulation, for the atsp;
one- and two-index potential variants

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
from pyscipopt import Model, quicksum, multidict

def mtztw(n,c,e,l):
    """mtzts: model for the traveling salesman problem with time windows
    (based on Miller-Tucker-Zemlin's one-index potential formulation)
    Parameters:
        - n: number of nodes
        - c[i,j]: cost for traversing arc (i,j)
        - e[i]: earliest date for visiting node i
        - l[i]: latest date for visiting node i
    Returns a model, ready to be solved.
    """
    model = Model("tsptw - mtz")

    x,u = {},{}
    for i in range(1,n+1):
        u[i] = model.addVar(lb=e[i], ub=l[i], vtype="C", name="u(%s)"%i)
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

    for i in range(1,n+1):
        for j in range(2,n+1):
            if i != j:
                M = max(l[i] + c[i,j] - e[j], 0)
                model.addCons(u[i] - u[j] + M*x[i,j] <= M-c[i,j], "MTZ(%s,%s)"%(i,j))

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")

    model.data = x,u
    return model



def mtz2tw(n,c,e,l):
    """mtz: model for the traveling salesman problem with time windows
    (based on Miller-Tucker-Zemlin's one-index potential formulation, stronger constraints)
    Parameters:
        - n: number of nodes
        - c[i,j]: cost for traversing arc (i,j)
        - e[i]: earliest date for visiting node i
        - l[i]: latest date for visiting node i
    Returns a model, ready to be solved.
    """
    model = Model("tsptw - mtz-strong")
    
    x,u = {},{}
    for i in range(1,n+1):
        u[i] = model.addVar(lb=e[i], ub=l[i], vtype="C", name="u(%s)"%i)
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

        for j in range(2,n+1):
            if i != j:
                M1 = max(l[i] + c[i,j] - e[j], 0)
                M2 = max(l[i] + min(-c[j,i], e[j]-e[i]) - e[j], 0)
                model.addCons(u[i] + c[i,j] - M1*(1-x[i,j]) + M2*x[j,i] <= u[j], "LiftedMTZ(%s,%s)"%(i,j))

    for i in range(2,n+1):
        model.addCons(e[i] + quicksum(max(e[j]+c[j,i]-e[i],0) * x[j,i] for j in range(1,n+1) if i != j) \
                        <= u[i], "LiftedLB(%s)"%i)

        model.addCons(u[i] <= l[i] - \
                        quicksum(max(l[i]-l[j]+c[i,j],0) * x[i,j] for j in range(2,n+1) if i != j), \
                        "LiftedUB(%s)"%i)

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")

    model.data = x,u
    return model


def tsptw2(n,c,e,l):
    """tsptw2: model for the traveling salesman problem with time windows
    (based on Miller-Tucker-Zemlin's formulation, two-index potential)
    Parameters:
        - n: number of nodes
        - c[i,j]: cost for traversing arc (i,j)
        - e[i]: earliest date for visiting node i
        - l[i]: latest date for visiting node i
    Returns a model, ready to be solved.
    """
    model = Model("tsptw2")
    
    x,u = {},{}
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
                u[i,j] = model.addVar(vtype="C", name="u(%s,%s)"%(i,j))

    for i in range(1,n+1):
        model.addCons(quicksum(x[i,j] for j in range(1,n+1) if j != i) == 1, "Out(%s)"%i)
        model.addCons(quicksum(x[j,i] for j in range(1,n+1) if j != i) == 1, "In(%s)"%i)

    for j in range(2,n+1):
        model.addCons(quicksum(u[i,j] + c[i,j]*x[i,j] for i in range(1,n+1) if i != j) -
                        quicksum(u[j,k] for k in range(1,n+1) if k != j) <= 0, "Relate(%s)"%j)

    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                model.addCons(e[i]*x[i,j] <= u[i,j], "LB(%s,%s)"%(i,j))
                model.addCons(u[i,j] <= l[i]*x[i,j], "UB(%s,%s)"%(i,j))

    model.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in x), "minimize")

    model.data = x,u
    return model


def distance(x1,y1,x2,y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def make_data(n,width):
    """make_data: compute matrix distance and time windows."""
    x = dict([(i,100*random.random()) for i in range(1,n+1)])
    y = dict([(i,100*random.random()) for i in range(1,n+1)])
    c = {}
    for i in range(1,n+1):
        for j in range(1,n+1):
            if j != i:
                c[i,j] = distance(x[i],y[i],x[j],y[j])

    e = {1:0}
    l = {1:0}
    start = 0
    delta = int(76.*math.sqrt(n)/n * width)+1
    for i in range(1,n):
        j = i+1
        start += c[i,j]
        e[j] = max(start-delta,0)
        l[j] = start + delta

    return c,x,y,e,l


if __name__ == "__main__":
    EPS = 1.e-6
    # n = 10
    # width = 10
    # c,x,y,e,l = make_data(n,width)

    n = 5
    c = { (1,1):0,  (1,2):9,  (1,3):10, (1,4):10, (1,5):10,
          (2,1):10, (2,2):0,  (2,3):9,  (2,4):10, (2,5):10,
          (3,1):10, (3,2):10, (3,3):0,  (3,4):9,  (3,5):10,
          (4,1):10, (4,2):10, (4,3):10, (4,4):0,  (4,5):9,
          (5,1):9,  (5,2):10, (5,3):10, (5,4):10, (5,5):0,
         }
    e = {1:0, 2:0, 3:0, 4:0, 5:0}
    l = {1:100, 2:100, 3:10, 4:100, 5:100}

    print(c)
    print(e)
    print(l)

    model = mtztw(n,c,e,l)
    model.optimize()
    x,u = model.data

    sol = [i for (v,i) in sorted([(model.getVal(u[i]),i) for i in u])]
    print("mtz:")
    print(sol)
    print("Optimal value:", model.getObjVal())

    model = mtz2tw(n,c,e,l)
    model.optimize()
    x,u = model.data

    sol = [i for (v,i) in sorted([(model.getVal(u[i]),i) for i in u])]
    print("mtz2:")
    print(sol)
    print("Optimal value:", model.getObjVal())

    # ### import networkx as NX
    # ### import matplotlib.pyplot as P
    # ### P.clf()
    # ### G = NX.Graph()
    # ### G.add_nodes_from(range(1,n+1))
    # ### position = {}
    # ### for i in range(1,n+1):
    # ###     position[i] = (x[i],y[i])
    # ###
    # ### for i in range(n-1):
    # ###     G.add_edge(perm[i], perm[i+1])
    # ### G.add_edge(perm[n-1],perm[0])
    # ### NX.draw(G,position)
    # ### P.show()


    print("TWO INDEX MODEL")
    model = tsptw2(n,c,e,l)
    model.optimize()
    print("Optimal value:", model.getObjVal())
    x,u = model.data
    for (i,j) in x:
        if model.getVal(x[i,j]) > EPS:
            print(x[i,j].name,i,j,model.getVal(x[i,j]))

    start_time = [0]*(n+1)
    for (i,j) in u:
        if model.getVal(u[i,j]) > EPS:
            print(u[i,j].name,i,j,model.getVal(u[i,j]))
            start_time[j] += model.getVal(u[i,j])

    start = [i for v,i in sorted([(start_time[i],i) for i in range(1,n+1)])]
    print(start)
