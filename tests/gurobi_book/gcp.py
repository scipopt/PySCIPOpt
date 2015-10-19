"""
gcp.py: model for the graph coloring problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from gurobipy import *

def gcp(V,E,K):
    """gcp -- model for minimizing the number of colors in a graph
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: upper bound on the number of colors
    Returns a model, ready to be solved.
    """
    model = Model("gcp")
    x,y = {},{}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)"%k)
        for i in V:
            x[i,k] = model.addVar(vtype="B", name="x(%s,%s)"%(i,k))
    model.update()

    for i in V:
        model.addConstr(quicksum(x[i,k] for k in range(K)) == 1, "AssignColor(%s)"%i)

    for (i,j) in E:
        for k in range(K):
            model.addConstr(x[i,k] + x[j,k] <= y[k], "NotSameColor(%s,%s,%s)"%(i,j,k))

    model.setObjective(quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


def gcp_low(V,E,K):
    """gcp_low -- model for minimizing the number of colors in a graph
    (use colors with low indices)
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: upper bound to the number of colors
    Returns a model, ready to be solved.
    """
    model = Model("gcp - low colors")
    x,y = {},{}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)"%k)
        for i in V:
            x[i,k] = model.addVar(vtype="B", name="x(%s,%s)"%(i,k))
    model.update()

    for i in V:
        model.addConstr(quicksum(x[i,k] for k in range(K)) == 1, "AssignColor(%s)" % i)

    for (i,j) in E:
        for k in range(K):
            model.addConstr(x[i,k] + x[j,k] <= y[k], "NotSameColor(%s,%s,%s)"%(i,j,k))

    for k in range(K-1):
        model.addConstr(y[k] >= y[k+1], "LowColor(%s)"%k)

    model.setObjective(quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


def gcp_sos(V,E,K):
    """gcp_sos -- model for minimizing the number of colors in a graph
    (use sos type 1 constraints)
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
        - K: upper bound to the number of colors
    Returns a model, ready to be solved.
    """
    model = Model("gcp - sos constraints")
    x,y = {},{}
    for k in range(K):
        y[k] = model.addVar(vtype="B", name="y(%s)"%k)
        for i in V:
            x[i,k] = model.addVar(vtype="B", name="x(%s,%s)"%(i,k))
    model.update()

    for i in V:
        model.addConstr(quicksum(x[i,k] for k in range(K)) == 1, "AssignColor(%s)" % i)
        model.addSOS(1, [x[i,k] for k in range(K)])

    for (i,j) in E:
        for k in range(K):
            model.addConstr(x[i,k] + x[j,k] <= y[k], "NotSameColor(%s,%s,%s)"%(i,j,k))

    for k in range(K-1):
        model.addConstr(y[k] >= y[k+1], "LowColor(%s)"%k)

    model.setObjective(quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    model.update()
    model.__data = x
    return model


import random
def make_data(n,prob):
    """make_data: prepare data for a random graph
    Parameters:
       - n: number of vertices
       - prob: probability of existence of an edge, for each pair of vertices
    Returns a tuple with a list of vertices and a list edges.
       """
    V = range(1,n+1)
    E = [(i,j) for i in V for j in V if i < j and random.random() < prob]
    return V,E


if __name__ == "__main__":
    random.seed(1)
    V,E = make_data(20,.5)
    K = 10        # upper bound to the number of colors
    print "n,K=",len(V),K

    model = gcp_low(V,E,K)
    model.optimize()
    print "Opt.value=",model.ObjVal
    x = model.__data
    color = {}
    for i in V:
        for k in range(K):
            if x[i,k].X > 0.5:
                color[i] = k
    print "colors:",color

    import time,sys
    setParam(GRB.Param.Threads,1)
    models = [gcp,gcp_low,gcp_sos]
    cpu = {}
    N = 25      # number of observations
    print "#size\t%s\t%s\t%s" % tuple(m.__name__ for m in models)
    for size in range(250):
        print size,"\t",
        K = size
        for prob in [0.1]:
            for m in models:
                name = m.__name__
                if not cpu.has_key((name,size-1,prob)) or cpu[name,size-1,prob] < 100:
                    cpu[name,size,prob] = 0.
                    for t in range(N):
                        tinit = time.clock()
                        random.seed(t)
                        V,E = make_data(size,prob)
                        model = m(V,E,K)
                        model.Params.OutputFlag = 0     # silent mode
                        model.optimize()
                        assert model.ObjVal >= 0 and model.ObjVal <= K
                        tend = time.clock()
                        cpu[name,size,prob] += tend - tinit
                    cpu[name,size,prob] /= N
                else:
                    cpu[name,size,prob] = "-"
                print cpu[name,size,prob],"\t",
        print
        sys.stdout.flush()
