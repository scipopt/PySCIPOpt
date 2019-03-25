##@file gpp.py
#@brief model for the graph partitioning problem
"""
Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""

from pyscipopt import Model, quicksum, multidict

def gpp(V,E):
    """gpp -- model for the graph partitioning problem
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns a model, ready to be solved.
    """
    model = Model("gpp")

    x = {}
    y = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
    for (i,j) in E:
        y[i,j] = model.addVar(vtype="B", name="y(%s,%s)"%(i,j))

    model.addCons(quicksum(x[i] for i in V) == len(V)/2, "Partition")

    for (i,j) in E:
        model.addCons(x[i] - x[j] <= y[i,j], "Edge(%s,%s)"%(i,j))
        model.addCons(x[j] - x[i] <= y[i,j], "Edge(%s,%s)"%(j,i))

    model.setObjective(quicksum(y[i,j] for (i,j) in E), "minimize")

    model.data = x
    return model


def gpp_qo(V,E):
    """gpp_qo -- quadratic optimization model for the graph partitioning problem
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns a model, ready to be solved.
    """
    model = Model("gpp")

    x = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)

    model.addCons(quicksum(x[i] for i in V) == len(V)/2, "Partition")

    model.setObjective(quicksum(x[i]*(1-x[j]) + x[j]*(1-x[i]) for (i,j) in E), "minimize")

    model.data = x
    return model


def gpp_qo_ps(V,E):
    """gpp_qo_ps -- quadratic optimization, positive semidefinite model for the graph partitioning problem
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns a model, ready to be solved.
    """
    model = Model("gpp")

    x = {}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)

    model.addCons(quicksum(x[i] for i in V) == len(V)/2, "Partition")

    model.setObjective(quicksum((x[i] - x[j]) * (x[i] - x[j]) for (i,j) in E), "minimize")

    model.data = x
    return model


def gpp_soco(V,E):
    """gpp -- model for the graph partitioning problem in soco
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns a model, ready to be solved.
    """
    model = Model("gpp model -- soco")

    x,s,z = {},{},{}
    for i in V:
        x[i] = model.addVar(vtype="B", name="x(%s)"%i)
    for (i,j) in E:
        s[i,j] = model.addVar(vtype="C", name="s(%s,%s)"%(i,j))
        z[i,j] = model.addVar(vtype="C", name="z(%s,%s)"%(i,j))

    model.addCons(quicksum(x[i] for i in V) == len(V)/2, "Partition")

    for (i,j) in E:
        model.addCons((x[i] + x[j] -1)*(x[i] + x[j] -1) <= s[i,j], "S(%s,%s)"%(i,j))
        model.addCons((x[j] - x[i])*(x[j] - x[i]) <= z[i,j], "Z(%s,%s)"%(i,j))
        model.addCons(s[i,j] + z[i,j]  == 1, "P(%s,%s)"%(i,j))

    # # triangle inequalities (seem to make model slower)
    # for i in V:
    #     for j in V:
    #         for k in V:
    #             if (i,j) in E and (j,k) in E and (i,k) in E:
    #                 print("\t***",(i,j,k)
    #                 model.addCons(z[i,j] + z[j,k] + z[i,k] <= 2, "T1(%s,%s,%s)"%(i,j,k))
    #                 model.addCons(z[i,j] + s[j,k] + s[i,k] <= 2, "T2(%s,%s,%s)"%(i,j,k))
    #                 model.addCons(s[i,j] + s[j,k] + z[i,k] <= 2, "T3(%s,%s,%s)"%(i,j,k))
    #                 model.addCons(s[i,j] + z[j,k] + s[i,k] <= 2, "T4(%s,%s,%s)"%(i,j,k))

    model.setObjective(quicksum(z[i,j] for (i,j) in E), "minimize")

    model.data = x,s,z
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
    V,E = make_data(4,.5)
    print("edges:",E)

    print("\n\n\nStandard model:")
    model = gpp(V,E)
    model.optimize()
    print("Optimal value:", model.getObjVal())
    x = model.data
    print("partition:")
    print([i for i in V if model.getVal(x[i]) >= .5])
    print([i for i in V if model.getVal(x[i]) < .5])

    print("\n\n\nQuadratic optimization")
    model = gpp_qo(V,E)
    model.optimize()
    model.writeProblem("gpp_qo.lp")
    print("Optimal value:", model.getObjVal())
    x = model.data
    print("partition:")
    print([i for i in V if model.getVal(x[i]) >= .5])
    print([i for i in V if model.getVal(x[i]) < .5])

    print("\n\n\nQuadratic optimization - positive semidefinite")
    model = gpp_qo_ps(V,E)
    model.optimize()
    model.writeProblem("gpp_qo.lp")
    print("Optimal value:", model.getObjVal())
    x = model.data
    print("partition:")
    print([i for i in V if model.getVal(x[i]) >= .5])
    print([i for i in V if model.getVal(x[i]) < .5])

    print("\n\n\nSecond order cone optimization")
    model = gpp_soco(V,E)
    model.optimize()
    model.writeProblem("tmp.lp")
    status = model.getStatus()
    if status == "optimal":
        print("Optimal value:", model.getObjVal())
        x,s,z = model.data
        print("partition:")
        print([i for i in V if model.getVal(x[i]) >= .5])
        print([i for i in V if model.getVal(x[i]) < .5])

        for (i,j) in s:
            print("(%s,%s)\t%s\t%s" % (i,j,model.getVal(s[i,j]),model.getVal(z[i,j])))
