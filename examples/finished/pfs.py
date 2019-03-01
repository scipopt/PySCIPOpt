##@file pfs.py
#@brief model for the permutation flow shop problem
"""
Use a position index formulation for modeling the permutation flow
shop problem, with the objective of minimizing the makespan (maximum
completion time).

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
from pyscipopt import Model, quicksum, multidict

def permutation_flow_shop(n,m,p):
    """gpp -- model for the graph partitioning problem
    Parameters:
        - n: number of jobs
        - m: number of machines
        - p[i,j]: processing time of job i on machine j
    Returns a model, ready to be solved.
    """
    model = Model("permutation flow shop")

    x,s,f = {},{},{}
    for j in range(1,n+1):
        for k in range(1,n+1):
            x[j,k] = model.addVar(vtype="B", name="x(%s,%s)"%(j,k))

    for i in range(1,m+1):
        for k in range(1,n+1):
            s[i,k] = model.addVar(vtype="C", name="start(%s,%s)"%(i,k))
            f[i,k] = model.addVar(vtype="C", name="finish(%s,%s)"%(i,k))

    for j in range(1,n+1):
        model.addCons(quicksum(x[j,k] for k in range(1,n+1)) == 1, "Assign1(%s)"%(j))
        model.addCons(quicksum(x[k,j] for k in range(1,n+1)) == 1, "Assign2(%s)"%(j))

    for i in range(1,m+1):
        for k in range(1,n+1):
            if k != n:
                model.addCons(f[i,k] <= s[i,k+1], "FinishStart(%s,%s)"%(i,k))
            if i != m:
                model.addCons(f[i,k] <= s[i+1,k], "Machine(%s,%s)"%(i,k))

            model.addCons(s[i,k] + quicksum(p[i,j]*x[j,k] for j in range(1,n+1)) <= f[i,k],
                            "StartFinish(%s,%s)"%(i,k))

    model.setObjective(f[m,n], "minimize")

    model.data = x,s,f
    return model


def make_data(n,m):
    """make_data: prepare matrix of m times n random processing times"""
    p = {}
    for i in range(1,m+1):
        for j in range(1,n+1):
            p[i,j] = random.randint(1,10)
    return p


def example():
    """creates example data set"""
    proc = [[2,3,1],[4,2,3],[1,4,1]]
    p = {}
    for i in range(3):
        for j in range(3):
            p[i+1,j+1] = proc[j][i]
    return p


if __name__ == "__main__":
    random.seed(1)
    n = 15
    m = 10
    p = make_data(n,m)

    # n = 3
    # m = 3
    # p = example()
    print("processing times (%s jobs, %s machines):" % (n,m))
    for i in range(1,m+1):
        for j in range(1,n+1):
            print(p[i,j],)
        print

    model = permutation_flow_shop(n,m,p)
    # model.write("permflow.lp")
    model.optimize()
    x,s,f = model.data
    print("Optimal value:", model.getObjVal())

    ### for (j,k) in sorted(x):
    ###     if x[j,k].X > 0.5:
    ###         print(x[j,k].VarName,x[j,k].X
    ###
    ### for i in sorted(s):
    ###     print(s[i].VarName,s[i].X
    ###
    ### for i in sorted(f):
    ###     print(f[i].VarName,f[i].X

    # x[j,k] = 1 if j is the k-th job; extract job sequence:
    seq = [j for (k,j) in sorted([(k,j) for (j,k) in x if model.getVal(x[j,k]) > 0.5])]
    print("optimal job permutation:",seq)
