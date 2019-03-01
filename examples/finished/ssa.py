##@file ssa.py
#@brief multi-stage (serial) safety stock allocation model
"""
Approach: use SOS2 constraints for modeling non-linear functions.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict
import math
import random

from piecewise import convex_comb_sos

def ssa(n,h,K,f,T):
    """ssa -- multi-stage (serial) safety stock allocation model
    Parameters:
        - n: number of stages
        - h[i]: inventory cost on stage i
        - K: number of linear segments
        - f: (non-linear) cost function
        - T[i]: production lead time on stage i
    Returns the model with the piecewise linear relation on added variables x, f, and z.
    """

    model = Model("safety stock allocation")

    # calculate endpoints for linear segments
    a,b = {},{}
    for i in range(1,n+1):
        a[i] = [k for k in range(K)]
        b[i] = [f(i,k) for k in range(K)]

    # x: net replenishment time for stage i
    # y: corresponding cost
    # s: piecewise linear segment of variable x
    x,y,s = {},{},{}
    L = {} # service time of stage i
    for i in range(1,n+1):
        x[i],y[i],s[i] = convex_comb_sos(model,a[i],b[i])
        if i == 1:
            L[i] = model.addVar(ub=0, vtype="C", name="L[%s]"%i)
        else:
            L[i] = model.addVar(vtype="C", name="L[%s]"%i)
    L[n+1] = model.addVar(ub=0, vtype="C", name="L[%s]"%(n+1))

    for i in range(1,n+1):
        # net replenishment time for each stage i
        model.addCons(x[i] + L[i] == T[i] + L[i+1])

    model.setObjective(quicksum(h[i]*y[i] for i in range(1,n+1)), "minimize")

    model.data = x,s,L
    return model



def make_data():
    """creates example data set"""
    n = 30      # number of stages
    z = 1.65    # for 95% service level
    sigma = 100 # demand's standard deviation
    h = {}      # inventory cost
    T = {}      # production lead time
    h[n] = 1
    for i in range(n-1,0,-1):
        h[i] = h[i+1] + random.randint(30,50)
    K = 0 # number of segments (=sum of processing times)
    for i in range(1,n+1):
        T[i] = random.randint(3,5)      # production lead time at stage i
        K += T[i]
    return z,sigma,h,T,K,n



if __name__ == "__main__":
    random.seed(1)

    z,sigma,h,T,K,n = make_data()
    def f(i,k):
        return sigma*z*math.sqrt(k)

    model = ssa(n,h,K,f,T)
    model.optimize()

    # model.write("ssa.lp")
    x,s,L = model.data
    for i in range(1,n+1):
        for k in range(K):
            if model.getVal(s[i][k]) >= 0.001:
                print(s[i][k].name,model.getVal(s[i][k]))
        print
    print("%10s%10s%10s%10s" % ("Period","x","L","T"))
    for i in range(1,n+1):
        print("%10s%10s%10s%10s" % (i,model.getVal(x[i]), model.getVal(L[i]), T[i]))

    print("Objective:",model.getObjVal())
