"""
lotsizing.py:  solve the multi-item, multi-stage lot-sizing problem.

Approaches:
    - mils_standard: standard formulation
    - mils_echelon: echelon formulation

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import random
from pyscipopt import Model, quicksum, multidict

def mils_standard(T,K,P,f,g,c,d,h,a,M,UB,phi):
    """
    mils_standard: standard formulation for the multi-item, multi-stage lot-sizing problem

    Parameters:
        - T: number of periods
        - K: set of resources
        - P: set of items
        - f[t,p]: set-up costs (on period t, for product p)
        - g[t,p]: set-up times
        - c[t,p]: variable costs
        - d[t,p]: demand values
        - h[t,p]: holding costs
        - a[t,k,p]: amount of resource k for producing p in period t
        - M[t,k]: resource k upper bound on period t
        - UB[t,p]: upper bound of production time of product p in period t
        - phi[(i,j)]: units of i required to produce a unit of j (j parent of i)
    """
    model = Model("multi-stage lotsizing -- standard formulation")

    y,x,I = {},{},{}
    Ts = range(1,T+1)
    for p in P:
        for t in Ts:
            y[t,p] = model.addVar(vtype="B", name="y(%s,%s)"%(t,p))
            x[t,p] = model.addVar(vtype="C",name="x(%s,%s)"%(t,p))
            I[t,p] = model.addVar(vtype="C",name="I(%s,%s)"%(t,p))
        I[0,p] = model.addVar(name="I(%s,%s)"%(0,p))

    for t in Ts:
        for p in P:
            # flow conservation constraints
            model.addCons(I[t-1,p] + x[t,p] == \
                            quicksum(phi[p,q]*x[t,q] for (p2,q) in phi if p2 == p) \
                            + I[t,p] + d[t,p],
                            "FlowCons(%s,%s)"%(t,p))

            # capacity connection constraints
            model.addCons(x[t,p] <= UB[t,p]*y[t,p], "ConstrUB(%s,%s)"%(t,p))

        # time capacity constraints
        for k in K:
            model.addCons(quicksum(a[t,k,p]*x[t,p] + g[t,p]*y[t,p] for p in P) <= M[t,k],
                            "TimeUB(%s,%s)"%(t,k))

    # initial inventory quantities
    for p in P:
        model.addCons(I[0,p] == 0, "InventInit(%s)"%(p))

    model.setObjective(\
        quicksum(f[t,p]*y[t,p] + c[t,p]*x[t,p] + h[t,p]*I[t,p] for t in Ts for p in P), \
        "minimize")

    model.data = y,x,I
    return model



def sum_phi(k,rho,pred,phi):
    for j in pred[k]:
        if (j,k) in rho:
            continue
        rho[j,k] = phi[j,k]
        sum_phi(j,rho,pred,phi)
        for i in pred[j]:
            if (i,k) in rho:
                rho[i,k] += rho[i,j] * phi[j,k]
            else:
                rho[i,k] = rho[i,j] * phi[j,k]
    return


def calc_rho(phi):
    v = set()
    for (i,j) in phi:
        v.add(i)
        v.add(j)
    pred,succ = {},{}
    for i in v:
        pred[i] = set()
        succ[i] = set()
    for (i,j) in phi:
        pred[j].add(i)
        succ[i].add(j)
    # set of vertices corresponding to end products:
    final = set(i for i in v if len(succ[i]) == 0)

    rho = {}
    for j in final:
        sum_phi(j,rho,pred,phi)
    return rho



def mils_echelon(T,K,P,f,g,c,d,h,a,M,UB,phi):
    """
    mils_echelon: echelon formulation for the multi-item, multi-stage lot-sizing problem

    Parameters:
        - T: number of periods
        - K: set of resources
        - P: set of items
        - f[t,p]: set-up costs (on period t, for product p)
        - g[t,p]: set-up times
        - c[t,p]: variable costs
        - d[t,p]: demand values
        - h[t,p]: holding costs
        - a[t,k,p]: amount of resource k for producing p in period t
        - M[t,k]: resource k upper bound on period t
        - UB[t,p]: upper bound of production time of product p in period t
        - phi[(i,j)]: units of i required to produce a unit of j (j parent of i)
    """
    rho = calc_rho(phi) # rho[(i,j)]: units of i required to produce a unit of j (j ancestor of i)

    model = Model("multi-stage lotsizing -- echelon formulation")

    y,x,E,H = {},{},{},{}
    Ts = range(1,T+1)
    for p in P:
        for t in Ts:
            y[t,p] = model.addVar(vtype="B", name="y(%s,%s)"%(t,p))
            x[t,p] = model.addVar(vtype="C", name="x(%s,%s)"%(t,p))
            H[t,p] = h[t,p] - sum([h[t,q]*phi[q,p] for (q,p2) in phi if p2 == p])
            E[t,p] = model.addVar(vtype="C", name="E(%s,%s)"%(t,p))        # echelon inventory
        E[0,p] = model.addVar(vtype="C", name="E(%s,%s)"%(0,p))    # echelon inventory

    for t in Ts:
        for p in P:
            # flow conservation constraints
            dsum = d[t,p] + sum([rho[p,q]*d[t,q] for (p2,q) in rho if p2 == p])
            model.addCons(E[t-1,p] + x[t,p] == E[t,p] + dsum, "FlowCons(%s,%s)"%(t,p))

            # capacity connection constraints
            model.addCons(x[t,p] <= UB[t,p]*y[t,p], "ConstrUB(%s,%s)"%(t,p))

        # time capacity constraints
        for k in K:
            model.addCons(quicksum(a[t,k,p]*x[t,p] + g[t,p]*y[t,p] for p in P) <= M[t,k],
                            "TimeUB(%s,%s)"%(t,k))


    # calculate echelon quantities
    for p in P:
        model.addCons(E[0,p] == 0, "EchelonInit(%s)"%(p))
        for t in Ts:
            model.addCons(E[t,p] >= quicksum(phi[p,q]*E[t,q] for (p2,q) in phi if p2 == p),
                            "EchelonLB(%s,%s)"%(t,p))

    model.setObjective(\
        quicksum(f[t,p]*y[t,p] + c[t,p]*x[t,p] + H[t,p]*E[t,p] for t in Ts for p in P), \
        "minimize")

    model.data = y,x,E
    return model


def make_data():
    """
    1..T: set of periods
    K: set of resources
    P: set of items
    f[t,p]: set-up costs
    g[t,p]: set-up times
    c[t,p]: variable costs
    d[t,p]: demand values
    h[t,p]: holding costs
    a[t,k,p]: amount of resource k for producing product p in period. t
    M[t,k]: resource upper bounds
    UB[t,p]: upper bound of production time of product p in period t
    phi[(i,j)] : units of i required to produce a unit of j (j parent of i)
    """
    T = 5
    K = [1]
    P = [1,2,3,4,5]
    _,f,g,c,d,h,UB = multidict({
        (1,1): [10, 1, 2, 0, 0.5, 24],
        (1,2): [10, 1, 2, 0, 0.5, 24],
        (1,3): [10, 1, 2, 0, 0.5, 24],
        (1,4): [10, 1, 2, 0, 0.5, 24],
        (1,5): [10, 1, 2, 0, 0.5, 24],
        (2,1): [10, 1, 2, 0, 0.5, 24],
        (2,2): [10, 1, 2, 0, 0.5, 24],
        (2,3): [10, 1, 2, 0, 0.5, 24],
        (2,4): [10, 1, 2, 0, 0.5, 24],
        (2,5): [10, 1, 2, 0, 0.5, 24],
        (3,1): [10, 1, 2, 0, 0.5, 24],
        (3,2): [10, 1, 2, 0, 0.5, 24],
        (3,3): [10, 1, 2, 0, 0.5, 24],
        (3,4): [10, 1, 2, 0, 0.5, 24],
        (3,5): [10, 1, 2, 0, 0.5, 24],
        (4,1): [10, 1, 2, 0, 0.5, 24],
        (4,2): [10, 1, 2, 0, 0.5, 24],
        (4,3): [10, 1, 2, 0, 0.5, 24],
        (4,4): [10, 1, 2, 0, 0.5, 24],
        (4,5): [10, 1, 2, 0, 0.5, 24],
        (5,1): [10, 1, 2, 0, 0.5, 24],
        (5,2): [10, 1, 2, 0, 0.5, 24],
        (5,3): [10, 1, 2, 0, 0.5, 24],
        (5,4): [10, 1, 2, 0, 0.5, 24],
        (5,5): [10, 1, 2, 5, 0.5, 24],
        })
    a = {
        (1,1,1): 1,
        (1,1,2): 1,
        (1,1,3): 1,
        (1,1,4): 1,
        (1,1,5): 1,
        (2,1,1): 1,
        (2,1,2): 1,
        (2,1,3): 1,
        (2,1,4): 1,
        (2,1,5): 1,
        (3,1,1): 1,
        (3,1,2): 1,
        (3,1,3): 1,
        (3,1,4): 1,
        (3,1,5): 1,
        (4,1,1): 1,
        (4,1,2): 1,
        (4,1,3): 1,
        (4,1,4): 1,
        (4,1,5): 1,
        (5,1,1): 1,
        (5,1,2): 1,
        (5,1,3): 1,
        (5,1,4): 1,
        (5,1,5): 1,
        }
    M = {
        (1,1): 15,
        (2,1): 15,
        (3,1): 15,
        (4,1): 15,
        (5,1): 15,
        }

    phi = {     # phi[(i,j)] : units of i required to produce a unit of j (j parent of i)
        (1,3):2,
        (2,3):3,
        (2,4):3/2.,
        (3,5):1/2.,
        (4,5):3
        }


    return T,K,P,f,g,c,d,h,a,M,UB,phi





def make_data_10():
    """
    1..T: set of periods
    K: set of resources
    P: set of items
    f[t,p]: set-up costs
    g[t,p]: set-up times
    c[t,p]: variable costs
    d[t,p]: demand values
    h[t,p]: holding costs
    a[t,k,p]: amount of resource k for producing product p in period. t
    M[t,k]: resource upper bounds
    UB[t,p]: upper bound of production time of product p in period t
    phi[(i,j)] : units of i required to produce a unit of j (j parent of i)
    """
    T = 5
    K = [1]
    P = [1,2,3,4,5,6,7,8,9,10]
    _, f, g, c, d, h, UB = multidict({
        (1,1): [10, 1, 2,  0, 0.5, 24],
        (1,2): [10, 1, 2,  0, 0.5, 24],
        (1,3): [10, 1, 2,  0, 0.5, 24],
        (1,4): [10, 1, 2,  0, 0.5, 24],
        (1,5): [10, 1, 2,  0, 0.5, 24],
        (1,6): [10, 1, 2,  0, 0.5, 24],
        (1,7): [10, 1, 2,  0, 0.5, 24],
        (1,8): [10, 1, 2,  0, 0.5, 24],
        (1,9): [10, 1, 2,  0, 0.5, 24],
        (1,10):[10, 1, 2,  0, 0.5, 24],
        (2,1): [10, 1, 2,  0, 0.5, 24],
        (2,2): [10, 1, 2,  0, 0.5, 24],
        (2,3): [10, 1, 2,  0, 0.5, 24],
        (2,4): [10, 1, 2,  0, 0.5, 24],
        (2,5): [10, 1, 2,  0, 0.5, 24],
        (2,6): [10, 1, 2,  0, 0.5, 24],
        (2,7): [10, 1, 2,  0, 0.5, 24],
        (2,8): [10, 1, 2,  0, 0.5, 24],
        (2,9): [10, 1, 2,  0, 0.5, 24],
        (2,10):[10, 1, 2,  0, 0.5, 24],
        (3,1): [10, 1, 2,  0, 0.5, 24],
        (3,2): [10, 1, 2,  0, 0.5, 24],
        (3,3): [10, 1, 2,  0, 0.5, 24],
        (3,4): [10, 1, 2,  0, 0.5, 24],
        (3,5): [10, 1, 2,  0, 0.5, 24],
        (3,6): [10, 1, 2,  0, 0.5, 24],
        (3,7): [10, 1, 2,  0, 0.5, 24],
        (3,8): [10, 1, 2,  0, 0.5, 24],
        (3,9): [10, 1, 2,  0, 0.5, 24],
        (3,10):[10, 1, 2,  0, 0.5, 24],
        (4,1): [10, 1, 2,  0, 0.5, 24],
        (4,2): [10, 1, 2,  0, 0.5, 24],
        (4,3): [10, 1, 2,  0, 0.5, 24],
        (4,4): [10, 1, 2,  0, 0.5, 24],
        (4,5): [10, 1, 2,  0, 0.5, 24],
        (4,6): [10, 1, 2,  0, 0.5, 24],
        (4,7): [10, 1, 2,  0, 0.5, 24],
        (4,8): [10, 1, 2,  0, 0.5, 24],
        (4,9): [10, 1, 2,  0, 0.5, 24],
        (4,10):[10, 1, 2,  0, 0.5, 24],
        (5,1): [10, 1, 2,  0, 0.5, 24],
        (5,2): [10, 1, 2,  0, 0.5, 24],
        (5,3): [10, 1, 2,  0, 0.5, 24],
        (5,4): [10, 1, 2,  0, 0.5, 24],
        (5,5): [10, 1, 2,  0, 0.5, 24],
        (5,6): [10, 1, 2,  0, 0.5, 24],
        (5,7): [10, 1, 2,  0, 0.5, 24],
        (5,8): [10, 1, 2,  0, 0.5, 24],
        (5,9): [10, 1, 2,  0, 0.5, 24],
        (5,10):[10, 1, 2,  5, 0.5, 24],
        })
    a = {
        (1,1,1): 1, (1,1,2): 1, (1,1,3): 1, (1,1,4): 1, (1,1,5): 1, (1,1,6): 1, (1,1,7): 1, (1,1,8): 1, (1,1,9): 1, (1,1,10): 1,
        (2,1,1): 1, (2,1,2): 1, (2,1,3): 1, (2,1,4): 1, (2,1,5): 1, (2,1,6): 1, (2,1,7): 1, (2,1,8): 1, (2,1,9): 1, (2,1,10): 1,
        (3,1,1): 1, (3,1,2): 1, (3,1,3): 1, (3,1,4): 1, (3,1,5): 1, (3,1,6): 1, (3,1,7): 1, (3,1,8): 1, (3,1,9): 1, (3,1,10): 1,
        (4,1,1): 1, (4,1,2): 1, (4,1,3): 1, (4,1,4): 1, (4,1,5): 1, (4,1,6): 1, (4,1,7): 1, (4,1,8): 1, (4,1,9): 1, (4,1,10): 1,
        (5,1,1): 1, (5,1,2): 1, (5,1,3): 1, (5,1,4): 1, (5,1,5): 1, (5,1,6): 1, (5,1,7): 1, (5,1,8): 1, (5,1,9): 1, (5,1,10): 1,
        }
    M = {
        (1,1): 25,
        (2,1): 25,
        (3,1): 25,
        (4,1): 25,
        (5,1): 25,
        (6,1): 25,
        (7,1): 25,
        (8,1): 25,
        (9,1): 25,
        (10,1):25,
        }

    phi = {     # phi[(i,j)] : units of i required to produce a unit of j (j parent of i)
        (1,2):1,
        (2,5):2,
        (3,4):3,
        (4,5):1,
        (5,6):1,
        (6,10):1/2.,
        (3,7):1,
        (7,8):3/2.,
        (8,9):3,
        (9,10):1
        }


    return T,K,P,f,g,c,d,h,a,M,UB,phi




if __name__ == "__main__":
    random.seed(1)

    # T,K,P,f,g,c,d,h,a,M,UB,phi = make_data()
    T,K,P,f,g,c,d,h,a,M,UB,phi = make_data_10()
    # print("periods",T
    # print("products:",P
    # print("resources:",K
    # print("demand",d
    # print("a (resource used)",a
    # print("g (setup time)",g
    # print("M",M
    # print("UB",UB
    # print("phi",phi
    # print("rho",calc_rho(phi)

    print("\n\nstandard model")
    model = mils_standard(T,K,P,f,g,c,d,h,a,M,UB,phi)
    # model.setParam("MIPGap",0.0)
    # model.write("lotsize_standard.lp")
    model.optimize()

    status = model.getStatus()
    if status == "unbounded":
        print("The model cannot be solved because it is unbounded")
    elif status == "optimal":
        print("Optimal value:", model.getObjVal())
        standard = model.getObjVal()
        y,x,I = model.data
        print("%7s%7s%7s%7s%12s%12s" % ("prod","t","dem","y","x","I"))
        for p in P:
            print("%7s%7s%7s%7s%12s%12s" % (p,0,"-","-","-",round(model.getVal(I[0,p]),5)))
            for t in range(1,T+1):
                print("%7s%7s%7s%7s%12s%12s" % (p,t,d[t,p],int(model.getVal(y[t,p])+0.5),round(model.getVal(x[t,p]),5),round(model.getVal(I[t,p]),5)))
        print("\n")
        for k in K:
            for t in range(1,T+1):
                print("resource %3s used in period %s: %12s / %-9s" % (k,t,sum(a[t,k,p]*model.getVal(x[t,p])+g[t,p]*model.getVal(y[t,p]) for p in P),M[t,k]))
    elif status != "unbounded" and status != "infeasible":
        print("Optimization was stopped with status",status)
    else:
        print("The model is infeasible")
    #    model.computeIIS()
    #    print("\nThe following constraint(s) cannot be satisfied:")
    #    for cnstr in model.getConstrs():
    #      if cnstr.IISConstr:
    #        print(cnstr.ConstrName)


    print("\n\nechelon model")
    model = mils_echelon(T,K,P,f,g,c,d,h,a,M,UB,phi)
    model.setRealParam("limits/gap", 0.0)
    # model.write("lotsize_echelon.lp")
    model.optimize()

    status = model.getStatus()
    if status == "unbounded":
        print("The model cannot be solved because it is unbounded")
    elif status == "optimal":
        print("Opt.value=",model.getObjVal())
        y,x,E = model.data
        print("%7s%7s%7s%7s%12s%12s" % ("t","prod","dem","y","x","E"))
        for p in P:
            print("%7s%7s%7s%7s%12s%12s" % ("t",p,"-","-","-",round(model.getVal(E[0,p]),5)))
            for t in range(1,T+1):
                print("%7s%7s%7s%7s%12s%12s" %\
                      (t,p,d[t,p],int(model.getVal(y[t,p])+0.5),round(model.getVal(x[t,p]),5),round(model.getVal(E[t,p]),5)))
        print("\n")
        for k in K:
            for t in range(1,T+1):
                print("resource %3s used in period %s: %12s / %-9s" % \
                      (k,t,sum(a[t,k,p]*model.getVal(x[t,p]) + g[t,p]*model.getVal(y[t,p]) for p in P),M[t,k]))
        print(model.getObjVal() - standard)
        assert abs(model.getObjVal() - standard) < 1.e-6
    elif status != "unbounded" and status != "infeasible":
        print("Optimization was stopped with status",status)
    else:
        print("The model is infeasible")
     #   model.computeIIS()
     #   print("\nThe following constraint(s) cannot be satisfied:")
     #   for cnstr in model.getConstrs():
     #     if cnstr.IISConstr:
     #       print(cnstr.ConstrName)
