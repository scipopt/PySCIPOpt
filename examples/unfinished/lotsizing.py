"""
lotsizing.py:  solve the multi-item lot-sizing problem.

Approaches:
    - mils: solve the problem using the standard formulation
    - mils_fl: solve the problem using the facility location (tighten) formulation

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import random
from pyscipopt import Model, quicksum

def mils(T,P,f,g,c,d,h,M):
    """
    mils: standard formulation for the multi-item lot-sizing problem
    Parameters:
        - T: number of periods
        - P: set of products
        - f[t,p]: set-up costs (on period t, for product p)
        - g[t,p]: set-up times
        - c[t,p]: variable costs
        - d[t,p]: demand values
        - h[t,p]: holding costs
        - M[t]: resource upper bound on period t
    Returns a model, ready to be solved.
    """
    def mils_callback(model,where):
        # remember to set     model.params.DualReductions = 0     before using!
        if where != GRB.Callback.MIPSOL and where != GRB.Callback.MIPNODE:
            return
        for p in P:
            for ell in Ts:
                lhs = 0
                S,L = [],[]
                for t in range(1,ell+1):
                    yt = model.cbGetSolution(y[t,p])
                    xt = model.cbGetSolution(x[t,p])
                    if D[t,ell,p]*yt < xt:
                        S.append(t)
                        lhs += D[t,ell,p]*yt
                    else:
                        L.append(t)
                        lhs += xt
                if lhs < D[1,ell,p]:
                    # add cutting plane constraint
                    model.cbLazy(quicksum(x[t,p] for t in L) +\
                                 quicksum(D[t,ell,p] * y[t,p] for t in S)
                                 >= D[1,ell,p])
        return

    model = Model("standard multi-item lotsizing")

    y,x,I = {},{},{}
    Ts = range(1,T+1)
    for p in P:
        for t in Ts:
            y[t,p] = model.addVar(vtype="B", name="y(%s,%s)"%(t,p))
            x[t,p] = model.addVar(vtype="C", name="x(%s,%s)"%(t,p))
            I[t,p] = model.addVar(vtype="C", name="I(%s,%s)"%(t,p))
        I[0,p] = 0

    for t in Ts:
        # time capacity constraints
        model.addCons(quicksum(g[t,p]*y[t,p] + x[t,p] for p in P) <= M[t], "TimeUB(%s)"%(t))

        for p in P:
            # flow conservation constraints
            model.addCons(I[t-1,p] + x[t,p] == I[t,p] + d[t,p], "FlowCons(%s,%s)"%(t,p))

            # capacity connection constraints
            model.addCons(x[t,p] <= (M[t]-g[t,p])*y[t,p], "ConstrUB(%s,%s)"%(t,p))

            # tighten constraints
            model.addCons(x[t,p] <= d[t,p]*y[t,p] + I[t,p], "Tighten(%s,%s)"%(t,p))

    model.setObjective(\
        quicksum(f[t,p]*y[t,p] + c[t,p]*x[t,p] + h[t,p]*I[t,p] for t in Ts for p in P),\
        "minimize")

    # compute D[i,j,p] = sum_{t=i}^j d[t,p]
    D = {}
    for p in P:
        for t in Ts:
            s = 0
            for j in range(t,T+1):
                s += d[j,p]
                D[t,j,p] = s

    model.data = y,x,I
    return model,mils_callback


def mils_fl(T,P,f,g,c,d,h,M):
    """
    mils_fl: facility location formulation for the multi-item lot-sizing problem

    Requires more variables, but gives a better solution because LB is
    better than the standard formulation.  It can be used as a
    heuristic method that is sometimes better than relax-and-fix.

    Parameters:
        - T: number of periods
        - P: set of products
        - f[t,p]: set-up costs (on period t, for product p)
        - g[t,p]: set-up times
        - c[t,p]: variable costs
        - d[t,p]: demand values
        - h[t,p]: holding costs
        - M[t]:   resource upper bound on period t
    Returns a model, ready to be solved.
    """
    Ts = range(1,T+1)

    model = Model("multi-item lotsizing -- facility location formulation")

    y,X = {},{}
    for p in P:
        for t in Ts:
            y[t,p] = model.addVar(vtype="B", name="y(%s,%s)"%(t,p))
            for s in range(1,t+1):
                X[s,t,p] = model.addVar(name="X(%s,%s,%s)"%(s,t,p))


    for t in Ts:
        # capacity constraints
        model.addCons(quicksum(X[t,s,p] for s in range(t,T+1) for p in P) + \
                        quicksum(g[t,p]*y[t,p] for p in P) <= M[t],
                        "Capacity(%s)"%(t))

        for p in P:
            # demand satisfaction constraints
            model.addCons(quicksum(X[s,t,p] for s in range(1,t+1)) == d[t,p], "Demand(%s,%s)"%(t,p))

            # connection constraints
            for s in range(1,t+1):
                model.addCons(X[s,t,p] <= d[t,p] * y[s,p], "Connect(%s,%s,%s)"%(s,t,p))

    C = {} # variable costs plus holding costs
    for p in P:
        for s in Ts:
            sumC = 0
            for t in range(s,T+1):
                C[s,t,p] = (c[s,p] + sumC)
                sumC += h[t,p]

    model.setObjective(quicksum(f[t,p]*y[t,p] for t in Ts for p in P) + \
                       quicksum(C[s,t,p]*X[s,t,p] for t in Ts for p in P for s in range(1,t+1)),
                       "minimize")


    model.data = y,X
    model.write("tmp.lp")
    return model



def trigeiro(T,N,factor):
    """
    Data generator for the multi-item lot-sizing problem
    it uses a simular algorithm for generating the standard benchmarks in:
    "Capacitated Lot Sizing with Setup Times" by
    William W. Trigeiro, L. Joseph Thomas, John O. McClain
    MANAGEMENT SCIENCE
    Vol. 35, No. 3, March 1989, pp. 353-366

    Parameters:
        - T: number of periods
        - N: number of products
        - factor: value for controlling constraining factor of capacity:
            - 0.75:  lightly-constrained instances
            - 1.10:  constrained instances
    """
    P = range(1,N+1)
    f,g,c,d,h,M = {},{},{},{},{},{}

    sumT = 0
    for t in range(1,T+1):
        for p in P:
            # capacity used per unit production: 1, except for
            # except for specific instances with random value in [0.5, 1.5]
            # (not tackled in our model)

            # setup times
            g[t,p] = 10 * random.randint(1,5)   # 10, 50: trigeiro's values

            # set-up costs
            f[t,p] = 100 * random.randint(1,10) # checked from Wolsey's instances
            c[t,p] = 0                          # variable costs

            # demands
            d[t,p] = 100+random.randint(-25,25) # checked from Wolsey's instances
            if t <= 4:
                if random.random() < .25:       # trigeiro's parameter
                    d[t,p] = 0
            sumT += g[t,p] + d[t,p]             # sumT is the total capacity usage in the lot-for-lot solution
            h[t,p] = random.randint(1,5)        # holding costs; checked from Wolsey's instances

    for t in range(1,T+1):
        M[t] = int(float(sumT)/float(T)/factor)

    return P,f,g,c,d,h,M



if __name__ == "__main__":

    # test only a subset of instances
    T,N,factor = 15,6,0.75
    P,f,g,c,d,h,M = trigeiro(T,N,factor)
    print("\n\n\nStandard formulation + cutting plane ======================")
    model,mils_callback = mils(T,P,f,g,c,d,h,M)
    model.setBoolParam("misc/allowstrongdualreds", 0)
    model.optimize(mils_callback)
    y,x,I = model.data

    print("Optimal value:",model.getObjVal())
    standard = model.getObjVal()
    print("%7s%7s%7s%7s%12s%12s" % ("prod","t","dem","y","x","I"))
    for p in P:
        for t in range(1,T+1):
            print("%7s%7s%7s%7s%12s%12s" % \
                  (p,t,d[t,p],int(y[t,p].X+0.5),round(x[t,p].X,5),round(I[t,p].X,5)))

    exit(0)
    # to test all instances: remove 'exit' above

    # sizes: 4 to 36 items, 15 to 30 time periods;   table 2: T = 15, 30, N = 6, 12, 24
    # all costs constant over time
    for T in [15,30]:
        for N in [6,12,24]:
            for factor in [0.75,1.0,1.1]:
                # 0.75: lightly-constrained instances
                # 1.1:  constrained instances

                random.seed(1)
                # T,P = 10,5   # number of periods and number of products
                # # T,P = 30,24  # number of periods and number of products
                # P,f,g,c,d,h,M = make_data(T,P)
                P,f,g,c,d,h,M = trigeiro(T,N,factor)
                #for i in d:
                #    d[i] = 100
                # print("f",f
                # print("g",g
                # print("c",c
                # print("d",d
                # print("h",h
                # print("M",M
                print("\n\n\n*** Trigeiros instance for %s periods, %s products, %s capacity factor" %\
                      (T,N,factor))

                # standard formulation
                print("\n\n\nStandard formulation======================")
                model,_ = mils(T,P,f,g,c,d,h,M)
                model.optimize()
                y,x,I = model.data

                status = model.getStatus()
                if status == "unbounded":
                    print("Unbounded instance")
                elif status == "optimal":
                    print("Optimal value:",model.getObjVal())
                    standard = model.getObjVal()
                    print("%7s%7s%7s%7s%12s%12s" % ("prod","t","dem","y","x","I"))
                    for p in P:
                        for t in range(1,T+1):
                            print("%7s%7s%7s%7s%12s%12s" % \
                                  (p,t,d[t,p],int(y[t,p].X+0.5),round(x[t,p].X,5),round(I[t,p].X,5)))
                elif status != "unbounded" and status != "infeasible":
                    print("Optimization stopped with status",status)
                else:
                    print("Instance infeasible")


                # standard formulation + cutting plane
                print("\n\n\nStandard formulation + cutting plane ======================")
                model,mils_callback = mils(T,P,f,g,c,d,h,M)
                model.params.DualReductions = 0
                model.optimize(mils_callback)
                y,x,I = model.data

                status = model.getStatus()
                if status == "unbounded":
                    print("Unbounded instance")
                elif status == "optimal":
                    print("Optimal value:",model.getObjVal())
                    standard = model.getObjVal()
                    print("%7s%7s%7s%7s%12s%12s" % ("prod","t","dem","y","x","I"))
                    for p in P:
                        for t in range(1,T+1):
                            print("%7s%7s%7s%7s%12s%12s" % \
                                  (p,t,d[t,p],int(y[t,p].X+0.5),round(x[t,p].X,5),round(I[t,p].X,5)))
                    assert abs(model.objval - standard) < 1.e-6
                elif status != "unbounded" and status != "infeasible":
                    print("Optimization stopped with status",status)
                else:
                    print("Instance infeasible")


                #  facility location formulation
                print("\n\n\nFacility location formulation======================")
                model = mils_fl(T,P,f,g,c,d,h,M)
                model.optimize()
                y,X = model.data

                status = model.getStatus()
                if status == "unbounded":
                    print("Unbounded instance")
                elif status == "optimal":
                    print("Optimal value:",model.getObjVal())
                    print("%7s%7s%7s%7s%12s%12s" % ("prod","t","dem","y","sum(X)","inv.mov"))
                    for p in P:
                        for t in range(1,T+1):
                            xd = sum(X[t,s,p].X for s in range(t,T+1))
                            I = xd - d[t,p]
                            print("%7s%7s%7s%7s%12s%12s" % \
                                  (p,t,d[t,p],int(y[t,p].X+0.5),round(xd,5),round(I)))
                    assert abs(model.objval - standard) < 1.e-6
                elif status != "unbounded" and status != "infeasible":
                    print("Optimization stopped with status",status)
                else:
                    print("Instance infeasible")
