# todo
"""
flp_nonlinear.py:  piecewise linear model for the capacitated facility location problem

minimize the total (weighted) travel cost from n customers to a
given set of facilities, with fixed costs and limited capacities;
costs are nonlinear (square root of the total quantity serviced
by a facility).

Approaches: use
  - convex combination
  - multiple selection
formulations defined in 'piecewise.py'.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
from pyscipopt import Model, quicksum, multidict
from piecewise import *

def flp_nonlinear_mselect(I,J,d,M,f,c,K):
    """flp_nonlinear_mselect --  use multiple selection model
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- piecewise linear version with multiple selection")
    
    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j

    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,z = {},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],z[j] = mult_selection(model,a[j],b[j])
        X[j].ub = M[j]
        # for i in I:
        #     model.addCons(
        #         x[i,j] <= \
        #         quicksum(min(d[i],a[j][k+1]) * z[j][k] for k in range(K)),\
        #         "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")

    model.data = x,X,F
    return model



def flp_nonlinear_cc_dis_strong(I,J,d,M,f,c,K):
    """flp_nonlinear_bin --  use convex combination model, with binary variables
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- piecewise linear version with convex combination")
    
    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j
    

    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,z = {},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],z[j] = convex_comb_dis(model,a[j],b[j])
        X[j].ub = M[j]
        for i in I:
            model.addCons(
                x[i,j] <= \
                quicksum(min(d[i],a[j][k+1]) * z[j][k] for k in range(K)),\
                "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")

    model.data = x,X,F
    return model



def flp_nonlinear_cc_dis(I,J,d,M,f,c,K):
    """flp_nonlinear_bin --  use convex combination model, with binary variables
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- piecewise linear version with convex combination")
    
    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j

    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,z = {},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],z[j] = convex_comb_dis(model,a[j],b[j])
        X[j].ub = M[j]
        # for i in I:
        #     model.addCons(
        #         x[i,j] <= \
        #         quicksum(min(d[i],a[j][k+1]) * z[j][k] for k in range(K)),\
        #         "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")

    model.data = x,X,F
    return model



def flp_nonlinear_cc_dis_log(I,J,d,M,f,c,K):
    """flp_nonlinear_cc_dis_log --  convex combination model with logarithmic number of binary variables
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- convex combination model with logarithmic number of binary variables")

    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j
    
    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,yL,yR = {},{},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],yL[j],yR[j] = convex_comb_dis_log(model,a[j],b[j])
        X[j].ub = M[j]
        # for i in I:
        #     model.addCons(
        #         x[i,j] <= \
        #         quicksum(min(d[i],a[j][k+1]) * (yL[j][k]+yR[j][k]) for k in range(K)),\
        #         "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")
 
    model.data = x,X,F
    return model



def flp_nonlinear_cc_agg(I,J,d,M,f,c,K):
    """flp_nonlinear_cc_agg --  aggregated convex combination model
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- piecewise linear aggregated convex combination")

    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j
    
    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,z = {},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],z[j] = convex_comb_agg(model,a[j],b[j])
        X[j].ub = M[j]
        # for i in I:
        #     model.addCons(
        #         x[i,j] <= \
        #         quicksum(min(d[i],a[j][k+1]) * z[j][k] for k in range(K)),\
        #         "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")

    
    model.data = x,X,F
    return model



def flp_nonlinear_cc_agg_log(I,J,d,M,f,c,K):
    """flp_nonlinear_cc_agg_logg --  aggregated convex combination model, with log. binary variables
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- piecewise linear version with convex combination")

    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j
    
    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,y = {},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],y[j] = convex_comb_agg_log(model,a[j],b[j])
        X[j].ub = M[j]
        # for i in I:
        #     model.addCons(
        #         x[i,j] <= \
        #         quicksum(min(d[i],a[j][k+1]) * (y[j][k]+y[j][k+1]) for k in range(K)),\
        #         "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")

    model.data = x,X,F
    return model



def flp_nonlinear_sos(I,J,d,M,f,c,K):
    """flp_nonlinear_sos --  use model with SOS constraints
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
        - K: number of linear pieces for approximation of non-linear cost function
    Returns a model, ready to be solved.
    """
    a,b = {},{}
    for j in J:
        U = M[j]
        L = 0
        width = U/float(K)
        a[j] = [k*width for k in range(K+1)]
        b[j] = [f[j]*math.sqrt(value) for value in a[j]]

    model = Model("nonlinear flp -- use model with SOS constraints")
    
    x = {}
    for j in J:
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j
    
    # total volume transported from plant j, corresponding (linearized) cost, selection variable:
    X,F,z = {},{},{}
    for j in J:
        # add constraints for linking piecewise linear part:
        X[j],F[j],z[j] = convex_comb_sos(model,a[j],b[j])
        X[j].ub = M[j]
        # for i in I:
        #     model.addCons(
        #         x[i,j] <= \
        #         quicksum(min(d[i],a[j][k+1]) * (z[j][k] + z[j][k+1])\
        #                  for k in range(len(a[j])-1)),
        #         "Strong(%s,%s)"%(i,j))

    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) == X[j], "Capacity(%s)"%j)

    model.setObjective(quicksum(F[j] for j in J) +\
                       quicksum(c[i,j]*x[i,j] for j in J for i in I),\
                       "minimize")
    
    model.data = x,X,F
    return model




def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def make_data(n,m,same=True):
    x,y = {},{}
    if same == True:
        I = range(1,n+1)
        J = range(1,m+1)
        for i in range(1,1+max(m,n)):    # positions of the points in the plane
            x[i] = random.random()
            y[i] = random.random()
    else:
        I = range(1,n+1)
        J = range(n+1,n+m+1)
        for i in I:    # positions of the points in the plane
            x[i] = random.random()
            y[i] = random.random()
        for j in J:    # positions of the points in the plane
            x[j] = random.random()
            y[j] = random.random()

    f,c,d,M = {},{},{},{}
    total_demand = 0.
    for i in I:
        for j in J:
            c[i,j] = int(100*distance(x[i],y[i],x[j],y[j])) + 1
        d[i] = random.randint(1,10)
        total_demand += d[i]

    total_cap = 0.
    r = {}
    for j in J:
        r[j] = random.randint(0,m)
        f[j] = random.randint(100,100+r[j]*m)
        M[j] = 1 + 100+r[j]*m - f[j]
        # M[j] = int(total_demand/m) + random.randint(1,m)
        total_cap += M[j]
    for j in J:
        M[j] = int(M[j] * total_demand / total_cap + 1) + random.randint(0,r[j])
        # print("%s\t%s\t%s" % (j,f[j],M[j])

    # print("demand:",total_demand
    # print("capacity:",sum([M[j] for j in J])

    return I,J,d,M,f,c,x,y


def example():
    I,d = multidict({1:80, 2:270, 3:250, 4:160, 5:180})                # demand
    J,M,f = multidict({10:[500,100], 11:[500,100], 12:[500,100]})     # capacity, fixed costs
    c = {(1,10):4,  (1,11):6,  (1,12):9,           # transportation costs
         (2,10):5,  (2,11):4,  (2,12):7,
         (3,10):6,  (3,11):3,  (3,12):4,
         (4,10):8,  (4,11):5,  (4,12):3,
         (5,10):10, (5,11):8,  (5,12):4,
         }
    x_pos = {1:0, 2:0, 3:0, 4:0, 5:0, 10:2, 11:2, 12:2}    # positions of the points in the plane
    y_pos = {1:2, 2:1, 3:0, 4:-1, 5:-2, 10:1, 11:0, 12:-1}
    return I,J,d,M,f,c,x_pos,y_pos



if __name__ == "__main__":
    # I,J,d,M,f,c,x_pos,y_pos = example()
    random.seed(1)
    n = 25
    m = 5
    I,J,d,M,f,c,x_pos,y_pos = make_data(n,m,same=False)
    # from flp_make_data import read_orlib,read_cortinhal
    # I,J,d,M,f,c,x_pos,y_pos = read_orlib("DATA/ORLIB/cap101.txt.gz")
    # I,J,d,M,f,c,x_pos,y_pos = read_cortinhal("DATA/8_25/A8_25_11.DAT")
    # I,J,d,M,f,c,x_pos,y_pos = example()
    K = 4
    print("demand:",d)
    print("cost:",c)
    print("fixed:",f)
    print("capacity:",M)
    # print("x:",x_pos
    # print("y:",y_pos
    print("number of intervals:",K)

    print("\n\n\nflp: multiple selection")
    model = flp_nonlinear_mselect(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput() # silent/verbose mode
    model.optimize()
    objMS = model.getObjVal()
    print("Obj.",objMS)

    print("\n\n\nflp: convex combination with binary variables")
    model = flp_nonlinear_cc_dis(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput() # silent/verbose mode
    model.optimize()
    objCC = model.getObjVal()
    print("Obj.",objCC)

    print("\n\n\nflp: convex combination with logarithmic number of binary variables")
    model = flp_nonlinear_cc_dis_log(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput() # silent/verbose mode
    model.optimize()
    objLOG = model.getObjVal()
    print("Obj.",objLOG)

    print("\n\n\nflp: model with SOS constraints")
    model = flp_nonlinear_sos(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput() # silent/verbose mode
    model.optimize()
    objSOS = model.getObjVal()
    print("Obj.",objSOS)

    print("\n\n\nflp: aggregated CC model")
    model = flp_nonlinear_cc_agg(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput() # silent/verbose mode
    model.optimize()
    objND = model.getObjVal()
    print("Obj.",objND)

    print("\n\n\nflp: aggregated CC model, log number variables")
    model = flp_nonlinear_cc_agg_log(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput() # silent/verbose mode
    model.optimize()
    objNDlog = model.getObjVal()
    print("Obj.",objNDlog)

    EPS = 1.e-4
    assert abs(objCC-objMS)<EPS and abs(objLOG-objMS)<EPS and abs(objSOS-objMS)<EPS\
           and abs(objSOS-objND)<EPS and abs(objSOS-objNDlog)<EPS
    edges = []
    flow = {}
    for (i,j) in sorted(x):
        if model.getVal(x[i,j]) > EPS:
           edges.append((i,j))
           flow[(i,j)] = model.getVal(x[i,j])
    print("\n\n\nflp: model with piecewise linear approximation of cost function")
    print("Obj.",model.getObjVal(),"\nedges",sorted(edges))
    print("flows:",flow)

    if x_pos == None:
        exit(0)

    try: # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        facilities = J
        client = I
        G.add_nodes_from(facilities)
        G.add_nodes_from(client)
        for (i,j) in edges:
            G.add_edge(i,j)

        position = {}
        for i in I + J:
            position[i] = (x_pos[i],y_pos[i])

        NX.draw(G,position,node_color="g",nodelist=client)
        NX.draw(G,position,node_color="y",nodelist=facilities)
        P.show()
    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")
