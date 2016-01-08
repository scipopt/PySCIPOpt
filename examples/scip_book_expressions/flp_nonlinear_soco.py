# todo
"""
flp_nonlinear.py:  soco approach to the capacitated facility location problem

minimize the total (weighted) travel cost from n customers to a
given set of facilities, with fixed costs and limited capacities;
costs are nonlinear (square root of the total quantity serviced
by a facility).

Approach: use a second-order cone optimization formulation.

Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""
import math
import random
from pyscipopt import Model, quicksum, multidict

def flp_nonlinear_soco(I,J,d,M,f,c):
    """flp_nonlinear_soco --  use
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for product i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
    Returns a model, ready to be solved.
    """
    model = Model("nonlinear flp -- soco formulation")

    x,X,u = {},{},{}
    for j in J:
        X[j] = model.addVar(ub=M[j], vtype="C", name="X(%s)"%j) # for sum_i x_ij
        u[j] = model.addVar(vtype="C", name="u(%s)"%j) # for replacing sqrt sum_i x_ij in soco
        for i in I:
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))  # i's demand satisfied from j
    
    # constraints for customer's demand satisfaction
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == 1, "Demand(%s)"%i)

    for j in J:
        model.addCons(quicksum(d[i]*x[i,j] for i in I) == X[j], "Capacity(%s)"%j)
        model.addQConstr(quicksum(f[j]**2*d[i]*x[i,j]*x[i,j] for i in I) <= u[j]*u[j], "SOC(%s)"%j)

    model.setObjective(quicksum(u[j] for j in J) +\
                       quicksum(c[i,j]*d[i]*x[i,j] for j in J for i in I),\
                       "minimize")

    model.data = x,u
    return model


if __name__ == "__main__":
    from flp_nonlinear import distance,make_data,example,read_orlib,read_cortinhal
    # I,J,d,M,f,c,x_pos,y_pos = example()
    K = 100
    random.seed(1)
    n = 25
    m = 5
    I,J,d,M,f,c,x_pos,y_pos = make_data(n,m)
    # I,J,d,M,f,c,x_pos,y_pos = read_orlib("DATA/cap41.txt.gz")
    # I,J,d,M,f,c,x_pos,y_pos = read_cortinhal("DATA/8_25/A8_25_11.DAT.gz")

    print("demand:",d)
    print("cost:",c)
    print("fixed:",f)
    print("capacity:",M)
    print("x:",x_pos)
    print("y:",y_pos)
    print("number of intervals:",K)

    from flp_nonlinear import flp_nonlinear_sos
    print("\n\n\nflp: model with SOS constraints")
    model = flp_nonlinear_sos(I,J,d,M,f,c,K)
    x,X,F = model.data
    model.hideOutput = 1 # silent/verbose mode
    model.optimize()
    objSOS = model.getObjVal()
    print("obj:",objSOS)

    EPS = 1.e-4
    # assert abs(objSOS-objSOCO) < EPS
    edges = []
    for (i,j) in sorted(x):
        if model.getVal(x[i,j]) > EPS:
           edges.append((i,j))
    print("\n\n\nflp: model with piecewise linear approximation of cost function")
    print("obj:",model.getObjVal(),"\nedges",sorted(edges))
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


    print("\n\n\nflp: soco model")
    model = flp_nonlinear_soco(I,J,d,M,f,c)
    x,u = model.data
    model.hideOutput = 1 # silent/verbose mode
    model.optimize()
    objSOCO = model.getObjVal()
    print("obj:",objSOCO)


    EPS = 1.e-4
    # assert abs(objSOS-objSOCO) < EPS
    edges = []
    for (i,j) in sorted(x):
        if model.getVal(x[i,j]) > EPS:
           edges.append((i,j))
    print("\n\n\nflp: model with piecewise linear approximation of cost function")
    print("obj:",model.getObjVal(),"\nedges",sorted(edges))

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
