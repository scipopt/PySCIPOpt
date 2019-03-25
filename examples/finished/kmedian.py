##@file kmedian.py
#@brief model for solving the k-median problem.
"""
minimize the total (weighted) travel cost for servicing
a set of customers from k facilities.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import math
import random
from pyscipopt import Model, quicksum, multidict

def kmedian(I,J,c,k):
    """kmedian -- minimize total cost of servicing customers from k facilities
    Parameters:
        - I: set of customers
        - J: set of potential facilities
        - c[i,j]: cost of servicing customer i from facility j
        - k: number of facilities to be used
    Returns a model, ready to be solved.
    """

    model = Model("k-median")
    x,y = {},{}

    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in I:
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))

    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == 1, "Assign(%s)"%i)
        for j in J:
            model.addCons(x[i,j] <= y[j], "Strong(%s,%s)"%(i,j))
    model.addCons(quicksum(y[j] for j in J) == k, "Facilities")

    model.setObjective(quicksum(c[i,j]*x[i,j] for i in I for j in J), "minimize")
    model.data = x,y

    return model


def distance(x1,y1,x2,y2):
    """return distance of two points"""
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def make_data(n,m,same=True):
    """creates example data set"""
    if same == True:
        I = range(n)
        J = range(m)
        x = [random.random() for i in range(max(m,n))]    # positions of the points in the plane
        y = [random.random() for i in range(max(m,n))]
    else:
        I = range(n)
        J = range(n,n+m)
        x = [random.random() for i in range(n+m)]    # positions of the points in the plane
        y = [random.random() for i in range(n+m)]
    c = {}
    for i in I:
        for j in J:
            c[i,j] = distance(x[i],y[i],x[j],y[j])

    return I,J,c,x,y


if __name__ == "__main__":
    import sys
    random.seed(67)
    n = 200
    m = n
    I,J,c,x_pos,y_pos = make_data(n,m,same=True)
    k = 20
    model = kmedian(I,J,c,k)
    # model.Params.Threads = 1
    model.optimize()
    EPS = 1.e-6
    x,y = model.data
    edges = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > EPS]
    facilities = [j for j in y if model.getVal(y[j]) > EPS]

    print("Optimal value:",model.getObjVal())
    print("Selected facilities:", facilities)
    print("Edges:", edges)
    print("max c:", max([c[i,j] for (i,j) in edges]))

    try: # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        facilities = set(j for j in J if model.getVal(y[j]) > EPS)
        other = set(j for j in J if j not in facilities)
        client = set(i for i in I if i not in facilities and i not in other)
        G.add_nodes_from(facilities)
        G.add_nodes_from(client)
        G.add_nodes_from(other)
        for (i,j) in edges:
            G.add_edge(i,j)

        position = {}
        for i in range(len(x_pos)):
            position[i] = (x_pos[i],y_pos[i])

        NX.draw(G,position,with_labels=False,node_color="w",nodelist=facilities)
        NX.draw(G,position,with_labels=False,node_color="c",nodelist=other,node_size=50)
        NX.draw(G,position,with_labels=False,node_color="g",nodelist=client,node_size=50)
        P.show()
    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")
