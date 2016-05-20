"""
kcenter_binary_search.py:  use bisection for solving the k-center problem

bisects the interval [0, max facility-customer distance] until finding a
distance such that all customers are covered, but decreasing that distance
by a small amount delta would leave some uncovered.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

def kcover(I,J,c,k):
    """kcover -- minimize the number of uncovered customers from k facilities.
    Parameters:
        - I: set of customers
        - J: set of potential facilities
        - c[i,j]: cost of servicing customer i from facility j
        - k: number of facilities to be used
    Returns a model, ready to be solved.
    """

    model = Model("k-center")

    z,y,x = {},{},{}
    for i in I:
        z[i] = model.addVar(vtype="B", name="z(%s)"%i, obj=1)
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in I:
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))

    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) + z[i] == 1, "Assign(%s)"%i)
        for j in J:
            model.addCons(x[i,j] <= y[j], "Strong(%s,%s)"%(i,j))

    model.addCons(sum(y[j] for j in J) == k, "k_center")
    model.data = x,y,z

    return model


def solve_kcenter(I,J,c,k,delta):
    """solve_kcenter -- locate k facilities minimizing distance of most distant customer.
    Parameters:
        I - set of customers
        J - set of potential facilities
        c[i,j] - cost of servicing customer i from facility j
        k - number of facilities to be used
        delta - tolerance for terminating bisection
    Returns:
        - list of facilities to be used
        - edges linking them to customers
    """

    model = kcover(I,J,c,k)
    x,y,z = model.data

    facilities,edges = [],[]
    LB = 0
    UB = max(c[i,j] for (i,j) in c)
    model.setObjlimit(0.1)
    while UB-LB > delta:
        theta = (UB+LB) / 2.
        # print "\n\ncurrent theta:", theta
        for j in J:
            for i in I:
                if c[i,j]>theta:
                    model.chgVarUb(x[i,j], 0.0)
                else:
                    model.chgVarUb(x[i,j], 1.0)

        # model.Params.OutputFlag = 0 # silent mode
        model.setObjlimit(.1)

        model.optimize()

        if model.getStatus == "optimal":
            # infeasibility = sum([z[i].X for i in I])
            # print "infeasibility=",infeasibility
            UB = theta
            facilities = [j for j in y if model.getVal(y[j]) > .5]
            edges = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > .5]
            # print "updated solution:"
            # print "facilities",facilities
            # print "edges",edges
        else:   # infeasibility > 0:
            LB = theta

    return facilities,edges



import math
import random
def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def make_data(n,m,same=True):
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
    random.seed(67)
    n = 200
    m = n
    I,J,c,x_pos,y_pos = make_data(n,m,same=True)
    k = 20
    delta = 1.e-4
    facilities,edges = solve_kcenter(I,J,c,k,delta)
    print("Selected facilities:", facilities)
    print("Edges:", edges)
    print("Max distance from a facility to a customer: ", max([c[i,j] for (i,j) in edges]))

    try: # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        facilities = set(facilities)
        unused = set(j for j in J if j not in facilities)
        client = set(i for i in I if i not in facilities and i not in unused)
        G.add_nodes_from(facilities)
        G.add_nodes_from(client)
        G.add_nodes_from(unused)
        for (i,j) in edges:
            G.add_edge(i,j)

        position = {}
        for i in range(len(x_pos)):
            position[i] = (x_pos[i],y_pos[i])

        NX.draw(G,position,with_labels=False,node_color="w",nodelist=facilities)
        NX.draw(G,position,with_labels=False,node_color="c",nodelist=unused,node_size=50)
        NX.draw(G,position,with_labels=False,node_color="g",nodelist=client,node_size=50)
        P.show()
    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")
