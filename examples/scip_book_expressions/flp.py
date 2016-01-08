"""
flp.py:  model for solving the capacitated facility location problem

minimize the total (weighted) travel cost from n customers
to some facilities with fixed costs and capacities.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

def flp(I,J,d,M,f,c):
    """flp -- model for the capacitated facility location problem
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
    Returns a model, ready to be solved.
    """

    model = Model("flp")

    x,y = {},{}
    for j in J:
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in I:
            x[i,j] = model.addVar(vtype="C", name="x(%s,%s)"%(i,j))

    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in M:
        model.addCons(quicksum(x[i,j] for i in I) <= M[j]*y[j], "Capacity(%s)"%i)

    for (i,j) in x:
        model.addCons(x[i,j] <= d[i]*y[j], "Strong(%s,%s)"%(i,j))

    model.setObjective(
        quicksum(f[j]*y[j] for j in J) +
        quicksum(c[i,j]*x[i,j] for i in I for j in J),
        "minimize")
    model.data = x,y

    return model


def make_data():
    I,d = multidict({1:80, 2:270, 3:250, 4:160, 5:180})            # demand
    J,M,f = multidict({1:[500,1000], 2:[500,1000], 3:[500,1000]}) # capacity, fixed costs
    c = {(1,1):4,  (1,2):6,  (1,3):9,    # transportation costs
         (2,1):5,  (2,2):4,  (2,3):7,
         (3,1):6,  (3,2):3,  (3,3):4,
         (4,1):8,  (4,2):5,  (4,3):3,
         (5,1):10, (5,2):8,  (5,3):4,
         }
    return I,J,d,M,f,c



if __name__ == "__main__":
    I,J,d,M,f,c = make_data()
    model = flp(I,J,d,M,f,c)
    model.optimize()

    EPS = 1.e-6
    x,y = model.data
    edges = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > EPS]
    facilities = [j for j in y if model.getVal(y[j]) > EPS]

    print("Optimal value:", model.getObjVal())
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    try: # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        other = [j for j in y if j not in facilities]
        customers = ["c%s"%i for i in d]
        G.add_nodes_from(facilities)
        G.add_nodes_from(other)
        G.add_nodes_from(customers)
        for (i,j) in edges:
            G.add_edge("c%s"%i,j)

        position = NX.drawing.layout.spring_layout(G)
        NX.draw(G,position,node_color="y",nodelist=facilities)
        NX.draw(G,position,node_color="g",nodelist=other)
        NX.draw(G,position,node_color="b",nodelist=customers)
        P.show()
    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")
