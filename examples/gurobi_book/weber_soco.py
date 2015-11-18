"""
weber_soco.py:  model for solving the weber problem using soco.

Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""
from gurobipy import *

def weber(I,x,y,w):
    """weber: model for solving the single source weber problem using soco.
    Parameters:
        - I: set of customers
        - x[i]: x position of customer i
        - y[i]: y position of customer i
        - w[i]: weight of customer i
    Returns a model, ready to be solved.
    """

    model = Model("weber")
    X,Y,z,xaux,yaux = {},{},{},{},{}
    X = model.addVar(lb=-GRB.INFINITY, vtype="C", name="X")
    Y = model.addVar(lb=-GRB.INFINITY, vtype="C", name="Y")
    for i in I:
        z[i] = model.addVar(vtype="C", name="z(%s)"%(i))
        xaux[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="xaux(%s)"%(i))
        yaux[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="yaux(%s)"%(i))
    model.update()

    for i in I:
        model.addConstr(xaux[i]*xaux[i] + yaux[i]*yaux[i] <= z[i]*z[i], "MinDist(%s)"%(i))
        model.addConstr(xaux[i] == (x[i]-X), "xAux(%s)"%(i))
        model.addConstr(yaux[i] == (y[i]-Y), "yAux(%s)"%(i))

    model.setObjective(quicksum(w[i]*z[i] for i in I), GRB.MINIMIZE)

    model.update()
    model.__data = X,Y,z
    return model


import random
def make_data(n,m):
    I = range(1,n+1)
    J = range(1,m+1)
    x,y,w = {},{},{}
    for i in I:
        x[i] = random.randint(0,100)
        y[i] = random.randint(0,100)
        w[i] = random.randint(1,5)
    return I,J,x,y,w


if __name__ == "__main__":
    import sys
    random.seed(3)
    n = 7
    m = 1
    I,J,x,y,w = make_data(n,m)
    print "data:"
    print "%s\t%8s\t%8s\t%8s" % ("i","x[i]","y[i]","w[i]")
    for i in I:
        print "%s\t%8g\t%8g\t%8g" % (i,x[i],y[i],w[i])
    print

    model = weber(I,x,y,w)
    model.optimize()
    X,Y,z = model.__data
    print "Optimal value=",model.ObjVal
    print "Selected position:",
    print "\t",(round(X.X),round(Y.X))
    print
    print "Solution:"
    print "%s\t%8s" % ("i","z[i]")
    for i in I:
        print "%s\t%8g" % (i,z[i].X)
    print
    try: # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        G.add_nodes_from(I)
        G.add_nodes_from(["D"])

        position = {}
        for i in I:
            position[i] = (x[i],y[i])
        position["D"] = (round(X.X),round(Y.X))

        NX.draw(G,pos=position,node_size=200,node_color="g",nodelist=I)
        NX.draw(G,pos=position,node_size=400,node_color="w",nodelist=["D"],alpha=0.5)
        P.savefig("weber.pdf",format="pdf",dpi=300)
        P.show()

    except ImportError:
        print "install 'networkx' and 'matplotlib' for plotting"





def weber_MS(I,J,x,y,w):
    """weber -- model for solving the weber problem using soco (multiple source version).
    Parameters:
        - I: set of customers
        - J: set of potential facilities
        - x[i]: x position of customer i
        - y[i]: y position of customer i
        - w[i]: weight of customer i
    Returns a model, ready to be solved.
    """
    M = max([((x[i]-x[j])**2 + (y[i]-y[j])**2) for i in I for j in I])
    model = Model("weber - multiple source")
    X,Y,v,u = {},{},{},{}
    xaux,yaux,uaux = {},{},{}
    for j in J:
        X[j] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="X(%s)"%j)
        Y[j] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="Y(%s)"%j)
        for i in I:
            v[i,j] = model.addVar(vtype="C", name="v(%s,%s)"%(i,j))
            u[i,j] = model.addVar(vtype="B", name="u(%s,%s)"%(i,j))
            xaux[i,j] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="xaux(%s,%s)"%(i,j))
            yaux[i,j] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="yaux(%s,%s)"%(i,j))
            uaux[i,j] = model.addVar(vtype="C", name="uaux(%s,%s)"%(i,j))

    model.update()

    for i in I:
        model.addConstr(quicksum(u[i,j] for j in J) == 1, "Assign(%s)"%i)
        for j in J:
            model.addConstr(xaux[i,j]*xaux[i,j] + yaux[i,j]*yaux[i,j] <= v[i,j]*v[i,j], "MinDist(%s,%s)"%(i,j))
            model.addConstr(xaux[i,j] == (x[i]-X[j]), "xAux(%s,%s)"%(i,j))
            model.addConstr(yaux[i,j] == (y[i]-Y[j]), "yAux(%s,%s)"%(i,j))
            model.addConstr(uaux[i,j] >= v[i,j] - M*(1-u[i,j]), "uAux(%s,%s)"%(i,j))

    model.setObjective(quicksum(w[i]*uaux[i,j] for i in I for j in J), GRB.MINIMIZE)

    model.update()
    model.__data = X,Y,v,u
    return model


if __name__ == "__main__":
    random.seed(3)
    n = 7
    m = 1
    I,J,x,y,w = make_data(n,m)
    model = weber_MS(I,J,x,y,w)
    model.optimize()
    X,Y,w,z = model.__data
    print "Optimal value=",model.ObjVal
    print "Selected positions:"
    for j in J:
        print "\t",(X[j].X,Y[j].X)
    for (i,j) in sorted(w.keys()):
        print "\t",(i,j),w[i,j].X,z[i,j].X

    EPS = 1.e-4
    edges = [(i,j) for (i,j) in z if z[i,j].X > EPS]

    try: # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        G.add_nodes_from(I)
        G.add_nodes_from("%s"%j for j in J)     # for distinguishing J from I, make nodes as strings
        for (i,j) in edges:
            G.add_edge(i,"%s"%j)

        position = {}
        for i in I:
            position[i] = (x[i],y[i])
        for j in J:
            position["%s"%j] = (X[j].X,Y[j].X)
        print position

        NX.draw(G,position,node_color="g",nodelist=I)
        NX.draw(G,position,node_color="w",nodelist=["%s"%j for j in J])
        P.show()
    except ImportError:
        print "install 'networkx' and 'matplotlib' for plotting"
