"""
diet.py: model for the modern diet problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

def diet(F,N,a,b,c,d):
    """diet -- model for the modern diet problem
    Parameters:
        F - set of foods
        N - set of nutrients
        a[i] - minimum intake of nutrient i
        b[i] - maximum intake of nutrient i
        c[j] - cost of food j
        d[j][i] - amount of nutrient i in food j
    Returns a model, ready to be solved.
    """

    model = Model("modern diet")

    # Create variables
    x,y,z = {},{},{}
    for j in F:
        x[j] = model.addVar(vtype="I", name="x(%s)" % j)

    for i in N:
        z[i] = model.addVar(lb=a[i], ub=b[i], vtype="C", name="z(%s)" % i)

    model.update()

    # Constraints:
    for i in N:
        model.addConstr(quicksum(d[j][i]*x[j] for j in F) == z[i], "Nutr(%s)" % i)

    model.setObjective(quicksum(c[j]*x[j]  for j in F),GRB.MINIMIZE )

    model.update()
    model.__data = x,y,z
    return model


inf = GRB.INFINITY
def make_inst():
    """make_inst: prepare data for the diet model"""
    F,c,d = multidict({       # cost # composition
        "CQPounder":  [ 360, {"Cal":556, "Carbo":39, "Protein":30,
                              "VitA":147,"VitC": 10, "Calc":221, "Iron":2.4}],
        "Big Mac"  :  [ 320, {"Cal":556, "Carbo":46, "Protein":26,
                              "VitA":97, "VitC":  9, "Calc":142, "Iron":2.4}],
        "FFilet"   :  [ 270, {"Cal":356, "Carbo":42, "Protein":14,
                              "VitA":28, "VitC":  1, "Calc": 76, "Iron":0.7}],
        "Chicken"  :  [ 290, {"Cal":431, "Carbo":45, "Protein":20,
                              "VitA": 9, "VitC":  2, "Calc": 37, "Iron":0.9}],
        "Fries"    :  [ 190, {"Cal":249, "Carbo":30, "Protein": 3,
                              "VitA": 0, "VitC":  5, "Calc":  7, "Iron":0.6}],
        "Milk"     :  [ 170, {"Cal":138, "Carbo":10, "Protein": 7,
                              "VitA":80, "VitC":  2, "Calc":227, "Iron": 0}],
        "VegJuice" :  [ 100, {"Cal": 69, "Carbo":17, "Protein": 1,
                              "VitA":750,"VitC":  2, "Calc":18,  "Iron": 0}],
        })

    N,a,b = multidict({       # min,max intake
        "Cal"     : [ 2000,  3000],
        "Carbo"   : [  300,  375 ],
        "Protein" : [   50,   60 ],
        "VitA"    : [  500,  750 ],
        "VitC"    : [   85,  100 ],
        "Calc"    : [  660,  900 ],
        "Iron"    : [  6.0,  7.5 ],
     })

    return F,N,a,b,c,d


if __name__ == "__main__":

    F,N,a,b,c,d = make_inst()
    model = diet(F,N,a,b,c,d)
    model.optimize()
    status = model.Status
    if status == GRB.Status.OPTIMAL:
        print "Optimal value:",model.ObjVal
        x,y,z = model.__data
        for j in x:
            if x[j].X > 0:
                print (j,x[j].X)
        print "amount of nutrients:"
        for i in z:
            print (i,z[i].X)
        exit(0)
    if status == GRB.Status.UNBOUNDED or status == GRB.Status.INF_OR_UNBD:
        model.setObjective(0,GRB.MAXIMIZE)
        model.optimize()
        status = model.Status
    if status == GRB.Status.OPTIMAL:
        print "Instance unbounded"
    elif status == GRB.Status.INFEASIBLE:
        print "Infeasible instance: violated constraints are:"
        model.computeIIS()
        model.write("diet.ilp")
        for c in model.getConstrs():
            if c.IISConstr:
                print c.ConstrName
        model.feasRelaxS(1,False,False,True)
        model.optimize()
        model.write("diet-feasiblity.lp")
        status = model.Status
        if status == GRB.Status.OPTIMAL:
            print "Opt. Value=",model.ObjVal
            for v in model.getVars():
                print v.VarName,v.X
            exit(0)
