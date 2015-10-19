"""
diet.py: model for the modern diet problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

def diet(F,N,a,b,c,d):
    """diet -- model for the modern diet problem
    Parameters:
        - F: set of foods
        - N: set of nutrients
        - a[i]: minimum intake of nutrient i
        - b[i]: maximum intake of nutrient i
        - c[j]: cost of food j
        - d[j][i]: amount of nutrient i in food j
    Returns a model, ready to be solved.
    """

    model = Model("modern diet")

    # Create variables
    x,y,z = {},{},{}
    for j in F:
        x[j] = model.addVar(vtype="I", name="x(%s)"%j)
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
    for i in N:
        z[i] = model.addVar(lb=a[i], ub=b[i], name="z(%s)"%j)
    v = model.addVar(vtype="C", name="v")
    model.update()

    # Constraints:
    for i in N:
        model.addConstr(quicksum(d[j][i]*x[j] for j in F) == z[i], "Nutr(%s)"%i)

    model.addConstr(quicksum(c[j]*x[j]  for j in F) == v, "Cost")

    for j in F:
        model.addConstr(y[j] <= x[j], "Eat(%s)"%j)

    # Objective:
    model.setObjective(quicksum(y[j]  for j in F), GRB.MAXIMIZE)
    model.__data = x,y,z,v

    return model


inf = GRB.INFINITY
def make_inst():
    """make_inst: prepare data for the diet model"""
    F,c,d = multidict({       # cost # composition
        "QPounder" :  [ 1.84, {"Cal":510, "Carbo":34, "Protein":28,
                               "VitA":15, "VitC":  6, "Calc":30, "Iron":20}],
        "McLean"   :  [ 2.19, {"Cal":370, "Carbo":35, "Protein":24, "VitA":15,
                               "VitC": 10, "Calc":20, "Iron":20}],
        "Big Mac"  :  [ 1.84, {"Cal":500, "Carbo":42, "Protein":25,
                               "VitA": 6, "VitC":  2, "Calc":25, "Iron":20}],
        "FFilet"   :  [ 1.44, {"Cal":370, "Carbo":38, "Protein":14,
                               "VitA": 2, "VitC":  0, "Calc":15, "Iron":10}],
        "Chicken"  :  [ 2.29, {"Cal":400, "Carbo":42, "Protein":31,
                               "VitA": 8, "VitC": 15, "Calc":15, "Iron": 8}],
        "Fries"    :  [  .77, {"Cal":220, "Carbo":26, "Protein": 3,
                               "VitA": 0, "VitC": 15, "Calc": 0, "Iron": 2}],
        "McMuffin" :  [ 1.29, {"Cal":345, "Carbo":27, "Protein":15,
                               "VitA": 4, "VitC":  0, "Calc":20, "Iron":15}],
        "1% LFMilk":  [  .60, {"Cal":110, "Carbo":12, "Protein": 9,
                               "VitA":10, "VitC":  4, "Calc":30, "Iron": 0}],
        "OrgJuice" :  [  .72, {"Cal": 80, "Carbo":20, "Protein": 1,
                               "VitA": 2, "VitC":120, "Calc": 2, "Iron": 2}],
        })

    N,a,b = multidict({       # min,max intake
        "Cal"     : [ 2000,  inf ],
        "Carbo"   : [  350,  375 ],
        "Protein" : [   55,  inf ],
        "VitA"    : [  100,  inf ],
        "VitC"    : [  100,  inf ],
        "Calc"    : [  100,  inf ],
        "Iron"    : [  100,  inf ],
     })

    return F,N,a,b,c,d


if __name__ == "__main__":

    F,N,a,b,c,d = make_inst()

    for b["Cal"] in [inf,3500,3000,2500]:

        print "\n\nDiet for a maximum of %s calories" % b["Cal"]
        model = diet(F,N,a,b,c,d)
        model.Params.OutputFlag = 0 # silent mode
        model.optimize()

        print "Optimal value:",model.ObjVal
        x,y,z,v = model.__data
        for j in x:
            if x[j].X > 0:
                print "%30s: %5s dishes --> %s added to objective" % (j,x[j].X,y[j].X)
        print "amount spent:",v.X

        print "amount of nutrients:"
        for i in z:
            print "%30s: %5s" % (i,z[i].X)
