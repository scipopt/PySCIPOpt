"""
diet.py: model for the modern diet problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
# todo: can we make it work as "from pyscipopt import *"?
from pyscipopt.scip import *

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
        y[j] = model.addVar(vtype="B", name="y(%s)"%j, obj=1.0) # todo: use setObjective() below
    for i in N:
        z[i] = model.addVar(lb=a[i], ub=b[i], name="z(%s)"%i)
    v = model.addVar(vtype="C", name="v")

    # Constraints:
    for i in N:
        coeffs = { x[j] : d[j][i] for j in F }
        coeffs[z[i]] = -1.0
        model.addCons(coeffs, lhs=0.0, rhs=0.0, name="Nutr(%s)"%i)

    coeffs = { x[j] : c[j] for j in F }
    coeffs[v] = -1.0
    model.addCons(coeffs, lhs=0.0, rhs=0.0, name="Cost")

    for j in F:
        coeffs = { y[j] : 1.0, x[j] : -1.0 }
        model.addCons(coeffs, lhs=None, rhs=0.0, name="Eat(%s)"%j)

    model.setMaximize()
    # Objective:
    # todo: model.setObjective(quicksum(y[j]  for j in F), GRB.MAXIMIZE)
    # todo: model.__data = x,y,z,v

    model.data = x,y,z,v

    return model


def make_inst():
    """make_inst: prepare data for the diet model"""

    # todo: use something similar to Gurobi's multidict()?
    c = { # cost
        "QPounder" :  1.84,
        "McLean"   :  2.19,
        "Big Mac"  :  1.84,
        "FFilet"   :  1.44,
        "Chicken"  :  2.29,
        "Fries"    :   .77,
        "McMuffin" :  1.29,
        "1% LFMilk":   .60,
        "OrgJuice" :   .72
    }

    d = { # composition
        "QPounder" : {"Cal":510, "Carbo":34, "Protein":28,
                      "VitA":15, "VitC":  6, "Calc":30, "Iron":20},
        "McLean"   : {"Cal":370, "Carbo":35, "Protein":24,
                      "VitA":15, "VitC": 10, "Calc":20, "Iron":20},
        "Big Mac"  : {"Cal":500, "Carbo":42, "Protein":25,
                      "VitA": 6, "VitC":  2, "Calc":25, "Iron":20},
        "FFilet"   : {"Cal":370, "Carbo":38, "Protein":14,
                      "VitA": 2, "VitC":  0, "Calc":15, "Iron":10},
        "Chicken"  : {"Cal":400, "Carbo":42, "Protein":31,
                      "VitA": 8, "VitC": 15, "Calc":15, "Iron": 8},
        "Fries"    : {"Cal":220, "Carbo":26, "Protein": 3,
                      "VitA": 0, "VitC": 15, "Calc": 0, "Iron": 2},
        "McMuffin" : {"Cal":345, "Carbo":27, "Protein":15,
                      "VitA": 4, "VitC":  0, "Calc":20, "Iron":15},
        "1% LFMilk": {"Cal":110, "Carbo":12, "Protein": 9,
                      "VitA":10, "VitC":  4, "Calc":30, "Iron": 0},
        "OrgJuice" : {"Cal": 80, "Carbo":20, "Protein": 1,
                      "VitA": 2, "VitC":120, "Calc": 2, "Iron": 2}
    }

    F = c.keys()

    a = { # min intake
        "Cal"     : 2000,
        "Carbo"   :  350,
        "Protein" :   55,
        "VitA"    :  100,
        "VitC"    :  100,
        "Calc"    :  100,
        "Iron"    :  100
    }

    b = { # max intake
        "Cal"     : None,
        "Carbo"   : 375,
        "Protein" : None,
        "VitA"    : None,
        "VitC"    : None,
        "Calc"    : None,
        "Iron"    : None
    }

    N = a.keys()

    return F,N,a,b,c,d


if __name__ == "__main__":

    F,N,a,b,c,d = make_inst()

    for b["Cal"] in [None,3500,3000,2500]:

        print()

        if b["Cal"] is None:
            print("Diet for an unlimited amount of calories")
        else:
            print("Diet for a maximum of {0:2.0f} calories".format(b["Cal"]))

        model = diet(F,N,a,b,c,d)
        model.hideOutput()
        model.optimize()

        print("Optimal value: %i different meals" % model.getObjVal())

        x,y,z,v = model.data

        for j in x:
            # to avoid strange results due to rounding errors we don't check for exactly 0
            if model.getVal(x[j]) > 0.1:
                print("{0:10s}: {1:2.0f} dishes --> {2:2.1f} added to objective".format(j, model.getVal(x[j]), model.getVal(y[j])))

        print("amount spent: {0:2.2f}".format(model.getVal(v)))

        print("amount of nutrients:")
        for i in z:
            print("{0:10s}: {1:5.1f}".format(i, model.getVal(z[i])))
