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

    # todo: simplify to model = Model("modern diet")
    model = Model()
    model.create()
    model.includeDefaultPlugins()
    model.createProbBasic("modern diet")

    # Create variables
    x,y,z = {},{},{}
    for j in F:
        x[j] = model.addVar(vtype="I", name="x(%s)"%j)
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
    for i in N:
        z[i] = model.addVar(lb=a[i], ub=b[i], name="z(%s)"%j)
    v = model.addVar(vtype="C", name="v")

    # Constraints:
    for i in N:
        model.addCons(quicksum(d[j][i]*x[j] for j in F) == z[i], name="Nutr(%s)"%i)

    model.addCons(quicksum(c[j]*x[j] for j in F) == v, name="Cost")

    for j in F:
        model.addCons(y[j] <= x[j], name="Eat(%s)"%j)

    # Objective:
    model.setObjective(quicksum(y[j] for j in F), SCIP_OBJSENSE_MAXIMIZE)
    model.__data = x,y,z,v

    return model


def make_inst():
    """make_inst: prepare data for the diet model"""

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
        "McLean"   : {"Cal":370, "Carbo":35, "Protein":24, "VitA":15,
                      "VitC": 10, "Calc":20, "Iron":20},
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

        print("\n\nDiet for a maximum of %s calories") % b["Cal"] if b["Cal"] != None else "unlimited"
        model = diet(F,N,a,b,c,d)
        model.setMessagehdlrQuiet(True) # silent mode
        model.optimize()

        print("Optimal value:",model.getObjVal())
        x,y,z,v = model.__data
        for j in x:
            if x[j].getSolVal() > 0:
                print("%30s: %5s dishes --> %s added to objective") % (j,x[j].getSolVal(),y[j].getSolVal())
        print("amount spent:",v.getSolVal())

        print("amount of nutrients:")
        for i in z:
            print("%30s: %5s") % (i,z[i].getSolVal())
