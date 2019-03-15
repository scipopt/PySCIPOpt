##@file diet.py 
#@brief model for the modern diet problem
"""
Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
# todo: can we make it work as "from pyscipopt import *"?
from pyscipopt import Model, quicksum, multidict

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

    # Constraints:
    for i in N:
        model.addCons(quicksum(d[j][i]*x[j] for j in F) == z[i], name="Nutr(%s)"%i)

    model.addCons(quicksum(c[j]*x[j] for j in F) == v, name="Cost")

    for j in F:
        model.addCons(y[j] <= x[j], name="Eat(%s)"%j)

    # Objective:
    model.setObjective(quicksum(y[j] for j in F), "maximize")
    model.data = x,y,z,v

    return model


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
        "Cal"     : [ 2000,  None ],
        "Carbo"   : [  350,  375 ],
        "Protein" : [   55,  None ],
        "VitA"    : [  100,  None ],
        "VitC"    : [  100,  None ],
        "Calc"    : [  100,  None ],
        "Iron"    : [  100,  None ],
     })

    return F,N,a,b,c,d


if __name__ == "__main__":

    F,N,a,b,c,d = make_inst()

    for b["Cal"] in [None,3500,3000,2500]:

        print("\nDiet for a maximum of {0} calories".format(b["Cal"] if b["Cal"] != None else "unlimited"))
        model = diet(F,N,a,b,c,d)
        model.hideOutput() # silent mode
        model.optimize()

        print("Optimal value:",model.getObjVal())
        x,y,z,v = model.data
        for j in x:
            if model.getVal(x[j]) > 0:
                print("{0:30s}: {1:3.1f} dishes --> {2:4.2f} added to objective".format(j,model.getVal(x[j]),model.getVal(y[j])))
        print("amount spent:",model.getObjVal())

        print("amount of nutrients:")
        for i in z:
            print("{0:30s}: {1:4.2f}".format(i,model.getVal(z[i])))
