"""
diet.py: model for the modern diet problem

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

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

    # Constraints:
    for i in N:
        model.addCons(quicksum(d[j][i]*x[j] for j in F) == z[i], name="Nutr(%s)" % i)

    model.setObjective(quicksum(c[j]*x[j]  for j in F), "minimize")

    model.data = x,y,z
    return model


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
    status = model.getStatus()
    if status == "optimal":
        print("Optimal value:",model.getObjVal())
        x,y,z = model.data
        for j in x:
            if model.getVal(x[j]) > 0:
                print((j,model.getVal(x[j])))
        print("amount of nutrients:")
        for i in z:
            print((i,model.getVal(z[i])))
        exit(0)
    if status == "unbounded" or status == "infeasible":
        model.setObjective(0, "maximize")
        model.optimize()
        status = model.getStatus()
    if status == "optimal":
        print("Instance unbounded")
    elif status == "infeasible":
        print("Infeasible instance")
#        model.computeIIS()
#        model.writeProblem("diet.lp")
#        for c in model.getConss():
#            if c.IISConstr:
#                print(c.name)
 #       model.feasRelaxS(1,False,False,True)
  #      model.optimize()
   #     model.writeProblem("diet-feasiblity.lp")
  #      status = model.Status
 #       if status == "optimal":
 #           print("Optimal Value=",model.getObjVal())
 #           for v in model.getVars():
 #               print(v.name,"=",model.getVal(v))
 #           exit(0)
