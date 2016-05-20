"""
lo1.py: Simple Gurobi example of linear programming:

maximize 15 x1+18 x2 + 30 x3
subject to 2 x1+ x2+ x3 <= 60
           x1  +2x2+ x2 <= 60
                     x3 <= 30
           x1,x2,x3 >= 0.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model

model = Model("lo1")

x1 = model.addVar(vtype="C", name="x1", obj = 15)
x2 = model.addVar(vtype="C", name="x2", obj = 18)
x3 = model.addVar(vtype="C", name="x3", lb=0, ub=30, obj=30)

model.addCons(2*x1 + x2 + x3 <= 60)
model.addCons(x1 + 2*x2 + x3 <= 60)

model.setMaximize()

model.optimize()

if model.getStatus() == "optimal":
    print("Optimal value:", model.getObjVal())
    print((x1.name, x2.name, x3.name), " = ", (model.getVal(x1), model.getVal(x2), model.getVal(x3)))
