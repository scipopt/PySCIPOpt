"""
lo1.py: Simple Gurobi example of linear programming:

maximize 15 x1+18 x2 + 30 x3
subject to 2 x1+ x2+ x3 <= 60
           x1  +2x2+ x2 <= 60
                     x3 <= 30
           x1,x2,x3 >= 0.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

model = Model("lo1")

x1 = model.addVar(vtype="C", name="x1")
x2 = model.addVar(vtype="C", name="x2")
x3 = model.addVar(lb=0, ub=30, vtype="C", name="x3")

model.update()

model.addConstr(2*x1 + x2 + x3 <= 60)
model.addConstr(x1 + 2*x2 + x3 <= 60)

model.setObjective(15*x1 + 18*x2 + 30*x3, GRB.MAXIMIZE)

model.optimize()

if model.Status == GRB.OPTIMAL:
    model.write("answer.sol")
    print "Opt. Value=",model.ObjVal
    for v in model.getVars():
        print v.VarName,v.X
