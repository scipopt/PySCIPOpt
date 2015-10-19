"""
lo-wines-simple.py: Simple Gurobi example of linear programming:

maximize  15x1 + 18x2 + 30x3
subject to 2x1 +   x2 +   x3 <= 60
           x1  +  2x2 +   x3 <= 60
                          x3 <= 30
           x1,x2,x3 >= 0

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

model = Model("Wine blending (simple version)")

x1 = model.addVar(vtype="C", name="x1")
x2 = model.addVar(vtype="C", name="x2")
x3 = model.addVar(vtype="C", name="x3")

model.update()

model.addConstr(2*x1 + x2 + x3 <= 60)
model.addConstr(x1 + 2*x2 + x3 <= 60)
model.addConstr(x3 <= 30)

model.setObjective(15*x1 + 18*x2 + 30*x3, GRB.MAXIMIZE)

model.optimize()

if model.Status == GRB.OPTIMAL:
    model.write("lo_wines_simple.sol")
    print "Opt. Value=",model.ObjVal
    for v in model.getVars():
        print v.VarName,v.X
else:
    print "Problem was not solved to optimality"
