"""
puzzle.py: solve a simple puzzle using gurobi

On a beach, there are octopuses, turtles and cranes.
Total number of legs for all is 80 while the number of heads is 32.
What are the minimum numbers of turtles and octopuses?

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *

model = Model("puzzle")
x = model.addVar(vtype="I", name="x")
y = model.addVar(vtype="I", name="y")
z = model.addVar(vtype="I", name="z")
model.update()

model.addConstr(x + y + z == 32, "Heads")
model.addConstr(2*x + 4*y + 8*z == 80, "Legs")

model.setObjective(y + z, GRB.MINIMIZE)

model.optimize()

print "Opt. Val.=",model.ObjVal
print "(x,y,z)=",(x.X,y.X,z.X)
