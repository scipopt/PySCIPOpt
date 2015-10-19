"""
lo_wines.py: Simple Gurobi example of linear programming.
It solves the same instance as lo_wines_simple.py:

maximize  15x1 + 18x2 + 30x3
subject to 2x1 +   x2 +   x3 <= 60
           x1  +  2x2 +   x3 <= 60
                          x3 <= 30
           x1,x2,x3 >= 0
Variables correspond to the production of three types of wine blends,
made from pure-grape wines.
Constraints correspond to the inventory of pure-grape wines.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from gurobipy import *
model = Model("Wine blending")

Grapes,Inventory = multidict({"Alfrocheiro":60, "Baga":60, "Castelao":30})
Blends,Profit = multidict({"Dry":15, "Medium":18, "Sweet":30})

Use = {
    ("Alfrocheiro","Dry"):2,
    ("Alfrocheiro","Medium"):1,
    ("Alfrocheiro","Sweet"):1,
    ("Baga","Dry"):1,
    ("Baga","Medium"):2,
    ("Baga","Sweet"):1,
    ("Castelao","Dry"):0,
    ("Castelao","Medium"):0,
    ("Castelao","Sweet"):1
    }

x = {}
for j in Blends:
    x[j] = model.addVar(vtype="C", name="x(%s)"%j)
model.update()

for i in Grapes:
    model.addConstr(quicksum(Use[i,j]*x[j] for j in Blends) <= Inventory[i], name="Use(%s)"%i)


model.setObjective(quicksum(Profit[j]*x[j] for j in Blends), GRB.MAXIMIZE)
model.update()
model.write("lo_wines.lp") # useful for debugging
model.optimize()

if model.Status == GRB.OPTIMAL:
    print "Opt. Value=",model.ObjVal
    for j in x:
        print x[j].VarName,x[j].X
    for c in model.getConstrs():
        print c.ConstrName,c.Pi
else:
    print "Problem was not solved to optimality"
