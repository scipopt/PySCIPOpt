"""
lo_wines.py: Simple SCIP example of linear programming.
It solves the same instance as lo_wines_simple.py:

maximize  15x + 18y + 30z
subject to 2x +   y +   z <= 60
           x  +  2y +   z <= 60
                        z <= 30
           x,y,z >= 0
Variables correspond to the production of three types of wine blends,
made from pure-grape wines.
Constraints correspond to the inventory of pure-grape wines.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

#Initialize model
model = Model("Wine blending")

Inventory = {"Alfrocheiro":60, "Baga":60, "Castelao":30}
Grapes = Inventory.keys()

Profit = {"Dry":15, "Medium":18, "Sweet":30}
Blends = Profit.keys()

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

# Create variables
x = {}
for j in Blends:
    x[j] = model.addVar(vtype="C", name="x(%s)"%j)

# Create constraints
c = {}
for i in Grapes:
    c[i] = model.addCons(quicksum(Use[i,j]*x[j] for j in Blends) <= Inventory[i], name="Use(%s)"%i)

# Objective
model.setObjective(quicksum(Profit[j]*x[j] for j in Blends), "maximize")

model.writeProblem("lo_wines.lp")  # useful for debugging

model.optimize()

if model.getStatus() == "optimal":
    print("Optimal value:", model.getObjVal())

    for j in x:
        print(x[j].name, "=", model.getVal(x[j]))
    for i in c:
        print("dual of", c[i].name, ":", model.getDualsolLinear(c[i]))
else:
    print("Problem could not be solved to optimality")
