"""
lo-wines-simple.py: Simple SCIP example of linear programming:

maximize  15x + 18y + 30z
subject to 2x +   y +   z <= 60
           x  +  2y +   z <= 60
                        z <= 30
           x,y,z >= 0

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2015
"""
from pyscipopt import Model

model = Model("Wine blending (simple version)")

x = model.addVar(vtype="C", name="x", obj=15)
y = model.addVar(vtype="C", name="y", obj=18)
z = model.addVar(vtype="C", name="z", obj=30)

model.addCons(2*x + y + z <= 60)
model.addCons(x + 2*y + z <= 60)
model.addCons(z <= 30)

model.setMaximize()

model.optimize()

if model.getStatus() == "optimal":
    model.writeProblem("lo_wines_simple.lp")
    print("Optimal value:", model.getObjVal())
    print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))
else:
    print("Problem could not be solved to optimality")