"""
lo_infeas.py: Simple SCIP example of an infeasible linear programming:

maximize    x1 + x2
subject to  x1 - x2 <= -1
           -x1 + x2 <= -1
            x1,x2 >= 0.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model

model = Model("lo infeas")

x1 = model.addVar(vtype="C", name="x1")
x2 = model.addVar(vtype="C", name="x2")

model.addCons(x1-x2 <= -1)
model.addCons(x2-x1 <= -1)

model.setObjective(x1 + x2, "maximize")

model.optimize()

status = model.getStatus()
if status == "optimal":
    print("Optimal value:", model.getObjVal())
    for v in model.getVars():
        print(v.name,model.getVal(v))
    exit(0)

if status == "unbounded" or status == "infeasible":
    model.setObjective(0, "maximize")
    model.optimize()
    status = model.getStatus()

if status == "optimal":
    print("Instance unbounded")
elif status == "infeasible":
    print("Infeasible instance: violated constraints are:")
 #   model.computeIIS() todo
  #  for c in model.getConstrs():
   #     if c.IISConstr:
    #        print(c.ConstrName)
else:
    print("Error: Solver finished with non-optimal status", status)