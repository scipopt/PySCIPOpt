"""
lo_unbnd.py: Simple Gurobi example of an infeasible linear programming:

maximize    x1 + x2
subject to  x1 - x2 <= -1
           -x1 + x2 <= -1
            x1,x2 >= 0.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model

model = Model("lo unbounded")

x1 = model.addVar(vtype="C", name="x1")
x2 = model.addVar(vtype="C", name="x2")

model.addCons(x1-x2 >= -1)
model.addCons(x2-x1 >= -1)

model.setObjective(x1 + x2, "maximize")

model.optimize()

status = model.getStatus()

if status == "optimal":
    print("Optimal value:",model.getObjVal())
    print((x1.name, x2.name), " = ", (model.getVal(x1), model.getVal(x2)))
    exit(0)

if status == "unbounded" or status == "infeasible":
    print("Unbounded or infeasible instance")
    model.setObjective(0, "maximize") #todo
    model.optimize()
    status = model.getStatus()

if status == "optimal":
    print("Instance unbounded")
elif status == "infeasible":
    print("Infeasible instance: violated constraints are:")
   # model.computeIIS() todo
    #for c in model.getConstrs():
   #     if c.IISConstr:
     #       print c.ConstrName
else:
    print("Error: Solver finished with non-optimal status",status)
  #  from grbcodes import grbcodes
  #  print grbcodes[status]
