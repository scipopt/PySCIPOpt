##@file tutorial/puzzle.py
#@brief solve a simple puzzle using SCIP
"""
On a beach there are octopuses, turtles and cranes.
The total number of legs of all animals is 80, while the number of heads is 32.
What are the minimum numbers of turtles and octopuses, respectively?

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model

model = Model("puzzle")
x = model.addVar(vtype="I", name="octopusses")
y = model.addVar(vtype="I", name="turtles")
z = model.addVar(vtype="I", name="cranes")

# Set up constraint for number of heads
model.addCons(x + y + z == 32, name="Heads")

# Set up constraint for number of legs
model.addCons(8*x + 4*y + 2*z == 80, name="Legs")

# Set objective function
model.setObjective(x + y, "minimize")

model.hideOutput()
model.optimize()

#solution = model.getBestSol()

print("Optimal value:", model.getObjVal())
print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))
