"""
puzzle.py: solve a simple puzzle using SCIP

On a beach there are octopuses, turtles and cranes.
The total number of legs of all animals is 80, while the number of heads is 32.
What are the minimum numbers of turtles and octopuses, respectively?

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt.scip import *

model = Model("puzzle")
x = model.addVar(vtype="I", name="octopusses", obj=1)
y = model.addVar(vtype="I", name="turtles", obj=1)
z = model.addVar(vtype="I", name="cranes")

# Set up constraint for number of heads
coeffs = { x : 1, y : 1, z : 1 }
model.addCons(coeffs, lhs=32, rhs=32, name="Heads")

# Set up constraints for number of legs
coeffs = { x : 8, y : 4 , z : 2 }
model.addCons(coeffs, lhs=80, rhs=80, name="Legs")

model.hideOutput()
model.optimize()

#solution = model.getBestSol()

print("Optimal value:", model.getObjVal())
print((x.name, y.name, z.name), " = ", (model.getVal(x), model.getVal(y), model.getVal(z)))
