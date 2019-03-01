##@file finished/logical.py
#@brief Tutorial example on how to use AND/OR/XOR constraints

from pyscipopt import Model
from pyscipopt import quicksum

"""

 AND/OR/XOR CONSTRAINTS

 Tutorial example on how to use AND/OR/XOR constraints.

 N.B.: standard SCIP XOR constraint works differently from AND/OR by design.
 The constraint is set with a boolean rhs instead of an integer resultant.
 cf. http://listserv.zib.de/pipermail/scip/2018-May/003392.html
 A workaround to get the resultant as variable is here proposed.

"""

def printFunc(name,m):
    """prints results"""
    print("* %s *" % name)
    objSet = bool(m.getObjective().terms.keys())
    print("* Is objective set? %s" % objSet)
    if objSet:
        print("* Sense: %s" % m.getObjectiveSense())
    for v in m.getVars():
        if v.name != "n":
            print("%s: %d" % (v, round(m.getVal(v))))
    print("\n")

# AND 
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = model.addVar("r","B")
model.addConsAnd([x,y,z],r)
model.addCons(x==1)
model.setObjective(r,sense="minimize")
model.optimize()
printFunc("AND",model)

# OR 
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = model.addVar("r","B")
model.addConsOr([x,y,z],r)
model.addCons(x==0)
model.setObjective(r,sense="maximize")
model.optimize()
printFunc("OR",model)

# XOR (r as boolean, standard) 
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = True
model.addConsXor([x,y,z],r)
model.addCons(x==1)
model.optimize()
printFunc("Standard XOR (as boolean)",model)

# XOR (r as variable, custom) 
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = model.addVar("r","B")
n = model.addVar("n","I") #auxiliary
model.addCons(r+quicksum([x,y,z]) == 2*n)
model.addCons(x==0)
model.setObjective(r,sense="maximize")
model.optimize()
printFunc("Custom XOR (as variable)",model)
