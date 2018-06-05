from pyscipopt import Model
from pyscipopt import quicksum

# AND #
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = model.addVar("r","B")
model.addConsAnd([x,y],r)
model.addCons(x==1)
model.setObjective(r,sense="minimize")
model.optimize()
print("* AND *")
for v in model.getVars():
    print("%s: %d" % (v, round(model.getVal(v))))

# OR #
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = model.addVar("r","B")
model.addConsOr([x,y],r)
model.addCons(x==0)
model.setObjective(r,sense="maximize")
model.optimize()
print("* OR *")
for v in model.getVars():
    print("%s: %d" % (v, round(model.getVal(v))))

# XOR (r as boolean, standard) #
model = Model()
model.hideOutput()
x = model.addVar("x","B")
y = model.addVar("y","B")
z = model.addVar("z","B")
r = True
model.addConsXor([x,y],r)
model.addCons(x==1)
model.optimize()
print("* XOR (as boolean) *")
for v in model.getVars():
    print("%s: %d" % (v, round(model.getVal(v))))
print("r: %s" % r)

# XOR (r as variable, custom) #
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
print("* XOR (as variable) *")
for v in model.getVars():
    if v.name != "n":
        print("%s: %d" % (v, round(model.getVal(v))))
