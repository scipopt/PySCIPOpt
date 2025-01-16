from pyscipopt import Model, exp, log, sqrt, sin
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

def test_nonlinear_objective():
    model = Model()

    v = model.addVar()
    w = model.addVar()
    x = model.addVar()
    y = model.addVar()
    z = model.addVar()

    obj = 0
    obj += exp(v)
    obj += log(w)
    obj += sqrt(x)
    obj += sin(y)
    obj += z**3 * y

    model.addCons(v + w + x + y + z <= 1)
    set_nonlinear_objective(model, obj, sense='maximize')

    model2 = Model()

    a = model2.addVar()
    b = model2.addVar()
    c = model2.addVar()
    d = model2.addVar()
    e = model2.addVar()

    obj2 = 0
    obj2 += exp(a)
    obj2 += log(b)
    obj2 += sqrt(c)
    obj2 += sin(d)
    obj2 += e**3 * d

    model2.addCons(a + b + c + d + e <= 1)
    
    t = model2.addVar(lb=-float("inf"),obj=1)
    model2.addCons(t <= obj2)
    model2.setMaximize()

    obj_expr = model.getObjective()
    assert obj_expr.degree() == 1

    model.setParam("numerics/epsilon", 10**(-5)) # bigger eps due to nonlinearities
    model2.setParam("numerics/epsilon", 10**(-5)) 

    model.optimize()
    model2.optimize()
    assert model.isEQ(model.getObjVal(), model2.getObjVal())