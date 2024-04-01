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
    model2 = Model(sourceModel=model)

    set_nonlinear_objective(model, obj, sense='maximize')
    t = model2.addVar("objective")
    model2.addCons(t <= obj)
    model2.setObjective(t, "maximize")

    obj_expr = model.getObjective()
    assert obj_expr.degree() == 1

    model.optimize()
    model2.optimize()
    #assert model.isEQ(model.getObjVal(), model2.getObjVal())