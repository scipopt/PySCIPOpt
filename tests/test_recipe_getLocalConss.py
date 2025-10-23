from pyscipopt import Model, SCIP_EVENTTYPE
from pyscipopt.recipes.getLocalConss import *
from helpers.utils import random_mip_1

def localconss(model, event):
    local_conss = getLocalConss(model)
    assert len(local_conss[1]) == getNLocalConss(model)[1]
    assert len(local_conss[0]) == len(model.getConss())
    assert local_conss[0] == model.getConss()

    vars = model.getVars()
    if model.getCurrentNode().getNumber() == 1:
        pass

    elif model.getCurrentNode().getNumber() == 2:
        model.data["local_cons1"] = model.addCons(vars[0] + vars[1] <= 1, name="c1", local=True)
        assert getNLocalConss(model)[1] == 1
        assert getLocalConss(model)[1][0] == model.data["local_cons1"]

    elif model.getCurrentNode().getNumber() == 4:
        local_conss = getLocalConss(model)
        model.data["local_cons2"] = model.addCons(vars[1] + vars[2] <= 1, name="c2", local=True)
        model.data["local_cons3"] = model.addCons(vars[2] + vars[3] <= 1, name="c3", local=True)
        assert getNLocalConss(model)[1] == 3
        assert getLocalConss(model)[1][0] == model.data["local_cons1"]
        assert getLocalConss(model)[1][1] == model.data["local_cons2"]
        assert getLocalConss(model)[1][2] == model.data["local_cons3"]

    elif model.getCurrentNode().getParent().getNumber() not in [2,4]:
        assert getLocalConss(model) == [model.getConss(), []]

def test_getLocalConss():
    model = random_mip_1(node_lim=4)
    model.data = {}

    model.attachEventHandlerCallback(localconss, [SCIP_EVENTTYPE.NODEFOCUSED])
    model.optimize()
    assert len(model.data) == 3