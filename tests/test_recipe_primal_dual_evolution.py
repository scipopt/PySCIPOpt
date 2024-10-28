from pyscipopt.recipes.primal_dual_evolution import attach_primal_dual_evolution_eventhdlr
from helpers.utils import bin_packing_model

def test_primal_dual_evolution():
    from random import randint

    model = bin_packing_model(sizes=[randint(1,40) for _ in range(120)],  capacity=50)
    model.setParam("limits/time",5)

    model.data = {"test": True}
    model = attach_primal_dual_evolution_eventhdlr(model)

    assert "test" in model.data
    assert "primal_log" in model.data

    model.optimize()

    for i in range(1, len(model.data["primal_log"])):
        if model.getObjectiveSense() == "minimize":
            assert model.data["primal_log"][i][1] <= model.data["primal_log"][i-1][1]
        else:
            assert model.data["primal_log"][i][1] >= model.data["primal_log"][i-1][1]
    
    for i in range(1, len(model.data["dual_log"])):
        if model.getObjectiveSense() == "minimize":
            assert model.data["dual_log"][i][1] >= model.data["dual_log"][i-1][1]
        else:
            assert model.data["dual_log"][i][1] <= model.data["dual_log"][i-1][1]
