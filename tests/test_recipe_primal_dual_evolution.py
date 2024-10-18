from pyscipopt.recipes.primal_dual_evolution import get_primal_dual_evolution
from helpers.utils import bin_packing_model

def test_primal_dual_evolution():
    from random import randint

    model = bin_packing_model(sizes=[randint(1,40) for _ in range(120)],  capacity=50)
    model.setParam("limits/time",5)

    model.data = {"test": True}
    model = get_primal_dual_evolution(model)

    assert "test" in model.data
    assert "primal_solutions" in model.data

    model.optimize()

    # these are required because the event handler doesn't capture the final state
    model.data["primal_solutions"].append((model.getSolvingTime(), model.getPrimalbound()))
    model.data["dual_solutions"].append((model.getSolvingTime(), model.getDualbound()))

    for i in range(1, len(model.data["primal_solutions"])):
        if model.getObjectiveSense() == "minimize":
            assert model.data["primal_solutions"][i][1] <= model.data["primal_solutions"][i-1][1]
        else:
            assert model.data["primal_solutions"][i][1] >= model.data["primal_solutions"][i-1][1]
    
    for i in range(1, len(model.data["dual_solutions"])):
        if model.getObjectiveSense() == "minimize":
            assert model.data["dual_solutions"][i][1] >= model.data["dual_solutions"][i-1][1]
        else:
            assert model.data["dual_solutions"][i][1] <= model.data["dual_solutions"][i-1][1]

    # how to get a simple plot of the data
    #from pyscipopt.recipes.primal_dual_evolution import plot_primal_dual_evolution
    #plot_primal_dual_evolution(model)