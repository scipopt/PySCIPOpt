from helpers.utils import bin_packing_model

from pyscipopt.recipes.structured_optimization_trace import (
    attach_structured_optimization_trace,
)


def test_structured_optimization_trace():
    from random import randint

    model = bin_packing_model(sizes=[randint(1, 40) for _ in range(120)], capacity=50)
    model.setParam("limits/time", 5)

    model.data = {"test": True}
    model = attach_structured_optimization_trace(model)

    assert "test" in model.data
    assert "trace" in model.data

    model.optimize()

    required_fields = {"time", "primalbound", "dualbound", "gap", "nodes", "nsol"}
    for record in model.data["trace"]:
        assert required_fields <= set(record.keys())

    primalbounds = [r["primalbound"] for r in model.data["trace"]]
    for i in range(1, len(primalbounds)):
        assert primalbounds[i] <= primalbounds[i - 1]

    dualbounds = [r["dualbound"] for r in model.data["trace"]]
    for i in range(1, len(dualbounds)):
        assert dualbounds[i] >= dualbounds[i - 1]
