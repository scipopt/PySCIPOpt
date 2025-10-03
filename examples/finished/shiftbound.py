from pyscipopt import (
    Model,
    SCIP_PARAMSETTING
)
from typing import List, Optional


def knapsack(
    instance_name: str,
    sizes: List[int],
    values: List[int],
    upper_bound: List[int],
    lower_bound: List[int],
    capacity: int,
    vtypes: Optional[List[str]],
) -> tuple[Model, dict]:
    """
    Model an instance of the knapsack problem

    Parameters:
    sizes: List[int] - the sizes of the items
    values: List[int] - the values of the items
    upper_bound: List[int] - upper bounds per variable
    lower_bound: List[int] - lower bounds per variable
    capacity: int - the knapsack capacity
    vtypes: Optional[List[str]] - variable types ("B", "I", "C")

    Returns:
    tuple[Model, dict] - the SCIP model and the variables dictionary
    """

    m = Model(f"Knapsack: {instance_name}")
    x = {}
    for i in range(len(sizes)):
        assert isinstance(sizes[i], int)
        assert isinstance(values[i], int)
        assert isinstance(upper_bound[i], int)
        assert isinstance(lower_bound[i], int)

        vt = "I"
        if vtypes is not None:
            assert len(vtypes) == len(sizes)
            assert isinstance(vtypes[i], str) or (vtypes[i] is None)
            vt = vtypes[i]

        x[i] = m.addVar(
            vtype=vt,
            obj=values[i],
            lb=lower_bound[i],
            ub=upper_bound[i],
            name=f"x{i}",
        )

    assert isinstance(capacity, int)
    m.addCons(sum(sizes[i] * x[i] for i in range(len(sizes))) <= capacity)

    m.setMaximize()

    return m, x


if __name__ == "__main__":
    instance_name = "Knapsack"
    sizes = [2, 1, 3]
    values = [2, 3, 1]
    upper_bounds = [1, 4, 1]
    lower_bounds = [0, 2, 0]
    capacity = 3

    model, var_list = knapsack(
        instance_name, sizes, values, upper_bounds, lower_bounds, capacity
    )

    model = Model()

    # isolate test: disable many automatic presolvers/propagators
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    for key in (
        "presolving/boundshift/maxrounds",
        "presolving/domcol/maxrounds",
        "presolving/dualsparsify/maxrounds",
        "presolving/implics/maxrounds",
        "presolving/inttobinary/maxrounds",
        "presolving/milp/maxrounds",
        "presolving/sparsify/maxrounds",
        "presolving/trivial/maxrounds",
        "propagating/dualfix/maxprerounds",
        "propagating/probing/maxprerounds",
        "propagating/symmetry/maxprerounds",
        "constraints/linear/maxprerounds",
    ):
        model.setParam(key, 0)

    # run presolve on instance
    model.presolve()