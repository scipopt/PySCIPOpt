from pyscipopt.scip import Model
import os
from helpers.utils import random_mip_1
from json import load
import pytest
import numpy as np


@pytest.fixture
def optimized_model():
    # Using small=True for speed across tests
    model = random_mip_1(small=True, node_lim=2400)
    model.optimize()
    return model


# model factory for testing solution-related statistics
@pytest.fixture
def make_optimized_model_with_fixed_primal_solutions():
    def _make(solutions):
        model = Model()

        x = model.addVar(vtype="I", lb=0, ub=2)
        y = model.addVar(vtype="I", lb=0, ub=2)
        z = model.addVar(vtype="I", lb=0, ub=2)

        model.addCons(x + y + z <= 2)

        model.setObjective(x + y + z, "maximize")

        for solution in solutions:
            sol = model.createOrigSol()

            x_val, y_val, z_val = solution
            sol[x] = x_val
            sol[y] = y_val
            sol[z] = z_val

            model.addSol(sol)

        model.optimize()
        return model
    
    return _make


def test_statistics_json(optimized_model):
    optimized_model.writeStatisticsJson("statistics.json")

    with open("statistics.json", "r") as f:
        data = load(f)
        assert data["origprob"]["problem_name"] == "model"

    os.remove("statistics.json")


def test_getNSolsFound(make_optimized_model_with_fixed_primal_solutions):
    all_feasible_solutions = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2)
    ]
    full_model = make_optimized_model_with_fixed_primal_solutions(all_feasible_solutions)
    n_sols_found = full_model.getNSolsFound()

    # here we now the exact number of feasible soutions, none will be found during optimization
    assert n_sols_found == len(all_feasible_solutions)  

    # Test with a subset of (non-optimal) feasible solutions
    feasible_solutions = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]

    subset_model = make_optimized_model_with_fixed_primal_solutions(feasible_solutions)
    # none of the provided feasible solutions is optimal, so we expect at least one more solution to be found during optimization
    assert subset_model.getNSolsFound() >= len(feasible_solutions) + 1


def test_getPrimalDualIntegral(optimized_model):
    primal_dual_integral = optimized_model.getPrimalDualIntegral()

    assert isinstance(primal_dual_integral, float)


def test_getNRuns(optimized_model):
    n_runs = optimized_model.getNRuns()

    assert isinstance(n_runs, int)
    assert n_runs >= 1


def test_getNReoptRuns(optimized_model):
    n_reopt_runs = optimized_model.getNReoptRuns()

    assert isinstance(n_reopt_runs, int)
    assert n_reopt_runs >= 0


def test_getNObjlimLeaves(optimized_model):
    n_objlim_leaves = optimized_model.getNObjlimLeaves()

    assert isinstance(n_objlim_leaves, int)
    assert n_objlim_leaves >= 0


def test_addNNodes(optimized_model):
    n_nodes_to_add = 5
    initial_n_nodes = optimized_model.getNTotalNodes()
    optimized_model.addNNodes(n_nodes_to_add)
    new_n_nodes = optimized_model.getNTotalNodes()

    assert new_n_nodes == initial_n_nodes + n_nodes_to_add


def test_getMaxTotalDepth(optimized_model):
    max_total_depth = optimized_model.getMaxTotalDepth()
    total_depth = optimized_model.getMaxDepth()

    assert isinstance(max_total_depth, int)
    assert max_total_depth >= 0
    assert max_total_depth >= total_depth


def test_getNBacktracks(optimized_model):
    n_backtracks = optimized_model.getNBacktracks()

    assert isinstance(n_backtracks, int)
    assert n_backtracks >= 0


def test_getAvgLowerbound(optimized_model):
    avg_lowerbound = optimized_model.getAvgLowerbound()
    leaves, children, siblings = optimized_model.getOpenNodes()
    open_nodes = leaves + children + siblings
    manual_avg_lowerbound = np.mean(
        [node.getLowerbound() for node in open_nodes]
        + [optimized_model.getFocusNode().getLowerbound()]
    )

    assert isinstance(avg_lowerbound, float)
    assert optimized_model.isEQ(manual_avg_lowerbound, avg_lowerbound)


def test_getAvgDualbound(optimized_model):
    avg_dualbound = optimized_model.getAvgDualbound()
    avg_lowerbound = optimized_model.getAvgLowerbound()

    assert isinstance(avg_dualbound, float)
    assert (
        optimized_model.isEQ(avg_dualbound, avg_lowerbound)
        or optimized_model.isEQ(avg_dualbound, -avg_lowerbound)
    )


def test_getDeterministicTime(optimized_model):
    det_time = optimized_model.getDeterministicTime()

    assert isinstance(det_time, float)
    assert det_time >= 0.0


def test_getUpperbound(optimized_model):
    upperbound = optimized_model.getUpperbound()
    lowerbound = optimized_model.getLowerbound()

    assert isinstance(upperbound, float)
    assert optimized_model.isGE(upperbound, lowerbound)


def test_getFirstPrimalBound(make_optimized_model_with_fixed_primal_solutions):
    # Test with a subset of (non-optimal) feasible solutions
    feasible_solutions = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    model = make_optimized_model_with_fixed_primal_solutions(feasible_solutions)
    first_primalbound = model.getFirstPrimalBound()
    primalbound = model.getPrimalbound()

    assert isinstance(first_primalbound, float)
    # SCIP will evaluate provided feasible solutions first and find the first feasible solution with objective value equal to 1
    # subsequently, SCIP will find the optimal solution with objective value equal to 2, which is the upper bound of the model
    assert model.isLT(first_primalbound, primalbound)


def test_getLowerboundRoot(optimized_model):
    lowerbound_root = optimized_model.getLowerboundRoot()
    lowerbound = optimized_model.getLowerbound()

    assert isinstance(lowerbound_root, float)
    assert optimized_model.isLE(lowerbound_root, lowerbound)
