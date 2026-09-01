from pyscipopt.scip import Model
import os
from helpers.utils import random_mip_1
from json import load
import pytest
import numpy as np


@pytest.fixture
def optimized_model():
    # Using small=True for speed across tests
    # finds 1 primal solution
    model = random_mip_1(small=True, node_lim=2400)
    model.optimize()
    return model


# very simple model with 2 primal solutions
# created because getting more than 1 primal solution from random_mip_1 requres setting a large node limit, which slows down the tests
@pytest.fixture
def optimized_model_with_primal_solutions():
    model = Model()

    x = model.addVar(vtype="I", lb=0, ub=5, name="x")
    y = model.addVar(vtype="I", lb=0, ub=5, name="y")
    z = model.addVar(vtype="I", lb=0, ub=5, name="z")

    model.addCons(x + y + z <= 5)
    model.addCons(x >= 1)
    model.addCons(y >= 1)

    model.setObjective(x + y + z, "maximize")

    model.optimize()

    return model


def test_statistics_json(optimized_model):
    optimized_model.writeStatisticsJson("statistics.json")

    with open("statistics.json", "r") as f:
        data = load(f)
        assert data["origprob"]["problem_name"] == "model"

    os.remove("statistics.json")


def test_getNSolsFound(optimized_model, optimized_model_with_primal_solutions):
    sols = optimized_model.getNSolsFound()

    assert sols >= 1

    sols_model_with_primals = optimized_model_with_primal_solutions.getNSolsFound()

    assert sols_model_with_primals >= 2


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


def test_addNNodes(optimized_model):
    initial_n_nodes = optimized_model.getNTotalNodes()
    optimized_model.addNNodes(5)
    new_n_nodes = optimized_model.getNTotalNodes()

    assert new_n_nodes == initial_n_nodes + 5


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
    manual_avg_lowerbound = 0.0
    if len(open_nodes) > 0:
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


def test_getFirstPrimalBound(optimized_model_with_primal_solutions):
    first_primal = optimized_model_with_primal_solutions.getFirstPrimalBound()
    upperbound = optimized_model_with_primal_solutions.getUpperbound()

    assert isinstance(first_primal, float)
    assert optimized_model_with_primal_solutions.isGT(first_primal, upperbound)


def test_getLowerboundRoot(optimized_model):
    lowerbound_root = optimized_model.getLowerboundRoot()
    lowerbound = optimized_model.getLowerbound()

    assert isinstance(lowerbound_root, float)
    assert optimized_model.isLE(lowerbound_root, lowerbound)
