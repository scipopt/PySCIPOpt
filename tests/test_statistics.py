import os
from helpers.utils import random_mip_1
from json import load
import pytest
import numpy as np


@pytest.fixture
def optimized_model():
    model = random_mip_1(small=True)  # Using small=True for speed across tests
    model.optimize()
    return model


def test_statistics_json(optimized_model):
    optimized_model.writeStatisticsJson("statistics.json")

    with open("statistics.json", "r") as f:
        data = load(f)
        assert data["origprob"]["problem_name"] == "model"

    os.remove("statistics.json")


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
            [node.getLowerbound() for node in open_nodes] + [optimized_model.getFocusNode().getLowerbound()]
        )

    assert isinstance(avg_lowerbound, float)
    assert manual_avg_lowerbound == pytest.approx(avg_lowerbound)


def test_getAvgDualbound(optimized_model):
    avg_dualbound = optimized_model.getAvgDualbound()
    avg_lowerbound = optimized_model.getAvgLowerbound()

    assert isinstance(avg_dualbound, float)
    assert avg_dualbound == pytest.approx(avg_lowerbound) or avg_dualbound == pytest.approx(-avg_lowerbound)


def test_getDeterministicTime(optimized_model):
    det_time = optimized_model.getDeterministicTime()

    assert isinstance(det_time, float)
    assert det_time >= 0.0


def test_getFirstPrimalBound(optimized_model):
    first_primal = optimized_model.getFirstPrimalBound()
    
    assert isinstance(first_primal, float)
