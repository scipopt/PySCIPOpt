import os
from helpers.utils import random_mip_1
from json import load
import pytest


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
