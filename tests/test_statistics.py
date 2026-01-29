import os
from helpers.utils import random_mip_1
from json import load

def test_statistics_json():
    model = random_mip_1()
    model.optimize()
    model.writeStatisticsJson("statistics.json")

    with open("statistics.json", "r") as f:
        data = load(f)
        assert data["origprob"]["problem_name"] == "model"
    
    os.remove("statistics.json")

def test_getPrimalDualIntegral():
    model = random_mip_1(small=True)
    model.optimize()
    primal_dual_integral = model.getPrimalDualIntegral()

    assert isinstance(primal_dual_integral, float)
