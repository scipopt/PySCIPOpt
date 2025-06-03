from pyscipopt import Model
from helpers.utils import random_mip_1

def test_statistics_json():
    model = random_mip_1()
    model.optimize()
    json_output = model.writeStatisticsJson("statistics.json")
