from concurrent.futures import ThreadPoolExecutor, as_completed
from pyscipopt import Model
from helpers.utils import random_mip_1

N_Threads = 4


def test_optimalNogil():
    ori_model = random_mip_1(disable_sepa=False, disable_heur=False, disable_presolve=False, node_lim=2000, small=True) 
    models = [Model(sourceModel=ori_model) for _ in range(N_Threads)]
    for i in range(N_Threads):
        models[i].setParam("randomization/permutationseed", i)

    ori_model.optimize()

    with ThreadPoolExecutor(max_workers=N_Threads) as executor:
        futures = [executor.submit(Model.optimizeNogil, model) for model in models]
        for future in as_completed(futures):
            pass
        for model in models:
            assert model.getStatus() == "optimal"    
            assert abs(ori_model.getObjVal() - model.getObjVal()) < 1e-6

