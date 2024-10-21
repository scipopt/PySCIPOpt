from pyscipopt import Model
from helpers.utils import random_mip_1


def test_copy():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    s.setObjective(4.0 * y, clear = False)

    c = s.addCons(x + 2 * y >= 1.0)

    s2 = Model(sourceModel=s)

    # solve problems
    s.optimize()
    s2.optimize()

    assert s.getObjVal() == s2.getObjVal()


def test_copyModel():
    ori_model = random_mip_1(disable_sepa=False, disable_huer=False, disable_presolve=False, node_lim=2000, small=False) 
    cpy_model = ori_model.copyModel()
    sub_model = Model(sourceModel=ori_model)
    
    assert len(ori_model.getParams()) == len(cpy_model.getParams()) > len(sub_model.getParams())
    assert ori_model.getNVars() == cpy_model.getNVars()
    assert ori_model.getNConss() == cpy_model.getNConss()

    ori_model.optimize()
    cpy_model.optimize()
    assert ori_model.getStatus() == cpy_model.getStatus() == "optimal"
    assert ori_model.getObjVal() == cpy_model.getObjVal()


def test_addCopyModelSol_BestSol_Sols():
    ori_model = random_mip_1(disable_sepa=False, disable_huer=False, disable_presolve=False, node_lim=2000, small=False)
    cpy_model0 = ori_model.copyModel()
    cpy_model1 = ori_model.copyModel()
    cpy_model2 = ori_model.copyModel()
    
    ori_model.optimize()
    solution = ori_model.getBestSol()

    cpy_model0.addCopyModelSol(solution)
    cpy_model1.addCopyModelBestSol(ori_model)
    cpy_model2.addCopyModelSols(ori_model)

    assert cpy_model0.getNSols() == 1
    assert cpy_model1.getNSols() == 1
    assert cpy_model2.getNSols() == ori_model.getNSols() >= 1

    cpy_model0.optimize()
    cpy_model1.optimize()
    cpy_model2.optimize()

    assert ori_model.getStatus() == "optimal"
    assert cpy_model0.getStatus() == "optimal"
    assert cpy_model1.getStatus() == "optimal"
    assert cpy_model2.getStatus() == "optimal"
    assert abs(ori_model.getObjVal() - cpy_model0.getObjVal()) < 1e-6
    assert abs(ori_model.getObjVal() - cpy_model1.getObjVal()) < 1e-6
    assert abs(ori_model.getObjVal() - cpy_model2.getObjVal()) < 1e-6
    
