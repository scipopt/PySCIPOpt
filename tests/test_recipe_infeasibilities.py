from pyscipopt import Model
from pyscipopt.recipes.infeasibilities import get_infeasible_constraints


def test_get_infeasible_constraints():
    m = Model()

    x = m.addVar(lb=0)
    m.setObjective(2*x)

    m.addCons(x <= 4)
    
    n_infeasibilities_detected = get_infeasible_constraints(m)[0]
    assert n_infeasibilities_detected == 0

    m.addCons(x <= -1)

    n_infeasibilities_detected = get_infeasible_constraints(m)[0]
    assert n_infeasibilities_detected == 1

    m.addCons(x == 2)

    n_infeasibilities_detected = get_infeasible_constraints(m)[0]
    assert n_infeasibilities_detected == 1

    m.addCons(x == -4)

    n_infeasibilities_detected = get_infeasible_constraints(m)[0]
    assert n_infeasibilities_detected == 2