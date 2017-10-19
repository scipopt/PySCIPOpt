from pyscipopt import Model
import pytest


def create_model_with_one_optimum():
    m = Model()
    x = m.addVar("x", vtype = 'C', obj = 1.0)
    y = m.addVar("y", vtype = 'C', obj = 2.0)
    c = m.addCons(x + 2 * y >= 1.0)
    m.data = [x,y], [c]
    return m

def test_writeProblem(tmpdir):
    model = create_model_with_one_optimum()
    model.optimize()
    assert model.getStatus() == "optimal", "model could not be optimized"

    probfile = tmpdir.join("x.cip")
    model.writeBestSol(str(probfile))
    assert probfile.exists(), "no problem file was written"
