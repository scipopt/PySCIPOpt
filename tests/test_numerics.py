from pyscipopt import Model
import pytest

def test_numerical_checks():
    m = Model()

    m.setParam("numerics/epsilon", 1e-10)
    m.setParam("numerics/feastol", 1e-3)

    assert m.isFeasEQ(1, 1.00001)
    assert not m.isEQ(1, 1.00001)
    
    assert m.isFeasLE(1, 0.99999) 
    assert not m.isLE(1, 0.99999) 

    assert m.isFeasGE(1, 1.00001)
    assert not m.isGE(1, 1.00001)

    assert not m.isFeasGT(1, 0.99999)
    assert m.isGT(1, 0.99999)
