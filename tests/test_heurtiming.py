import pytest

from pyscipopt import Model, SCIP_HEURTIMING

def test_heurTiming():
    model = Model()
    model.setHeurTiming('rins', SCIP_HEURTIMING.BEFORENODE)
    print("timing of rins: %d\n" % model.getHeurTiming('rins'))
