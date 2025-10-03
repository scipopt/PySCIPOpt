"""
Tests for Model.aggregateVars (wrapper around SCIPaggregateVars).
"""

from pyscipopt import (
    Model,
    Presol,
    SCIP_PRESOLTIMING,
    SCIP_RESULT,
)


class _AggPresol(Presol):
    """
    Minimal presolver that aggregates two given variables and records the flags.
    """

    def __init__(self, varx, vary, scalarx, scalary, rhs):
        self._args = (varx, vary, scalarx, scalary, rhs)
        self.last = None  # (infeasible, redundant, aggregated)

    def presolexec(self, nrounds, presoltiming):
        x, y, ax, ay, rhs = self._args
        infeas, redun, aggr = self.model.aggregateVars(x, y, ax, ay, rhs)
        self.last = (bool(infeas), bool(redun), bool(aggr))
        # return SUCCESS to indicate presolver did work
        return {"result": SCIP_RESULT.SUCCESS}


def _build_model_xy(vtype="C", lbx=0.0, ubx=10.0, lby=0.0, uby=10.0):
    """
    Build a tiny model with two variables x and y.
    """
    m = Model("agg-vars-test")
    m.hideOutput()
    x = m.addVar(name="x", vtype=vtype, lb=lbx, ub=ubx)
    y = m.addVar(name="y", vtype=vtype, lb=lby, ub=uby)
    # trivial objective to have a complete model
    m.setMaximize()
    return m, x, y


def test_aggregate_vars_success():
    """
    Aggregation succeeds for x - y = 0 on continuous variables with
    compatible bounds, when called from a presolver.
    """
    model, x, y = _build_model_xy(
        vtype="C", lbx=0.0, ubx=10.0, lby=0.0, uby=10.0
    )

    presol = _AggPresol(x, y, 1.0, -1.0, 0.0)
    model.includePresol(
        presol,
        "agg-test",
        "aggregate x and y",
        priority=10**7,
        maxrounds=1,
        timing=SCIP_PRESOLTIMING.FAST,
    )

    model.presolve()
    assert presol.last is not None
    infeasible, redundant, aggregated = presol.last

    assert not infeasible
    assert aggregated
    # model should stay consistent
    model.optimize()