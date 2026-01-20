import json
from random import randint

import pytest
from helpers.utils import bin_packing_model

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr
from pyscipopt.recipes.trace_run import trace_run


def test_trace_run_in_memory():
    model = bin_packing_model(sizes=[randint(1, 40) for _ in range(120)], capacity=50)
    model.setParam("limits/time", 5)

    model.data = {"test": True}

    with trace_run(model, path=None):
        model.optimize()

    assert "test" in model.data
    assert "trace" in model.data

    required_fields = {"time", "primalbound", "dualbound", "gap", "nodes", "nsol"}
    for record in model.data["trace"]:
        assert required_fields <= set(record.keys())

    primalbounds = [r["primalbound"] for r in model.data["trace"]]
    for i in range(1, len(primalbounds)):
        assert primalbounds[i] <= primalbounds[i - 1]

    dualbounds = [r["dualbound"] for r in model.data["trace"]]
    for i in range(1, len(dualbounds)):
        assert dualbounds[i] >= dualbounds[i - 1]

    types = [r["type"] for r in model.data["trace"]]
    assert "run_end" in types


def test_trace_run_file_output(tmp_path):
    model = bin_packing_model(sizes=[randint(1, 40) for _ in range(120)], capacity=50)
    model.setParam("limits/time", 5)

    path = tmp_path / "trace.jsonl"

    with trace_run(model, path=str(path)):
        model.optimize()

    assert path.exists()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) > 0

    types = [r["type"] for r in records]
    assert "run_end" in types


class _StopOnBest(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        # SCIPが想定している安全な中断
        self.model.interruptSolve()

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)


def test_trace_run_forced_exception_after_bestsol():
    model = bin_packing_model(sizes=[randint(1, 40) for _ in range(120)], capacity=50)

    model.setParam("limits/time", 5)

    stopper = _StopOnBest()
    model.includeEventhdlr(stopper, "stopper", "Stop on bestsol")

    with pytest.raises(RuntimeError):
        with trace_run(model, path=None):
            model.optimize()
            raise RuntimeError("forced after interrupt")

    types = [r["type"] for r in model.data["trace"]]
    assert "bestsol_found" in types
    assert "run_end" in types
