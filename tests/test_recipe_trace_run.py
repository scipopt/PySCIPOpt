import json
from random import randint

import pytest
from helpers.utils import bin_packing_model

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr
from pyscipopt.recipes.trace_run import optimizeNogilTrace, optimizeTrace


@pytest.fixture(
    params=[optimizeTrace, optimizeNogilTrace], ids=["optimize", "optimize_nogil"]
)
def optimize(request):
    return request.param


def test_trace_run_in_memory(optimize):
    model = bin_packing_model(sizes=[randint(1, 40) for _ in range(120)], capacity=50)
    model.setParam("limits/time", 5)

    model.data = {"test": True}

    optimize(model, path=None)

    assert "test" in model.data
    assert "trace" in model.data

    required_fields = {"time", "primalbound", "dualbound", "gap", "nodes", "nsol"}

    types = [r["type"] for r in model.data["trace"]]
    assert ("bestsol_found" in types) or ("dualbound_improved" in types)

    for record in model.data["trace"]:
        if record["type"] != "run_end":
            assert required_fields <= set(record.keys())

    primalbounds = [r["primalbound"] for r in model.data["trace"] if "primalbound" in r]
    for i in range(1, len(primalbounds)):
        assert primalbounds[i] <= primalbounds[i - 1]

    dualbounds = [r["dualbound"] for r in model.data["trace"] if "dualbound" in r]
    for i in range(1, len(dualbounds)):
        assert dualbounds[i] >= dualbounds[i - 1]

    assert "run_end" in types


def test_trace_run_file_output(optimize, tmp_path):
    model = bin_packing_model(sizes=[randint(1, 40) for _ in range(120)], capacity=50)
    model.setParam("limits/time", 5)

    path = tmp_path / "trace.jsonl"

    optimize(model, path=str(path))

    assert path.exists()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) > 0

    types = [r["type"] for r in records]
    assert "run_end" in types


class _InterruptOnBest(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        self.model.interruptSolve()


def test_optimize_with_trace_records_run_end_on_interrupt(optimize):
    model = bin_packing_model(
        sizes=[randint(1, 40) for _ in range(120)],
        capacity=50,
    )
    model.setParam("limits/time", 5)

    model.includeEventhdlr(_InterruptOnBest(), "stopper", "Interrupt on bestsol")

    optimize(model, path=None)

    types = [r["type"] for r in model.data["trace"]]
    assert "bestsol_found" in types
    assert "run_end" in types
