import json

import pytest
from helpers.utils import bin_packing_model

from pyscipopt.recipes.structured_optimization_trace import (
    attach_structured_optimization_trace,
    structured_optimization_trace,
)


def _model():
    model = bin_packing_model(sizes=list(range(1, 41)) * 3, capacity=50)
    model.setParam("limits/time", 5)
    return model


def _assert_progress_records(records):
    required_fields = {"time", "primalbound", "dualbound", "gap", "nodes", "nsol"}

    for record in records:
        if record["type"] != "run_end":
            assert required_fields <= set(record.keys())

    primalbounds = [r["primalbound"] for r in records if "primalbound" in r]
    for i in range(1, len(primalbounds)):
        assert primalbounds[i] <= primalbounds[i - 1]

    dualbounds = [r["dualbound"] for r in records if "dualbound" in r]
    for i in range(1, len(dualbounds)):
        assert dualbounds[i] >= dualbounds[i - 1]


def test_attach_structured_optimization_trace_in_memory():
    model = _model()
    model.data = {"test": True}

    model = attach_structured_optimization_trace(model)

    assert "test" in model.data
    assert "trace" in model.data

    model.optimize()

    assert model.data["trace"]
    assert all("type" in record for record in model.data["trace"])
    assert "run_end" not in [r["type"] for r in model.data["trace"]]
    _assert_progress_records(model.data["trace"])


@pytest.mark.parametrize("optimize", ["optimize", "optimizeNogil"])
def test_structured_optimization_trace_context_in_memory(optimize):
    model = _model()
    model.data = {"test": True}

    with structured_optimization_trace(model):
        getattr(model, optimize)()

    assert "test" in model.data
    assert "trace" in model.data

    types = [r["type"] for r in model.data["trace"]]
    assert "run_end" in types
    assert model.data["trace"][-1]["type"] == "run_end"
    assert model.data["trace"][-1]["status"] == "finished"
    _assert_progress_records(model.data["trace"])


def test_structured_optimization_trace_file_output(tmp_path):
    model = _model()
    path = tmp_path / "trace.jsonl"

    with structured_optimization_trace(model, path=str(path)):
        model.optimize()

    assert path.exists()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == model.data["trace"]
    assert records[-1]["type"] == "run_end"
    assert records[-1]["status"] == "finished"
    _assert_progress_records(records)


def test_structured_optimization_trace_records_run_end_on_exception():
    model = _model()

    with pytest.raises(ValueError):
        with structured_optimization_trace(model):
            raise ValueError("test error")

    assert model.data["trace"] == [
        {
            "type": "run_end",
            "status": "exception",
            "exception": "ValueError",
            "message": "test error",
        }
    ]


def test_structured_optimization_trace_reuses_handler_for_repeated_contexts(tmp_path):
    model = _model()
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    with structured_optimization_trace(model, path=str(first_path)):
        pass

    handler = model.data["_structured_optimization_trace_handler"]

    with structured_optimization_trace(model, path=str(second_path)):
        pass

    assert model.data["_structured_optimization_trace_handler"] is handler

    first_records = [json.loads(line) for line in first_path.read_text().splitlines()]
    second_records = [json.loads(line) for line in second_path.read_text().splitlines()]

    assert first_records == [{"type": "run_end", "status": "finished"}]
    assert second_records == [{"type": "run_end", "status": "finished"}]


def test_structured_optimization_trace_rejects_nested_contexts():
    model = _model()

    with structured_optimization_trace(model):
        with pytest.raises(RuntimeError):
            with structured_optimization_trace(model):
                pass
