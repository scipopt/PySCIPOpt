"""
Structured optimization progress tracing helpers.

This recipe records selected solving progress events as dictionaries in
``model.data["trace"]``. Each progress record includes the solving time, primal
bound, dual bound, gap, node count, and number of solutions at the time the
event was observed.

Use ``structured_optimization_trace(model, path=...)`` as a context manager
when the trace should be scoped to one optimization run. It records events in
memory, optionally writes the same records as JSONL, and appends a final
``run_end`` record when the context exits. If the context exits with a Python
exception, the ``run_end`` record includes exception metadata and the exception
is re-raised.

Example:
    with structured_optimization_trace(model, path="trace.jsonl"):
        model.optimize()

Use ``attach_structured_optimization_trace(model)`` for simple in-memory
tracing with an event handler attached directly to the model. This API does not
manage a Python finalization scope, so it does not emit a final ``run_end``
record.

Example:
    attach_structured_optimization_trace(model)
    model.optimize()
    trace = model.data["trace"]
"""

import json

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model

_TRACE_HANDLER_KEY = "_structured_optimization_trace_handler"


class _TraceEventhdlr(Eventhdlr):
    def __init__(self):
        self.trace = None
        self.write_run_end_active = False

    def eventinit(self):
        for event_type in (
            SCIP_EVENTTYPE.BESTSOLFOUND,
            SCIP_EVENTTYPE.DUALBOUNDIMPROVED,
        ):
            self.model.catchEvent(event_type, self)

    def eventexec(self, event):
        if self.trace is not None:
            self.trace._handle_event(event)


def _attach_trace_handler(model: Model, trace):
    if not hasattr(model, "data") or model.data is None:
        model.data = {}

    handler = model.data.get(_TRACE_HANDLER_KEY)
    if handler is None:
        handler = _TraceEventhdlr()
        model.includeEventhdlr(
            handler,
            "structured_trace",
            "Structured optimization trace handler",
        )
        model.data[_TRACE_HANDLER_KEY] = handler

    if handler.trace is not None:
        if not handler.trace.write_run_end and not trace.write_run_end:
            model.data["trace"] = []
            return handler

        raise RuntimeError(
            "structured optimization trace is already active for this model"
        )

    model.data["trace"] = []
    handler.trace = trace

    return handler


class _StructuredOptimizationTrace:
    """Internal trace controller shared by both public APIs."""

    def __init__(self, model: Model, path=None, write_run_end=True):
        self.model = model
        self.path = path
        self.write_run_end = write_run_end
        self._fh = None
        self._handler = None

    def __enter__(self):
        if self.path is not None:
            self._fh = open(self.path, "w", encoding="utf-8")

        self._handler = _attach_trace_handler(self.model, self)

        if self.write_run_end:
            self._handler.write_run_end_active = True

        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            fields = {"status": "finished"}
        else:
            fields = {
                "status": "exception",
                "exception": exc_type.__name__,
                "message": str(exc) if exc is not None else None,
            }

        try:
            if self.write_run_end:
                self._write_event("run_end", fields)
        finally:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

            if self._handler is not None and self._handler.trace is self:
                self._handler.trace = None
                if self.write_run_end:
                    self._handler.write_run_end_active = False

        return False

    def _handle_event(self, event):
        event_type = event.getType()
        if event_type == SCIP_EVENTTYPE.BESTSOLFOUND:
            self._write_snapshot("bestsol_found")
        elif event_type == SCIP_EVENTTYPE.DUALBOUNDIMPROVED:
            self._write_snapshot("dualbound_improved")

    def _snapshot_now(self):
        return {
            "time": self.model.getSolvingTime(),
            "primalbound": self.model.getPrimalbound(),
            "dualbound": self.model.getDualbound(),
            "gap": self.model.getGap(),
            "nodes": self.model.getNNodes(),
            "nsol": self.model.getNSols(),
        }

    def _write_snapshot(self, event_type):
        snapshot = self._snapshot_now()
        self._write_event(event_type, snapshot)

    def _write_event(self, event_type, fields=None):
        event = {"type": event_type}
        if fields is not None:
            event.update(fields)

        self.model.data["trace"].append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()


def structured_optimization_trace(model: Model, path=None):
    """
    Return a context manager for structured optimization progress tracing.

    The context manager records progress events in ``model.data["trace"]``. If
    ``path`` is given, it also writes each record as one JSON object per line and
    flushes after every write. On exit, it appends a final ``run_end`` record and
    closes the JSONL output, if any.

    Args:
        model: SCIP Model.
        path: Optional JSONL output path. If None, records are only stored in
            ``model.data["trace"]``.

    Returns:
        A context manager that traces optimization progress for ``model``.
    """
    return _StructuredOptimizationTrace(model, path=path, write_run_end=True)


def attach_structured_optimization_trace(model: Model):
    """
    Attach an event handler that records structured optimization progress.

    This attach-style API records progress events in ``model.data["trace"]`` and
    returns the same model. It is intended for simple in-memory tracing. Use
    ``structured_optimization_trace(model, path=...)`` when JSONL output or a
    final ``run_end`` record is required.

    Args:
        model: SCIP Model.

    Returns:
        The same model with the structured trace event handler attached.
    """
    trace = _StructuredOptimizationTrace(model, write_run_end=False)
    _attach_trace_handler(model, trace)

    return model
