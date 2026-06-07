import json

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model

_TRACE_HANDLER_KEY = "_structured_optimization_trace_handler"


class _TraceEventhdlr(Eventhdlr):
    def __init__(self):
        self.trace = None
        self.write_run_end_active = False
        self._caught_events = set()

    def eventinit(self):
        for event_type in (
            SCIP_EVENTTYPE.BESTSOLFOUND,
            SCIP_EVENTTYPE.DUALBOUNDIMPROVED,
        ):
            if event_type not in self._caught_events:
                self.model.catchEvent(event_type, self)
                self._caught_events.add(event_type)

    def eventexec(self, event):
        if self.trace is not None:
            self.trace._handle_event(event)


class _StructuredOptimizationTrace:
    """Internal trace controller shared by both public APIs."""

    def __init__(self, model: Model, path=None, write_run_end=True):
        self.model = model
        self.path = path
        self.write_run_end = write_run_end
        self._fh = None
        self._handler = None
        self._last_snapshot: dict[str, object] = {}

    def __enter__(self):
        if not hasattr(self.model, "data") or self.model.data is None:
            self.model.data = {}

        self._handler = self.model.data.get(_TRACE_HANDLER_KEY)
        if self._handler is None:
            self._handler = _TraceEventhdlr()
            self.model.includeEventhdlr(
                self._handler,
                "structured_trace",
                "Structured optimization trace handler",
            )
            self.model.data[_TRACE_HANDLER_KEY] = self._handler

        if self._handler.write_run_end_active:
            raise RuntimeError(
                "structured optimization trace is already active for this model"
            )

        self.model.data["trace"] = []

        if self.path is not None:
            self._fh = open(self.path, "w", encoding="utf-8")

        self._handler.trace = self
        if self.write_run_end:
            self._handler.write_run_end_active = True

        return self

    def __exit__(self, exc_type, exc, tb):
        fields = {}
        if self._last_snapshot:
            fields.update(self._last_snapshot)

        if exc_type is None:
            fields["status"] = "finished"
        else:
            fields.update(
                {
                    "status": "exception",
                    "exception": exc_type.__name__,
                    "message": str(exc) if exc is not None else None,
                }
            )

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
        self._last_snapshot = snapshot
        self._write_event(event_type, snapshot)

    def _write_event(self, event_type, fields=None):
        event = {"type": event_type}
        if fields:
            event.update(fields)

        self.model.data["trace"].append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()


def structured_optimization_trace(model: Model, path=None):
    """
    Create a context manager for structured optimization progress tracing.

    Records progress events in ``model.data["trace"]`` while the model is
    optimized inside the context. If ``path`` is given, the same records are
    written as JSONL and flushed after each write.

    On exit, appends a final ``run_end`` record and closes JSONL output. If the
    context exits with a Python exception, the ``run_end`` record includes
    exception metadata and the exception is re-raised.

    Args:
        model: SCIP Model.
        path: Optional JSONL output path. If None, records are only stored in
            ``model.data["trace"]``.

    Usage:
        with structured_optimization_trace(model, path="trace.jsonl"):
            model.optimize()

        with structured_optimization_trace(model):
            model.optimizeNogil()
    """
    return _StructuredOptimizationTrace(model, path=path, write_run_end=True)


def attach_structured_optimization_trace(model: Model):
    """
    Attaches an event handler that records structured optimization progress.

    This is the attach-style API for simple in-memory tracing. It records
    progress events in ``model.data["trace"]`` but does not manage a Python
    finalization scope, so it does not emit a final ``run_end`` record.

    Use ``structured_optimization_trace(model, path=...)`` as a context manager
    when JSONL output, ``run_end`` records, flushing, closing, or exception
    finalization are required.

    Args:
        model: SCIP Model

    Usage:
        attach_structured_optimization_trace(model)
        model.optimize()
        trace = model.data["trace"]
    """
    trace = _StructuredOptimizationTrace(model, write_run_end=False)
    trace.__enter__()

    return model
