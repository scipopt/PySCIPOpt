import json

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr


class _TraceRun:
    """
    Record optimization progress in real time while the solver is running.

    Args
    ----
    model: pyscipopt.Model
    path: str | None
        - None: in-memory only
        - str : also write JSONL (one JSON object per line) for streaming/real-time consumption

    Returns
    -------
    None
        Updates `model.data["trace"]` as a side effect.

    Usage
    -----
    optimizeTrace(model)                     # real-time in-memory trace
    optimizeTrace(model, path="trace.jsonl")      # real-time JSONL stream + in-memory
    optimizeNogilTrace(model, path="trace.jsonl") # nogil variant
    """

    def __init__(self, model, path=None):
        self.model = model
        self.path = path
        self._fh = None
        self._handler = None
        self._caught_events = set()
        self._last_snapshot = {}

    def __enter__(self):
        if not hasattr(self.model, "data") or self.model.data is None:
            self.model.data = {}
        self.model.data["trace"] = []

        if self.path is not None:
            self._fh = open(self.path, "w")

        class _TraceEventhdlr(Eventhdlr):
            def eventinit(hdlr):
                for et in (
                    SCIP_EVENTTYPE.BESTSOLFOUND,
                    SCIP_EVENTTYPE.DUALBOUNDIMPROVED,
                ):
                    self.model.catchEvent(et, hdlr)
                    self._caught_events.add(et)

            def eventexec(hdlr, event):
                et = event.getType()
                if et == SCIP_EVENTTYPE.BESTSOLFOUND:
                    snapshot = self._snapshot_now()
                    self._last_snapshot = snapshot
                    self._write_event("bestsol_found", fields=snapshot, flush=True)
                elif et == SCIP_EVENTTYPE.DUALBOUNDIMPROVED:
                    snapshot = self._snapshot_now()
                    self._last_snapshot = snapshot
                    self._write_event(
                        "dualbound_improved", fields=snapshot, flush=False
                    )

        self._handler = _TraceEventhdlr()
        self.model.includeEventhdlr(
            self._handler, "realtime_trace_jsonl", "Realtime trace jsonl handler"
        )

        return self

    def __exit__(self, exc_type, exc, tb):
        fields = {}
        if self._last_snapshot:
            fields.update(self._last_snapshot)

        if exc_type is not None:
            fields.update(
                {
                    "status": "exception",
                    "exception": exc_type.__name__,
                    "message": str(exc) if exc is not None else None,
                }
            )

        try:
            self._write_event("run_end", fields=fields, flush=True)
        finally:
            if self._fh:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

            if self._handler is not None:
                for et in self._caught_events:
                    try:
                        self.model.dropEvent(et, self._handler)
                    except Exception:
                        pass  # Best-effort cleanup; continue dropping remaining events
                self._caught_events.clear()
                self._handler = None

        return False

    def _snapshot_now(self) -> dict:
        return {
            "time": self.model.getSolvingTime(),
            "primalbound": self.model.getPrimalbound(),
            "dualbound": self.model.getDualbound(),
            "gap": self.model.getGap(),
            "nodes": self.model.getNNodes(),
            "nsol": self.model.getNSols(),
        }

    def _write_event(self, event_type, fields=None, flush=True):
        event = {"type": event_type}
        if fields:
            event.update(fields)

        self.model.data["trace"].append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event) + "\n")
            if flush:
                self._fh.flush()


def optimizeTrace(model, path=None):
    with _TraceRun(model, path):
        model.optimize()


def optimizeNogilTrace(model, path=None):
    with _TraceRun(model, path):
        model.optimizeNogil()
