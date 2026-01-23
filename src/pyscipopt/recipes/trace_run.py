import json

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr


class _TraceRun:
    def __init__(self, model, path=None):
        self.model = model
        self.path = path
        self._fh = None
        self._handler = None

        self._last_snapshot = {}

    def __enter__(self):
        if not hasattr(self.model, "data") or self.model.data is None:
            self.model.data = {}
        self.model.data.setdefault("trace", [])

        if self.path is not None:
            self._fh = open(self.path, "w")

        class _TraceEventhdlr(Eventhdlr):
            def eventinit(s):
                self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, s)
                self.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, s)

            def eventexec(s, event):
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
        self.model.includeEventhdlr(self._handler, "trace_run", "Trace run handler")

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
                for et in (
                    SCIP_EVENTTYPE.BESTSOLFOUND,
                    SCIP_EVENTTYPE.DUALBOUNDIMPROVED,
                ):
                    try:
                        self.model.dropEvent(et, self._handler)
                    except Exception:
                        pass
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
