import json

from pyscipopt import SCIP_EVENTTYPE, Eventhdlr


class _TraceRun:
    def __init__(self, model, path=None):
        self.model = model
        self.path = path
        self._fh = None
        self._handler = None

    def __enter__(self):
        if not hasattr(self.model, "data") or self.model.data is None:
            self.model.data = {}
        self.model.data["trace"] = []

        if self.path is not None:
            self._fh = open(self.path, "w", buffering=1)

        class _TraceEventhdlr(Eventhdlr):
            def eventinit(s):
                s.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, s)
                s.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, s)

            def eventexec(s, event):
                self._write_event("solution_update")

            def eventexit(s):
                s.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, s)
                s.model.dropEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, s)

        self._handler = _TraceEventhdlr()
        self.model.includeEventhdlr(self._handler, "trace_run", "Trace run handler")

        return None

    def __exit__(self, exc_type, exc, tb):
        try:
            self._write_event("solve_finish")
        finally:
            if self._fh:
                self._fh.close()
                self._fh = None
            if self._handler is not None:
                self._handler.eventexit()
                self._handler = None

    def _write_event(self, event_type):
        event = {
            "type": event_type,
            "time": self.model.getSolvingTime(),
            "primalbound": self.model.getPrimalbound(),
            "dualbound": self.model.getDualbound(),
            "gap": self.model.getGap(),
            "nodes": self.model.getNNodes(),
            "nsol": self.model.getNSols(),
        }
        self.model.data["trace"].append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()


def trace_run(model, path=None):
    return _TraceRun(model, path)
