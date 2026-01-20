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
        self.model.data.setdefault("trace", [])

        if self.path is not None:
            self._fh = open(self.path, "w")

        class _TraceEventhdlr(Eventhdlr):
            def eventinit(s):
                s.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, s)
                s.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, s)

            def eventexec(s, event):
                et = event.getType()
                if et == SCIP_EVENTTYPE.BESTSOLFOUND:
                    self._write_event("bestsol_found", flush=True)
                elif et == SCIP_EVENTTYPE.DUALBOUNDIMPROVED:
                    self._write_event("dualbound_improved", flush=False)

        self._handler = _TraceEventhdlr()
        self.model.includeEventhdlr(self._handler, "trace_run", "Trace run handler")

        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._write_event("run_end", flush=True)
        finally:
            if self._fh:
                try:
                    self._fh.close()
                finally:
                    self._fh = None
            if self._handler is not None:
                try:
                    self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self._handler)
                except Exception:
                    pass
                try:
                    self.model.dropEvent(
                        SCIP_EVENTTYPE.DUALBOUNDIMPROVED, self._handler
                    )
                except Exception:
                    pass
                self._handler = None

        return False

    def _write_event(self, event_type, flush=True):
        event = {
            "type": event_type,
            "time": self.model.getSolvingTime(),
            "primalbound": self.model.getPrimalbound(),
            "dualbound": self.model.getDualbound(),
            "gap": self.model.getGap(),
            "nodes": self.model.getNNodes(),
            "nsol": self.model.getNSols(),
        }
        if event_type == "run_end":
            status = self.model.getStatus()
            event["status"] = getattr(status, "name", None) or repr(status)
        self.model.data["trace"].append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event) + "\n")
            if flush:
                self._fh.flush()


def trace_run(model, path=None):
    return _TraceRun(model, path)
