from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model


def attach_structured_optimization_trace(model: Model):
    """
    Attaches an event handler that records optimization progress in structured JSONL format.

    Args:
        model: SCIP Model
    """

    class _TraceEventhdlr(Eventhdlr):
        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
            self.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, self)

        def eventexec(self, event):
            record = {
                "time": self.model.getSolvingTime(),
                "primalbound": self.model.getPrimalbound(),
                "dualbound": self.model.getDualbound(),
                "gap": self.model.getGap(),
                "nodes": self.model.getNNodes(),
                "nsol": self.model.getNSols(),
            }
            self.model.data["trace"].append(record)

    if not hasattr(model, "data") or model.data is None:
        model.data = {}
    model.data["trace"] = []

    hdlr = _TraceEventhdlr()
    model.includeEventhdlr(
        hdlr, "structured_trace", "Structured optimization trace handler"
    )

    return model
