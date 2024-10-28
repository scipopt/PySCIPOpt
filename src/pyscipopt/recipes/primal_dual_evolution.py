from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, Eventhdlr

def attach_primal_dual_evolution_eventhdlr(model: Model):
    """
    
    """
    class GapEventhdlr(Eventhdlr):

        def eventinit(self): # we want to collect best primal solutions and best dual solutions
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
            self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
            self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
            

        def eventexec(self, event):
            # if a new best primal solution was found, we save when it was found and also its objective
            if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
                self.model.data["primal_log"].append((self.model.getSolvingTime(), self.model.getPrimalbound()))
            
            if not self.model.data["dual_log"]:
                self.model.data["dual_log"].append((self.model.getSolvingTime(), self.model.getDualbound()))
            
            if self.model.getObjectiveSense() == "minimize":
                if self.model.isGT(self.model.getDualbound(), self.model.data["dual_log"][-1][1]):
                    self.model.data["dual_log"].append((self.model.getSolvingTime(), self.model.getDualbound()))
            else:
                if self.model.isLT(self.model.getDualbound(), self.model.data["dual_log"][-1][1]):
                    self.model.data["dual_log"].append((self.model.getSolvingTime(), self.model.getDualbound()))

        def eventexitsol(self):
            if self.model.data["primal_log"][-1] and self.model.getPrimalbound() != self.model.data["primal_log"][-1][1]:
                self.model.data["primal_log"].append((self.model.getSolvingTime(), self.model.getPrimalbound()))
            
            if not self.model.data["dual_log"]:
                self.model.data["dual_log"].append((self.model.getSolvingTime(), self.model.getDualbound()))
            
            if self.model.getObjectiveSense() == "minimize":
                if self.model.isGT(self.model.getDualbound(), self.model.data["dual_log"][-1][1]):
                    self.model.data["dual_log"].append((self.model.getSolvingTime(), self.model.getDualbound()))
            else:
                if self.model.isLT(self.model.getDualbound(), self.model.data["dual_log"][-1][1]):
                    self.model.data["dual_log"].append((self.model.getSolvingTime(), self.model.getDualbound()))


    if not hasattr(model, "data"):
        model.data = {}

    model.data["primal_log"] = []
    model.data["dual_log"] = []
    hdlr = GapEventhdlr()
    model.includeEventhdlr(hdlr, "gapEventHandler", "Event handler which collects primal and dual solution evolution")

    return model