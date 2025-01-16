from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, Eventhdlr

def attach_primal_dual_evolution_eventhdlr(model: Model):
    """
    Attaches an event handler to a given SCIP model that collects primal and dual solutions,
    along with the solving time when they were found.
    The data is saved in model.data["primal_log"] and model.data["dual_log"]. They consist of
    a list of tuples, each tuple containing the solving time and the corresponding solution.

    A usage example can be found in examples/finished/plot_primal_dual_evolution.py. The
    example takes the information provided by this recipe and uses it to plot the evolution
    of the dual and primal bounds over time. 
    """
    class GapEventhdlr(Eventhdlr):

        def eventinit(self): # we want to collect best primal solutions and best dual solutions
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
            self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
            self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
            

        def eventexec(self, event):
            # if a new best primal solution was found, we save when it was found and also its objective
            if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
                self.model.data["primal_log"].append([self.model.getSolvingTime(), self.model.getPrimalbound()])
            
            if not self.model.data["dual_log"]:
                self.model.data["dual_log"].append([self.model.getSolvingTime(), self.model.getDualbound()])
            
            if self.model.getObjectiveSense() == "minimize":
                if self.model.isGT(self.model.getDualbound(), self.model.data["dual_log"][-1][1]):
                    self.model.data["dual_log"].append([self.model.getSolvingTime(), self.model.getDualbound()])
            else:
                if self.model.isLT(self.model.getDualbound(), self.model.data["dual_log"][-1][1]):
                    self.model.data["dual_log"].append([self.model.getSolvingTime(), self.model.getDualbound()])
                    

    if not hasattr(model, "data") or model.data==None:
        model.data = {}

    model.data["primal_log"] = []
    model.data["dual_log"] = []
    hdlr = GapEventhdlr()
    model.includeEventhdlr(hdlr, "gapEventHandler", "Event handler which collects primal and dual solution evolution")

    return model