from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, Eventhdlr

def get_primal_dual_evolution(model: Model):

    class GapEventhdlr(Eventhdlr):

        def eventinit(self): # we want to collect best primal solutions and best dual solutions
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
            self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
            self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
            

        def eventexec(self, event):
            # if a new best primal solution was found, we save when it was found and also its objective
            if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
                self.model.data["primal_solutions"].append((self.model.getSolvingTime(), self.model.getPrimalbound()))
            
            if not self.model.data["dual_solutions"]:
                self.model.data["dual_solutions"].append((self.model.getSolvingTime(), self.model.getDualbound()))
            
            if self.model.getObjectiveSense() == "minimize":
                if self.model.isGT(self.model.getDualbound(), self.model.data["dual_solutions"][-1][1]):
                    self.model.data["dual_solutions"].append((self.model.getSolvingTime(), self.model.getDualbound()))
            else:
                if self.model.isLT(self.model.getDualbound(), self.model.data["dual_solutions"][-1][1]):
                    self.model.data["dual_solutions"].append((self.model.getSolvingTime(), self.model.getDualbound()))

    if not hasattr(model, "data"):
        model.data = {}

    model.data["primal_solutions"] = []
    model.data["dual_solutions"] = []
    hdlr = gapEventhdlr()
    model.includeEventhdlr(hdlr, "gapEventHandler", "Event handler which collects primal and dual solution evolution")

    return model

def plot_primal_dual_evolution(model: Model):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise("matplotlib is required to plot the solution. Try running `pip install matplotlib` in the command line.")

    time_primal, val_primal = zip(*model.data["primal_solutions"])
    plt.plot(time_primal, val_primal, label="Primal bound")
    time_dual, val_dual = zip(*model.data["dual_solutions"])
    plt.plot(time_dual, val_dual, label="Dual bound")

    plt.legend(loc="best")
    plt.show()