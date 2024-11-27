"""
This example show how to retrieve the primal and dual solutions during the optimization process
and plot them as a function of time. The model is about gas transportation and can be found in
PySCIPOpt/tests/helpers/utils.py

It makes use of the attach_primal_dual_evolution_eventhdlr recipe.

Requires matplotlib, and may require PyQt6 to show the plot.
"""

from pyscipopt import Model

def plot_primal_dual_evolution(model: Model):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to plot the solution. Try running `pip install matplotlib` in the command line.\
                          You may also need to install PyQt6 to show the plot.")

    assert model.data["primal_log"], "Could not find any feasible solutions"
    time_primal, val_primal = map(list,zip(*model.data["primal_log"]))
    time_dual, val_dual = map(list,zip(*model.data["dual_log"]))

    
    if time_primal[-1] < time_dual[-1]:
        time_primal.append(time_dual[-1])
        val_primal.append(val_primal[-1])

    if time_primal[-1] > time_dual[-1]:
        time_dual.append(time_primal[-1])
        val_dual.append(val_dual[-1])
        
    plt.plot(time_primal, val_primal, label="Primal bound")
    plt.plot(time_dual, val_dual, label="Dual bound")

    plt.legend(loc="best")
    plt.show()

if __name__=="__main__":
    from pyscipopt.recipes.primal_dual_evolution import attach_primal_dual_evolution_eventhdlr
    import os
    import sys

    # just a way to import files from different folders, not important
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tests/helpers')))
    
    from utils import gastrans_model
    
    model = gastrans_model()
    model.data = {}
    attach_primal_dual_evolution_eventhdlr(model)

    model.optimize()
    plot_primal_dual_evolution(model)
