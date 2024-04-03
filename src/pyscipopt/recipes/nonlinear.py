from pyscipopt import Model


def set_nonlinear_objective(model: Model, expr, sense="minimize"):
    """
    Takes a nonlinear expression and performs an epigraph reformulation.
    """

    assert expr.degree() > 1, "For linear objectives, please use the setObjective method."
    new_obj = model.addVar(lb=-float("inf"), obj=1)
    if sense == "minimize":
        model.addCons(expr <= new_obj)
        model.setMinimize()
    elif sense == "maximize":
        model.addCons(expr >= new_obj)
        model.setMaximize()
    else:
        raise Warning("unrecognized optimization sense: %s" % sense)
