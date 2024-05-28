from pyscipopt import Model, SCIP_PARAMEMPHASIS


def get_infeasible_constraints(orig_model: Model, verbose=False):
    """
    Given a model, adds slack variables to all the constraints and minimizes their sum.
    Non-zero slack variables correspond to infeasible constraints.
    """

    model = Model(sourceModel=orig_model, origcopy=True) # to preserve the model
    slack = {}
    aux   = {}
    for c in model.getConss():

        slack[c.name] = model.addVar(lb=-float("inf"), name=c.name) 

        model.addConsCoeff(c, slack[c.name], 1)

        # getting the absolute value because of <= and >= constraints 
        aux[c.name] = model.addVar(obj=1)
        model.addCons(aux[c.name] >= slack[c.name])
        model.addCons(aux[c.name] >= -slack[c.name])


    model.hideOutput()
    model.setPresolve(0) # just to be safe, maybe we can use presolving
    model.setEmphasis(SCIP_PARAMEMPHASIS.PHASEFEAS) # focusing on model feasibility
    #model.setParam("limits/solutions", 1) # SCIP sometimes returns the incorrect stage when models are prematurely stopped
    model.optimize()

    n_infeasibilities_detected = 0
    for v in aux:
        if model.isGT(model.getVal(aux[v]), 0):
            n_infeasibilities_detected += 1
            print("Constraint %s is causing an infeasibility." % v)

    if verbose:
        if n_infeasibilities_detected > 0:
            print("If the constraint names are unhelpful, consider giving them\
                a suitable name when creating the model with model.addCons(..., name=\"the_name_you_want\")")
        else:
            print("Model is feasible.")
    
    return n_infeasibilities_detected, aux