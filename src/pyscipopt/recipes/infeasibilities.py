from pyscipopt import Model, quicksum


def get_infeasible_constraints(orig_model: Model, verbose=False):
    """
    Given a model, adds slack variables to all the constraints and minimizes a binary variable that indicates if they're positive.
    Positive slack variables correspond to infeasible constraints.
    """
    
    model = Model(sourceModel=orig_model, origcopy=True) # to preserve the model

    slack      = {}
    aux        = {}
    binary     = {}
    aux_binary = {}

    for c in model.getConss():

        slack[c.name] = model.addVar(lb=-float("inf"), name="s_"+c.name) 
        model.addConsCoeff(c, slack[c.name], 1)
        binary[c.name] = model.addVar(vtype="B") # Binary variable to get minimum infeasible constraints. See PR #857.

        # getting the absolute value because of <= and >= constraints 
        aux[c.name] = model.addVar()
        model.addCons(aux[c.name] >= slack[c.name])
        model.addCons(aux[c.name] >= -slack[c.name])
        
        # modeling aux > 0 => binary = 1 constraint. See https://or.stackexchange.com/q/12142/5352 for an explanation
        aux_binary[c.name] = model.addVar(vtype="B")
        model.addCons(binary[c.name]+aux_binary[c.name] == 1)
        model.addConsSOS1([aux[c.name], aux_binary[c.name]])

    model.setObjective(quicksum(binary[c.name] for c in orig_model.getConss()))
    model.hideOutput()
    model.optimize()

    n_infeasibilities_detected = 0
    for c in binary:
        if model.isGT(model.getVal(binary[c]), 0):
            n_infeasibilities_detected += 1
            print("Constraint %s is causing an infeasibility." % c)

    if verbose:
        if n_infeasibilities_detected > 0:
            print("If the constraint names are unhelpful, consider giving them\
                a suitable name when creating the model with model.addCons(..., name=\"the_name_you_want\")")
        else:
            print("Model is feasible.")
    
    return n_infeasibilities_detected, aux