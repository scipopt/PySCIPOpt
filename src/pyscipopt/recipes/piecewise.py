from typing import List

from pyscipopt import Model, quicksum, Variable, Constraint


def add_piecewise_linear_cons(model: Model, X: Variable, Y: Variable, a: List[float], b: List[float]) -> Constraint:
    """add constraint of the form y = f(x), where f is a piecewise linear function

            :param model: pyscipopt model to add the constraint to
            :param X: x variable
            :param Y: y variable
            :param a: array with x-coordinates of the points in the piecewise linear relation
            :param b: array with y-coordinate of the points in the piecewise linear relation

            Disclaimer: For the moment, can only model 2d piecewise linear functions
            Adapted from https://github.com/scipopt/PySCIPOpt/blob/master/examples/finished/piecewise.py
        """
    assert len(a) == len(b), "Must have the same number of x and y-coordinates"

    K = len(a) - 1
    w, z = {}, {}
    for k in range(K):
        w[k] = model.addVar(lb=-model.infinity())
        z[k] = model.addVar(vtype="B")

    for k in range(K):
        model.addCons(w[k] >= a[k] * z[k])
        model.addCons(w[k] <= a[k + 1] * z[k])

    model.addCons(quicksum(z[k] for k in range(K)) == 1)

    model.addCons(X == quicksum(w[k] for k in range(K)))

    c = [float(b[k + 1] - b[k]) / (a[k + 1] - a[k]) for k in range(K)]
    d = [b[k] - c[k] * a[k] for k in range(K)]

    new_cons = model.addCons(Y == quicksum(d[k] * z[k] + c[k] * w[k] for k in range(K)))

    return new_cons
