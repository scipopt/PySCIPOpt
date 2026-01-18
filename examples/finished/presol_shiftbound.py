"""
Example showing a custom presolver using PySCIPOpt's Presol plugin.

This example reproduces the logic of boundshift.c from the SCIP source
as closely as possible using PySCIPOpt.
A simple knapsack problem was chosen to let the presolver plugin
operate on.
"""

from pyscipopt import (
    Model,
    SCIP_PARAMSETTING,
    SCIP_PRESOLTIMING,
    Presol,
    SCIP_RESULT
)
from typing import List, Optional


class ShiftboundPresolver(Presol):
    """
    A presolver that converts variable domains from [a, b] to [0, b - a].

    Attributes:
    maxshift: float - Maximum absolute shift allowed.
    flipping: bool - Whether to allow flipping (multiplying by -1) for
                     differentiation.
    integer: bool - Whether to shift only integer ranges.
    """

    def __init__(
        self,
        maxshift: float = float("inf"),
        flipping: bool = True,
        integer: bool = True,
    ):
        self.maxshift = maxshift
        self.flipping = flipping
        self.integer = integer

    def presolexec(self, nrounds, presoltiming):
        # the greatest absolute value by which bounds can be shifted to avoid
        # large constant offsets
        MAXABSBOUND = 1000.0

        scip = self.model

        # check whether aggregation of variables is not allowed
        if scip.getParam("presolving/donotaggr"):
            return {"result": SCIP_RESULT.DIDNOTRUN}

        scipvars = scip.getVars()
        nbinvars = scip.getNBinVars()  # number of binary variables
        # infer number of non-binary variables
        nvars = scip.getNVars() - nbinvars

        # if number of non-binary variables equals zero
        if nvars == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # copy the non-binary variables into a separate list.
        # this slice works because SCIP orders variables by type, starting with
        # binary variables
        vars = scipvars[nbinvars:]

        # loop over the non-binary variables
        for var in reversed(vars):
            # sanity check that variable is indeed not binary
            assert var.vtype() != "BINARY"

            # do not shift non-active (fixed or (multi-)aggregated) variables
            if not var.isActive():
                continue

            # get current variable's bounds
            lb = var.getLbGlobal()
            ub = var.getUbGlobal()

            # It can happen that integer variable bounds have not been
            # propagated yet or contain small noise. This could result in an
            # aggregation that might trigger assertions when updating bounds of
            # aggregated variables (floating-point rounding errors).
            # check if variable is integer
            if var.vtype() != "CONTINUOUS":
                # assert if bounds are integral
                assert scip.isIntegral(lb)
                assert scip.isIntegral(ub)

                # round the bound values for integral variables
                lb = scip.adjustedVarLb(var, lb)
                ub = scip.adjustedVarUb(var, ub)

            # sanity check lb < ub
            assert scip.isLE(lb, ub)
            # check if variable is already fixed
            if scip.isEQ(lb, ub):
                continue
            # only operate on integer variables
            if self.integer and not scip.isIntegral(ub - lb):
                continue

            # bounds are shiftable if all following conditions hold
            cases = [
                not scip.isEQ(lb, 0.0),
                scip.isLT(ub, scip.infinity()),
                scip.isGT(lb, -scip.infinity()),
                scip.isLT(ub - lb, self.maxshift),
                scip.isLE(abs(lb), MAXABSBOUND),
                scip.isLE(abs(ub), MAXABSBOUND),
            ]
            if all(cases):
                # indicators for status of aggregation
                infeasible = False
                redundant = False
                aggregated = False

                # create new variable with same properties as the current
                # variable but with an added "_shift" suffix
                orig_name = var.name
                newvar = scip.addVar(
                    name=f"{orig_name}_shift",
                    vtype=f"{var.vtype()}",
                    lb=0.0,
                    ub=(ub - lb),
                    obj=0.0,
                )

                # aggregate old variable with new variable
                # check if self.flipping is True
                if self.flipping:
                    # check if |ub| < |lb|
                    if abs(ub) < abs(lb):
                        infeasible, redundant, aggregated = scip.aggregateVars(
                            var, newvar, 1.0, 1.0, ub
                        )
                    else:
                        infeasible, redundant, aggregated = scip.aggregateVars(
                            var, newvar, 1.0, -1.0, lb
                        )
                else:
                    infeasible, redundant, aggregated = scip.aggregateVars(
                        var, newvar, 1.0, -1.0, lb
                    )

                # problem has now become infeasible
                if infeasible:
                    result = SCIP_RESULT.CUTOFF
                else:
                    # sanity check flags
                    assert redundant
                    assert aggregated

                    result = SCIP_RESULT.SUCCESS

            else:
                result = SCIP_RESULT.DIDNOTFIND

        return {"result": result}


def knapsack(
    instance_name: str,
    sizes: List[int],
    values: List[int],
    upper_bound: List[int],
    lower_bound: List[int],
    capacity: int,
    vtypes: Optional[List[str]],
) -> tuple[Model, dict]:
    """
    Model an instance of the knapsack problem

    Parameters:
    sizes: List[int] - the sizes of the items
    values: List[int] - the values of the items
    upper_bound: List[int] - upper bounds per variable
    lower_bound: List[int] - lower bounds per variable
    capacity: int - the knapsack capacity
    vtypes: Optional[List[str]] - variable types ("B", "I", "C")

    Returns:
    tuple[Model, dict] - the SCIP model and the variables dictionary
    """

    m = Model(f"Knapsack: {instance_name}")
    x = {}
    for i in range(len(sizes)):
        assert isinstance(sizes[i], int)
        assert isinstance(values[i], int)
        assert isinstance(upper_bound[i], int)
        assert isinstance(lower_bound[i], int)

        vt = "I"
        if vtypes is not None:
            assert len(vtypes) == len(sizes)
            assert isinstance(vtypes[i], str) or (vtypes[i] is None)
            vt = vtypes[i]

        x[i] = m.addVar(
            vtype=vt,
            obj=values[i],
            lb=lower_bound[i],
            ub=upper_bound[i],
            name=f"x{i}",
        )

    assert isinstance(capacity, int)
    m.addCons(sum(sizes[i] * x[i] for i in range(len(sizes))) <= capacity)

    m.setMaximize()

    return m, x


if __name__ == "__main__":
    instance_name = "Knapsack"
    sizes = [2, 1, 3]
    values = [2, 3, 1]
    upper_bounds = [1, 4, 1]
    lower_bounds = [0, 2, 0]
    capacity = 3

    model, var_list = knapsack(
        instance_name, sizes, values, upper_bounds, lower_bounds, capacity, None
    )


    # isolate test: disable many automatic presolvers/propagators
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    for key in (
        "presolving/boundshift/maxrounds",
        "presolving/domcol/maxrounds",
        "presolving/dualsparsify/maxrounds",
        "presolving/implics/maxrounds",
        "presolving/inttobinary/maxrounds",
        "presolving/sparsify/maxrounds",
        "presolving/trivial/maxrounds",
        "propagating/dualfix/maxprerounds",
        "propagating/probing/maxprerounds",
        "propagating/symmetry/maxprerounds",
        "constraints/linear/maxprerounds",
    ):
        model.setParam(key, 0)

    # register and apply custom boundshift presolver
    presolver = ShiftboundPresolver(
        maxshift=float("inf"), flipping=True, integer=True
    )
    model.includePresol(
        presolver,
        "shiftbound",
        "converts variables with domain [a,b] to variables with domain [0,b-a]",
        priority=7900000,
        maxrounds=-1,
        timing=SCIP_PRESOLTIMING.FAST,
    )

    # run presolve on instance
    model.presolve()