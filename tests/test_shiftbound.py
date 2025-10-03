"""
Tests for the Shiftbound presolver.
Parametrised knapsack instances + presolver option combitions.
"""

import logging
import pytest
from pyscipopt import (
    SCIP_PARAMSETTING,
    SCIP_PRESOLTIMING
)
from typing import List, Tuple, Optional
from PySCIPOpt.examples.finished import ShiftboundPresolver, knapsack 

# define a few small, fast instances to exercise different shapes and types
INSTANCES: List[
    Tuple[
        List[int],  # item sizes (i.e., constraint vector)
        List[int],  # item values (i.e., coefficient vector)
        List[int],  # variable upper bound
        List[int],  # variable lower bound
        int,  # capacity constraint value
        Optional[List[str]],  # variable types
    ]
] = [
    # small integer knapsack
    ([2, 1, 3], [2, 3, 1], [1, 4, 1], [0, 2, 0], 3, None),
    # small integer knapsack (flipped)
    ([2, 1, 3], [2, 3, 1], [1, -2, 1], [0, -4, 0], 3, None),
    # vtype continuous
    ([2, 1, 3], [2, 3, 1], [1, 4, 1], [0, 2, 0], 3, ["C", "C", "C"]),
    # vtype integer
    ([2, 1, 3], [2, 3, 1], [1, 4, 1], [0, 2, 0], 3, ["I", "I", "I"]),
    # above MAXABSBOUND
    ([2, 1, 3], [2, 3, 1], [1, 4, 1001], [0, 2, 1000], 3, None),
    # no variables to shift
    ([2, 1, 3], [2, 3, 1], [1, 2, 1], [0, 0, 0], 3, None),
]

INSTANCE_IDS = [
    "small-integer",
    "small-integer-flipped",
    "vtype continuous",
    "vtype integer",
    "above MAXABSBOUND",
    "no variables to shift",
]

# presolver option combinations; parametrise doNotAggr, maxshift, and
# flipping and integer flags
PRESOLVER_OPTIONS = [
    (True, None, True, True),
    (False, 0, True, True),
    (False, None, True, True),
    (False, None, True, False),
    (False, None, False, True),
    (False, None, False, False),
]

# pre-define the amount of variables that should be aggregated for each
# presolver option
EXPECTED_VALUE = [
    # small integer knapsack
    (0, 0, 1, 1, 1, 1),
    # small integer knapsack (flipped)
    (0, 0, 1, 1, 1, 1),
    # vtype continuous
    (0, 0, 1, 1, 1, 1),
    # vtype integer
    (0, 0, 1, 1, 1, 1),
    # above MAXABSBOUND
    (0, 0, 1, 1, 1, 1),
    # no variables to shift
    (0, 0, 0, 0, 0, 0),
]

# build explicit (instance, options, expected_value) test cases
TEST_CASES = []
TEST_IDS = []
for inst_idx, inst in enumerate(INSTANCES):
    for opt_idx, opt in enumerate(PRESOLVER_OPTIONS):
        expected_for_inst = EXPECTED_VALUE[inst_idx][opt_idx]
        TEST_CASES.append((inst, opt, expected_for_inst))
        TEST_IDS.append(f"{INSTANCE_IDS[inst_idx]}")

@pytest.fixture
def instance(request):
    # get human-readable instance id that comes from `ids=INSTANCE_IDS`
    instance_name = None
    if (
        hasattr(request, "node")
        and hasattr(request.node, "callspec")
        and request.node.callspec is not None
    ):
        instance_name = request.node.callspec.id[:-1]
    else:
        # fallback when callspec/id is not available
        instance_name = "instance-" + str(hash(request.param))[:8]

    sizes, values, ubs, lbs, capacity, vtypes = request.param
    model, vars_list = knapsack(
        instance_name, sizes, values, ubs, lbs, capacity, vtypes
    )

    # return a tuple so tests can inspect variables easily
    try:
        yield (
            model,
            vars_list,
            {
                "sizes": sizes,
                "values": values,
                "ubs": ubs,
                "lbs": lbs,
                "capacity": capacity,
                "vtypes": vtypes,
                "instance_name": instance_name,
            },
        )
    finally:
        # cleanup
        try:
            model.freeProb()
        except Exception:
            pass


@pytest.mark.parametrize(
    "instance, options, expected_value",
    TEST_CASES,
    ids=TEST_IDS,
    indirect=["instance"],
)
def test_shiftbound(instance, options, expected_value):
    model, vars_list, meta = instance
    doNotAggr, maxshift, flipping, integer = options

    # silence solver output
    model.hideOutput()

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
        "presolving/milp/maxrounds",
        "presolving/sparsify/maxrounds",
        "presolving/trivial/maxrounds",
        "propagating/dualfix/maxprerounds",
        "propagating/probing/maxprerounds",
        "propagating/symmetry/maxprerounds",
        "constraints/linear/maxprerounds",
    ):
        try:
            model.setParam(key, 0)
        except Exception:
            # parameter might not exist on older/newer SCIP builds; ignore
            pass

    if isinstance(doNotAggr, bool):
        try:
            model.setParam("presolving/donotaggr", doNotAggr)
        except Exception:
            # parameter might not exist on older/newer SCIP builds; ignore
            pass

    # Register and apply custom boundshift presolver
    if not (isinstance(maxshift, float) or isinstance(maxshift, int)):
        maxshift = float("inf")
    presolver = ShiftboundPresolver(
        maxshift=maxshift, flipping=flipping, integer=integer
    )
    model.includePresol(
        presolver,
        "shiftbound",
        "converts variables with domain [a,b] to variables with domain [0,b-a]",
        priority=7900000,
        maxrounds=1,
        timing=SCIP_PRESOLTIMING.FAST,
    )
    # set presolver calls to one (maxrounds=1) to keep tests deterministic

    # run presolve on instance
    model.presolve()
    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        model.printStatistics()
        model.printProblem()
        model.printProblem(trans=True)

    # count shifted variables created by the presolver (names ending with
    # "_shift")
    shifted_names = []
    for v in model.getVars(transformed=True):
        name = None
        if hasattr(v, "name"):
            name = v.name
        else:
            try:
                name = v.getName()
            except Exception:
                name = None
        if name.endswith("_shift"):
            shifted_names.append(name)

    shifted_count = len(shifted_names)

    assert shifted_count == expected_value, (
        f"expected {expected_value} shifted variables for test "
        f'"{meta.get("instance_name")}" '
        f"with options {options}, got {shifted_count}"
    )