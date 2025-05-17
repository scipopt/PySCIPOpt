from pyscipopt import Model, quicksum, SCIP_PARAMSETTING, exp, log, sqrt, sin
from typing import List

def random_mip_1(disable_sepa=True, disable_heur=True, disable_presolve=True, node_lim=2000, small=False):
    model = Model()

    x0 = model.addVar(lb=-2, ub=4)
    r1 = model.addVar()
    r2 = model.addVar()
    y0 = model.addVar(lb=3)
    t = model.addVar(lb=None)
    l = model.addVar(vtype="I", lb=-9, ub=18)
    u = model.addVar(vtype="I", lb=-3, ub=99)

    more_vars = []
    if small:
        n = 100
    else:
        n = 500
    for i in range(n):
        more_vars.append(model.addVar(vtype="I", lb=-12, ub=40))
        model.addCons(quicksum(v for v in more_vars) <= (40 - i) * quicksum(v for v in more_vars[::2]))

    for i in range(100):
        more_vars.append(model.addVar(vtype="I", lb=-52, ub=10))
        if small:
            model.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[65::2]))
        else:
            model.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[405::2]))

    model.addCons(r1 >= x0)
    model.addCons(r2 >= -x0)
    model.addCons(y0 == r1 + r2)
    model.addCons(t + l + 7 * u <= 300)
    model.addCons(t >= quicksum(v for v in more_vars[::3]) - 10 * more_vars[5] + 5 * more_vars[9])
    model.addCons(more_vars[3] >= l + 2)
    model.addCons(7 <= quicksum(v for v in more_vars[::4]) - x0)
    model.addCons(quicksum(v for v in more_vars[::2]) + l <= quicksum(v for v in more_vars[::4]))

    model.setObjective(t - quicksum(j * v for j, v in enumerate(more_vars[20:-40])))

    if disable_sepa:
        model.setSeparating(SCIP_PARAMSETTING.OFF)
    if disable_heur:
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
    if disable_presolve:
        model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setParam("limits/nodes", node_lim)

    return model


def random_lp_1():
    random_mip_1().relax()
    return random_mip_1()


def random_nlp_1():
    model = Model()

    v = model.addVar(name="v", ub=2)
    w = model.addVar(name="w", ub=3)
    x = model.addVar(name="x", ub=4)
    y = model.addVar(name="y", ub=1.4)
    z = model.addVar(name="z", ub=4)

    model.addCons(exp(v) + log(w) + sqrt(x) + sin(y) + z ** 3 * y <= 5)
    model.setObjective(v + w + x + y + z, sense='maximize')

    return model


def knapsack_model(weights=[4, 2, 6, 3, 7, 5], costs=[7, 2, 5, 4, 3, 4], knapsack_size = 15):
    # create solver instance
    s = Model("Knapsack")

    # setting the objective sense to maximise
    s.setMaximize()

    assert len(weights) == len(costs)

    # adding the knapsack variables
    knapsackVars = []
    varNames = []
    varBaseName = "Item"
    for i in range(len(weights)):
        varNames.append(varBaseName + "_" + str(i))
        knapsackVars.append(s.addVar(varNames[i], vtype='I', obj=costs[i], ub=1.0))

    # adding a linear constraint for the knapsack constraint
    s.addCons(quicksum(w * v for (w, v) in zip(weights, knapsackVars)) <= knapsack_size)

    return s


def bin_packing_model(sizes: List[int], capacity: int) -> Model:
    model = Model("Binpacking")
    n = len(sizes)
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(vtype="B", name=f"x{i}_{j}")
    y = [model.addVar(vtype="B", name=f"y{i}") for i in range(n)]

    for i in range(n):
        model.addCons(
            quicksum(x[i, j] for j in range(n)) == 1
        )

    for j in range(n):
        model.addCons(
            quicksum(sizes[i] * x[i, j] for i in range(n)) <= capacity * y[j]
        )

    model.setObjective(
        quicksum(y[j] for j in range(n)), "minimize"
    )

    return model


# test gastrans: see example in <model path>/examples/CallableLibrary/src/gastrans.c
# of course there is a more pythonic/elegant way of implementing this, probably
# starting by using a proper graph structure
def gastrans_model():
    GASTEMP = 281.15
    RUGOSITY = 0.05
    DENSITY = 0.616
    COMPRESSIBILITY = 0.8
    nodes = [
        #   name          supplylo   supplyup pressurelo pressureup   cost
        ("Anderlues", 0.0, 1.2, 0.0, 66.2, 0.0),  # 0
        ("Antwerpen", None, -4.034, 30.0, 80.0, 0.0),  # 1
        ("Arlon", None, -0.222, 0.0, 66.2, 0.0),  # 2
        ("Berneau", 0.0, 0.0, 0.0, 66.2, 0.0),  # 3
        ("Blaregnies", None, -15.616, 50.0, 66.2, 0.0),  # 4
        ("Brugge", None, -3.918, 30.0, 80.0, 0.0),  # 5
        ("Dudzele", 0.0, 8.4, 0.0, 77.0, 2.28),  # 6
        ("Gent", None, -5.256, 30.0, 80.0, 0.0),  # 7
        ("Liege", None, -6.385, 30.0, 66.2, 0.0),  # 8
        ("Loenhout", 0.0, 4.8, 0.0, 77.0, 2.28),  # 9
        ("Mons", None, -6.848, 0.0, 66.2, 0.0),  # 10
        ("Namur", None, -2.120, 0.0, 66.2, 0.0),  # 11
        ("Petange", None, -1.919, 25.0, 66.2, 0.0),  # 12
        ("Peronnes", 0.0, 0.96, 0.0, 66.2, 1.68),  # 13
        ("Sinsin", 0.0, 0.0, 0.0, 63.0, 0.0),  # 14
        ("Voeren", 20.344, 22.012, 50.0, 66.2, 1.68),  # 15
        ("Wanze", 0.0, 0.0, 0.0, 66.2, 0.0),  # 16
        ("Warnand", 0.0, 0.0, 0.0, 66.2, 0.0),  # 17
        ("Zeebrugge", 8.87, 11.594, 0.0, 77.0, 2.28),  # 18
        ("Zomergem", 0.0, 0.0, 0.0, 80.0, 0.0)  # 19
    ]
    arcs = [
        # node1  node2  diameter length active */
        (18, 6, 890.0, 4.0, False),
        (18, 6, 890.0, 4.0, False),
        (6, 5, 890.0, 6.0, False),
        (6, 5, 890.0, 6.0, False),
        (5, 19, 890.0, 26.0, False),
        (9, 1, 590.1, 43.0, False),
        (1, 7, 590.1, 29.0, False),
        (7, 19, 590.1, 19.0, False),
        (19, 13, 890.0, 55.0, False),
        (15, 3, 890.0, 5.0, True),
        (15, 3, 395.0, 5.0, True),
        (3, 8, 890.0, 20.0, False),
        (3, 8, 395.0, 20.0, False),
        (8, 17, 890.0, 25.0, False),
        (8, 17, 395.0, 25.0, False),
        (17, 11, 890.0, 42.0, False),
        (11, 0, 890.0, 40.0, False),
        (0, 13, 890.0, 5.0, False),
        (13, 10, 890.0, 10.0, False),
        (10, 4, 890.0, 25.0, False),
        (17, 16, 395.5, 10.5, False),
        (16, 14, 315.5, 26.0, True),
        (14, 2, 315.5, 98.0, False),
        (2, 12, 315.5, 6.0, False)
    ]

    model = Model()

    # create flow variables
    flow = {}
    for arc in arcs:
        flow[arc] = model.addVar("flow_%s_%s" % (nodes[arc[0]][0], nodes[arc[1]][0]),  # names of nodes in arc
                                 lb=0.0 if arc[4] else None)  # no lower bound if not active

    # pressure difference variables
    pressurediff = {}
    for arc in arcs:
        pressurediff[arc] = model.addVar("pressurediff_%s_%s" % (nodes[arc[0]][0], nodes[arc[1]][0]),
                                         # names of nodes in arc
                                         lb=None)

    # supply variables
    supply = {}
    for node in nodes:
        supply[node] = model.addVar("supply_%s" % (node[0]), lb=node[1], ub=node[2], obj=node[5])

    # square pressure variables
    pressure = {}
    for node in nodes:
        pressure[node] = model.addVar("pressure_%s" % (node[0]), lb=node[3] ** 2, ub=node[4] ** 2)

    # node balance constrains, for each node i: outflows - inflows = supply
    for nid, node in enumerate(nodes):
        # find arcs that go or end at this node
        flowbalance = 0
        for arc in arcs:
            if arc[0] == nid:  # arc is outgoing
                flowbalance += flow[arc]
            elif arc[1] == nid:  # arc is incoming
                flowbalance -= flow[arc]
            else:
                continue

        model.addCons(flowbalance == supply[node], name="flowbalance%s" % node[0])

    # pressure difference constraints: pressurediff[node1 to node2] = pressure[node1] - pressure[node2]
    for arc in arcs:
        model.addCons(pressurediff[arc] == pressure[nodes[arc[0]]] - pressure[nodes[arc[1]]],
                      "pressurediffcons_%s_%s" % (nodes[arc[0]][0], nodes[arc[1]][0]))

    # pressure loss constraints:
    # active arc: flow[arc]^2 + coef * pressurediff[arc] <= 0.0
    # regular pipes: flow[arc] * abs(flow[arc]) - coef * pressurediff[arc] == 0.0
    # coef = 96.074830e-15*diameter(i)^5/(lambda*compressibility*temperatur*length(i)*density)
    # lambda = (2*log10(3.7*diameter(i)/rugosity))^(-2)
    from math import log10
    for arc in arcs:
        coef = 96.074830e-15 * arc[2] ** 5 * (2.0 * log10(3.7 * arc[2] / RUGOSITY)) ** 2 / COMPRESSIBILITY / GASTEMP / \
               arc[3] / DENSITY
        if arc[4]:  # active
            model.addCons(flow[arc] ** 2 + coef * pressurediff[arc] <= 0.0,
                          "pressureloss_%s_%s" % (nodes[arc[0]][0], nodes[arc[1]][0]))
        else:
            model.addCons(flow[arc] * abs(flow[arc]) - coef * pressurediff[arc] == 0.0,
                          "pressureloss_%s_%s" % (nodes[arc[0]][0], nodes[arc[1]][0]))

    return model


def knapsack_lp(weights, costs):
    return knapsack_model(weights, costs).relax()


def bin_packing_lp(sizes, capacity):
    return bin_packing_model(sizes, capacity).relax()


def gastrans_lp():
    gastrans_model().relax()

    return gastrans_model()