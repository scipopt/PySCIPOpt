from pyscipopt import Model

def plot_primal_dual_evolution(model: Model):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise("matplotlib is required to plot the solution. Try running `pip install matplotlib` in the command line.")

    time_primal, val_primal = zip(*model.data["primal_log"])
    plt.plot(time_primal, val_primal, label="Primal bound")
    time_dual, val_dual = zip(*model.data["dual_log"])
    plt.plot(time_dual, val_dual, label="Dual bound")

    plt.legend(loc="best")
    plt.show()


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


if __name__=="__main__":
    from pyscipopt.recipes.primal_dual_evolution import attach_primal_dual_evolution_eventhdlr

    model = gastrans_model()
    model.attach_primal_dual_evolution_eventhdlr()

    model.optimize()
    plot_primal_dual_evolution(model)
