"""
Parallel machine scheduling with release dates (Pm|r_j|C_max), modelled three
ways with SCIP's logical constraints instead of big-M constraints:

    and_or_indicator_model  and, or and indicator constraints
    disjunction_model       disjunction constraints
    time_indexed_model      cardinality constraints

The formulations differ in what they contribute to the LP relaxation, which
shows in the number of nodes SCIP needs.
"""

from pyscipopt import Model, quicksum


def base_model(p, r, m, name):
    """Everything except the condition that jobs on one machine do not overlap."""
    jobs = range(len(p))
    machines = range(m)
    horizon = max(r) + sum(p)  # no job needs to start later than this

    model = Model(name)

    start = {j: model.addVar(vtype="C", lb=r[j], ub=horizon - p[j], name=f"start_{j}") for j in jobs}
    assign = {(j, k): model.addVar(vtype="B", name=f"assign_{j}_{k}") for j in jobs for k in machines}
    makespan = model.addVar(vtype="C", ub=horizon, name="makespan")

    for j in jobs:
        # not an xor constraint: xor fixes the parity, so with three machines
        # a job could be assigned to all of them
        model.addCons(quicksum(assign[j, k] for k in machines) == 1)
        model.addCons(start[j] + p[j] <= makespan)

    model.setObjective(makespan, "minimize")

    return model, start, assign, makespan


def and_or_indicator_model(p, r, m):
    """
    before[i, j] = 1 forces j to start after i is done (indicator). For each
    pair and machine: same = assign[i, k] and assign[j, k], ordered =
    before[i, j] or before[j, i], and same implies ordered (indicator).
    """
    model, start, assign, makespan = base_model(p, r, m, "and-or-indicator")
    jobs = range(len(p))
    machines = range(m)

    before = {(i, j): model.addVar(vtype="B", name=f"before_{i}_{j}") for i in jobs for j in jobs if i != j}

    for i in jobs:
        for j in jobs:
            if i != j:
                model.addConsIndicator(start[i] + p[i] <= start[j], binvar=before[i, j], name=f"seq_{i}_{j}")

    for i in jobs:
        for j in jobs:
            if i < j:
                ordered = model.addVar(vtype="B", name=f"ordered_{i}_{j}")
                model.addConsOr([before[i, j], before[j, i]], ordered)
                for k in machines:
                    same = model.addVar(vtype="B", name=f"same_{i}_{j}_{k}")
                    model.addConsAnd([assign[i, k], assign[j, k]], same)
                    model.addConsIndicator(ordered >= 1, binvar=same, name=f"ordered_if_same_{i}_{j}_{k}")

    return model, start, assign, makespan


def disjunction_model(p, r, m):
    """
    For each pair and machine: i is not on the machine, or j is not, or i
    finishes before j starts, or j finishes before i starts.

    A disjunction is enforced by branching only and adds nothing to the LP
    relaxation, so without the load constraint at the end SCIP enumerates
    schedules. Try removing it.
    """
    model, start, assign, makespan = base_model(p, r, m, "disjunction")
    jobs = range(len(p))
    machines = range(m)

    for i in jobs:
        for j in jobs:
            if i < j:
                for k in machines:
                    model.addConsDisjunction(
                        [assign[i, k] <= 0, assign[j, k] <= 0,
                         start[i] + p[i] <= start[j], start[j] + p[j] <= start[i]],
                        name=f"no_overlap_{i}_{j}_{k}",
                    )

    for k in machines:
        # total processing time on a machine is a lower bound on the makespan
        model.addCons(quicksum(p[j] * assign[j, k] for j in jobs) <= makespan, name=f"load_{k}")

    return model, start, assign, makespan


def time_indexed_model(p, r, m):
    """
    x[j, k, t] = 1 if job j starts on machine k at time t. A cardinality
    constraint per machine and time step allows at most one running job.
    """
    model, start, assign, makespan = base_model(p, r, m, "time-indexed")
    jobs = range(len(p))
    machines = range(m)
    horizon = max(r) + sum(p)
    slots = {j: range(r[j], horizon - p[j] + 1) for j in jobs}  # possible start times

    x = {}
    for j in jobs:
        for k in machines:
            for t in slots[j]:
                x[j, k, t] = model.addVar(vtype="B", name=f"x_{j}_{k}_{t}")

    for j in jobs:
        model.addCons(quicksum(x[j, k, t] for k in machines for t in slots[j]) == 1)
        model.addCons(start[j] == quicksum(t * x[j, k, t] for k in machines for t in slots[j]))
        for k in machines:
            model.addCons(assign[j, k] == quicksum(x[j, k, t] for t in slots[j]))

    for k in machines:
        for t in range(horizon):
            # jobs that would be running on machine k at time t
            running = [x[j, k, s] for j in jobs for s in range(t - p[j] + 1, t + 1) if (j, k, s) in x]
            if len(running) > 1:
                model.addConsCardinality(running, 1, name=f"one_job_{k}_{t}")

    return model, start, assign, makespan


def print_schedule(p, m, start, assign, model):
    for k in range(m):
        on_k = sorted((round(model.getVal(start[j])), j) for j in range(len(p)) if model.getVal(assign[j, k]) > 0.5)
        jobs = "  ".join(f"job {j} [{st}, {st + p[j]})" for st, j in on_k)
        print(f"  machine {k}: {jobs}")


if __name__ == "__main__":
    p = [2, 2, 2, 9, 8, 7, 4, 5, 3, 6]
    r = [0, 0, 0, 1, 1, 1, 3, 5, 2, 4]
    m = 3

    for build in (and_or_indicator_model, disjunction_model, time_indexed_model):
        model, start, assign, makespan = build(p, r, m)
        model.hideOutput()
        model.optimize()
        print(f"{model.getProbName()}: {model.getStatus()}, makespan {model.getObjVal():g}, "
              f"{model.getNNodes()} nodes, {model.getSolvingTime():.2f}s")
        print_schedule(p, m, start, assign, model)
