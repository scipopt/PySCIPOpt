"""
Example of modelling with SCIP's logical constraint handlers.

The problem is parallel machine scheduling with release dates and makespan
objective, Pm|r_j|C_max: n jobs with processing times p_j and release dates
r_j have to be assigned to m identical machines and sequenced so that the
last job finishes as early as possible. Jobs on the same machine must not
overlap, and that condition is where the formulations differ. Each one
states it with a different constraint type instead of big-M constraints:

    and_or_indicator_model  and, or and indicator constraints
    disjunction_model       disjunction constraints
    time_indexed_model      cardinality constraints

All three are solved on the same instance and reach the same makespan.
They differ a lot in how much branching SCIP needs, because the constraint
types differ in what they contribute to the LP relaxation. The disjunction
formulation shows the extreme case: a disjunction is enforced by branching
alone, so the model needs an extra linear constraint to solve in reasonable
time.
"""

from pyscipopt import Model, quicksum


def base_model(p, r, m, name):
    """
    Variables and constraints shared by all formulations.

    Every job is assigned to exactly one machine, starts after its release
    date and finishes before the makespan. What is missing is the condition
    that jobs on the same machine do not overlap.
    """
    jobs = range(len(p))
    machines = range(m)

    # Running all jobs on one machine after the last release is always feasible,
    # so no job needs to start after this horizon.
    horizon = max(r) + sum(p)

    model = Model(name)

    start = {j: model.addVar(vtype="C", lb=r[j], ub=horizon - p[j], name=f"start_{j}") for j in jobs}
    assign = {(j, k): model.addVar(vtype="B", name=f"assign_{j}_{k}") for j in jobs for k in machines}
    makespan = model.addVar(vtype="C", ub=horizon, name="makespan")

    for j in jobs:
        # Exactly one machine per job. An xor constraint would be wrong here:
        # it fixes the parity of the number of true variables, so with three
        # machines it would also allow assigning a job to all three.
        model.addCons(quicksum(assign[j, k] for k in machines) == 1)
        model.addCons(start[j] + p[j] <= makespan)

    model.setObjective(makespan, "minimize")

    return model, start, assign, makespan


def and_or_indicator_model(p, r, m):
    """
    before[i, j] = 1 means job i finishes before job j starts, enforced by an
    indicator constraint. For every pair of jobs and every machine, an and
    constraint detects that both jobs run on that machine, an or constraint
    detects that the pair is sequenced one way or the other, and an indicator
    constraint requires the second whenever the first holds.
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
                # ordered = before[i, j] or before[j, i]
                ordered = model.addVar(vtype="B", name=f"ordered_{i}_{j}")
                model.addConsOr([before[i, j], before[j, i]], ordered)
                for k in machines:
                    # same = assign[i, k] and assign[j, k]
                    same = model.addVar(vtype="B", name=f"same_{i}_{j}_{k}")
                    model.addConsAnd([assign[i, k], assign[j, k]], same)
                    # same machine => the pair is ordered
                    model.addConsIndicator(ordered >= 1, binvar=same, name=f"ordered_if_same_{i}_{j}_{k}")

    return model, start, assign, makespan


def disjunction_model(p, r, m):
    """
    For every pair of jobs and every machine, a disjunction constraint states
    that one of the following holds: job i is not on the machine, job j is not
    on the machine, i finishes before j starts, or j finishes before i starts.

    No auxiliary binary variables are needed for the sequencing, but SCIP
    enforces a disjunction by branching on its members only, so the LP
    relaxation knows nothing about the non-overlap condition. On its own the
    model makes SCIP enumerate schedules. The load constraint at the end is a
    weaker linear consequence of non-overlap that gives the LP a useful bound;
    remove it to see the difference.
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
        # jobs on one machine run one after the other, so their total
        # processing time is a lower bound on the makespan
        model.addCons(quicksum(p[j] * assign[j, k] for j in jobs) <= makespan, name=f"load_{k}")

    return model, start, assign, makespan


def time_indexed_model(p, r, m):
    """
    x[j, k, t] = 1 means job j starts on machine k at time t. Each job starts
    exactly once, and a cardinality constraint per machine and time step
    allows at most one of the jobs that would be running then to be started.
    Start times and machine assignments are recovered linearly from x.
    """
    model, start, assign, makespan = base_model(p, r, m, "time-indexed")
    jobs = range(len(p))
    machines = range(m)
    horizon = max(r) + sum(p)

    # possible start times of each job
    slots = {j: range(r[j], horizon - p[j] + 1) for j in jobs}

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
            # jobs that occupy machine k at time t if started at s
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
