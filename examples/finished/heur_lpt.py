"""
Parallel machine scheduling with release dates (Pm|r_j|C_max), solved with a
big-M formulation that is warm-started by a custom heuristic.

The heuristic is a Heur plugin that runs once before the root node. It builds
an LPT (longest processing time first) list schedule and hands it to SCIP as
an incumbent.
"""

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum


def lpt_schedule(p, r, m):
    """
    LPT list scheduling: whenever a machine becomes free, start the longest
    released job on it. Returns {job: (machine, start)} and the makespan.
    """
    unscheduled = set(range(len(p)))
    free_at = [0] * m
    schedule = {}

    while unscheduled:
        machine = min(range(m), key=lambda k: free_at[k])
        now = free_at[machine]

        released = [j for j in unscheduled if r[j] <= now]
        if not released:
            now = min(r[j] for j in unscheduled)
            released = [j for j in unscheduled if r[j] <= now]

        job = max(released, key=lambda j: p[j])
        schedule[job] = (machine, now)
        free_at[machine] = now + p[job]
        unscheduled.remove(job)

    return schedule, max(free_at)


class LPTHeur(Heur):
    """Offers the LPT schedule to SCIP as a solution."""

    def __init__(self, p, r, m, start, assign, before, makespan):
        super().__init__()
        self.p = p
        self.r = r
        self.m = m
        self.start = start
        self.assign = assign
        self.before = before
        self.makespan = makespan
        self.done = False

    def heurexec(self, heurtiming, nodeinfeasible):
        # run once; SCIP processes the root again after a restart
        if self.done:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.done = True

        schedule, cmax = lpt_schedule(self.p, self.r, self.m)

        # Build the solution in the original space. Presolving may have fixed
        # some variables (symmetry handling does so here), and setting a
        # conflicting value on a fixed variable of a transformed solution fails.
        sol = self.model.createOrigSol(self)

        sol[self.makespan] = cmax
        for j, (machine, st) in schedule.items():
            sol[self.start[j]] = st
            for k in range(self.m):
                sol[self.assign[j, k]] = 1 if k == machine else 0

        for i, (machine_i, st_i) in schedule.items():
            for j, (machine_j, st_j) in schedule.items():
                if i != j:
                    same_machine = machine_i == machine_j
                    sol[self.before[i, j]] = 1 if same_machine and st_i < st_j else 0

        accepted = self.model.trySol(sol)
        print(f"LPT heuristic: makespan {cmax} {'accepted' if accepted else 'rejected'}")

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        return {"result": SCIP_RESULT.DIDNOTFIND}


def build_model(p, r, m):
    """
    Disjunctive big-M model.

    start[j]     - start time of job j
    assign[j, k] - 1 if job j runs on machine k
    before[i, j] - 1 if job i finishes before job j starts
    makespan     - completion time of the last job
    """
    jobs = range(len(p))
    machines = range(m)
    horizon = max(r) + sum(p)  # no job needs to start later than this

    model = Model("Pm|r_j|C_max")

    start = {j: model.addVar(vtype="C", lb=r[j], ub=horizon - p[j], name=f"start_{j}") for j in jobs}
    assign = {(j, k): model.addVar(vtype="B", name=f"assign_{j}_{k}") for j in jobs for k in machines}
    before = {(i, j): model.addVar(vtype="B", name=f"before_{i}_{j}") for i in jobs for j in jobs if i != j}
    makespan = model.addVar(vtype="C", ub=horizon, name="makespan")

    for j in jobs:
        model.addCons(quicksum(assign[j, k] for k in machines) == 1)
        model.addCons(start[j] + p[j] <= makespan)

    for i in jobs:
        for j in jobs:
            if i != j:
                model.addCons(start[i] + p[i] <= start[j] + horizon * (1 - before[i, j]))

    # jobs on the same machine have to be sequenced
    for i in jobs:
        for j in jobs:
            if i < j:
                for k in machines:
                    model.addCons(assign[i, k] + assign[j, k] - 1 <= before[i, j] + before[j, i])

    model.setObjective(makespan, "minimize")

    return model, start, assign, before, makespan


def print_schedule(p, schedule, m):
    for k in range(m):
        on_k = sorted((st, j) for j, (machine, st) in schedule.items() if machine == k)
        jobs = "  ".join(f"job {j} [{st}, {st + p[j]})" for st, j in on_k)
        print(f"  machine {k}: {jobs}")


if __name__ == "__main__":
    p = [2, 2, 2, 9, 8, 7, 4, 5, 3, 6]
    r = [0, 0, 0, 1, 1, 1, 3, 5, 2, 4]
    m = 3

    schedule, cmax = lpt_schedule(p, r, m)
    print(f"LPT schedule, makespan {cmax}:")
    print_schedule(p, schedule, m)

    model, start, assign, before, makespan = build_model(p, r, m)

    heur = LPTHeur(p, r, m, start, assign, before, makespan)
    model.includeHeur(heur, "lpt", "LPT list scheduling warm start", "L",
                      freq=0, timingmask=SCIP_HEURTIMING.BEFORENODE)  # freq=0: root node only

    model.optimize()

    print(f"\nSCIP status: {model.getStatus()}")
    print(f"optimal schedule, makespan {model.getObjVal():g}:")
    best = {}
    for j in range(len(p)):
        machine = next(k for k in range(m) if model.getVal(assign[j, k]) > 0.5)
        best[j] = (machine, round(model.getVal(start[j])))
    print_schedule(p, best, m)
