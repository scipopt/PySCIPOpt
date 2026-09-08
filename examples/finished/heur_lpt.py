"""
Example showing a custom primal heuristic using PySCIPOpt's Heur plugin.

The heuristic warm-starts a scheduling MIP: before SCIP processes the root
node, it builds a feasible schedule with a fast list-scheduling rule and
hands it to SCIP as an incumbent solution.

The problem is parallel machine scheduling with release dates and makespan
objective, Pm|r_j|C_max: n jobs with processing times p_j and release dates
r_j have to be assigned to m identical machines and sequenced so that the
last job finishes as early as possible. The MIP is a disjunctive (big-M)
formulation. The heuristic is the LPT (longest processing time first) list
scheduling rule: whenever a machine becomes free, it starts the longest job
that has already been released.
"""

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_HEURTIMING, quicksum


def lpt_schedule(p, r, m):
    """
    LPT list scheduling for Pm|r_j|C_max.

    Whenever a machine becomes free, start the longest released job on it.
    If no job is released yet, wait for the next release.

    Returns a dict job -> (machine, start time) and the makespan.
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
    """
    Primal heuristic that offers the LPT schedule to SCIP as a solution.

    It needs the problem data and the model's variables to translate the
    schedule into variable values.
    """

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
        # The schedule does not depend on the search state, so one run is
        # enough. SCIP processes the root node again after a restart, which
        # would call the heuristic a second time otherwise.
        if self.done:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.done = True

        schedule, cmax = lpt_schedule(self.p, self.r, self.m)

        # The solution is built in the original space: after presolving, SCIP
        # may have fixed or aggregated variables (here symmetry handling fixes
        # the machine of some jobs), and setting a conflicting value on such a
        # variable in a transformed solution is an error. Passing the heuristic
        # to createOrigSol tells SCIP who found the solution, so it shows up
        # under this heuristic's display character in the log.
        sol = self.model.createOrigSol(self)

        sol[self.makespan] = cmax
        for j, (machine, st) in schedule.items():
            sol[self.start[j]] = st
            for k in range(self.m):
                sol[self.assign[j, k]] = 1 if k == machine else 0

        # Jobs on the same machine never overlap, so the one starting first
        # also finishes before the other starts.
        for i, (machine_i, st_i) in schedule.items():
            for j, (machine_j, st_j) in schedule.items():
                if i != j:
                    same_machine = machine_i == machine_j
                    sol[self.before[i, j]] = 1 if same_machine and st_i < st_j else 0

        # trySol checks feasibility and stores the solution if it is accepted.
        accepted = self.model.trySol(sol)
        print(f"LPT heuristic: makespan {cmax} {'accepted' if accepted else 'rejected'}")

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        return {"result": SCIP_RESULT.DIDNOTFIND}


def build_model(p, r, m):
    """
    Disjunctive big-M model for Pm|r_j|C_max.

    Variables:
        start[j]     - start time of job j
        assign[j, k] - 1 if job j runs on machine k
        before[i, j] - 1 if job i finishes before job j starts
        makespan     - completion time of the last job

    Returns the model and its variables.
    """
    n = len(p)
    jobs = range(n)
    machines = range(m)

    # Running all jobs on one machine after the last release is always feasible,
    # so no job needs to start after this horizon.
    horizon = max(r) + sum(p)

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
                # if i is sequenced before j, j cannot start until i is done
                model.addCons(start[i] + p[i] <= start[j] + horizon * (1 - before[i, j]))

    for i in jobs:
        for j in jobs:
            if i < j:
                for k in machines:
                    # two jobs on the same machine have to be sequenced
                    model.addCons(assign[i, k] + assign[j, k] - 1 <= before[i, j] + before[j, i])

    model.setObjective(makespan, "minimize")

    return model, start, assign, before, makespan


def print_schedule(p, schedule, m):
    for k in range(m):
        on_k = sorted((st, j) for j, (machine, st) in schedule.items() if machine == k)
        jobs = "  ".join(f"job {j} [{st}, {st + p[j]})" for st, j in on_k)
        print(f"  machine {k}: {jobs}")


if __name__ == "__main__":
    # On this instance the greedy LPT schedule is not optimal, so the log shows
    # SCIP starting from the heuristic's incumbent and improving on it.
    p = [2, 2, 2, 9, 8, 7, 4, 5, 3, 6]
    r = [0, 0, 0, 1, 1, 1, 3, 5, 2, 4]
    m = 3

    schedule, cmax = lpt_schedule(p, r, m)
    print(f"LPT schedule, makespan {cmax}:")
    print_schedule(p, schedule, m)

    model, start, assign, before, makespan = build_model(p, r, m)

    heur = LPTHeur(p, r, m, start, assign, before, makespan)
    # freq=0: only call the heuristic at depth 0, i.e. at the root node.
    model.includeHeur(heur, "lpt", "LPT list scheduling warm start", "L",
                      freq=0, timingmask=SCIP_HEURTIMING.BEFORENODE)

    model.optimize()

    print(f"\nSCIP status: {model.getStatus()}")
    print(f"optimal schedule, makespan {model.getObjVal():g}:")
    best = {}
    for j in range(len(p)):
        machine = next(k for k in range(m) if model.getVal(assign[j, k]) > 0.5)
        best[j] = (machine, round(model.getVal(start[j])))
    print_schedule(p, best, m)
