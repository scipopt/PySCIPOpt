from pyscipopt import Model, Conshdlr, quicksum, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING, SCIP_PARAMSETTING
from itertools import combinations

# -----------------------------
# Model data
# -----------------------------
def running_example(example):
    if example == 1:
        demand = [400, 400, 800, 800, 1200, 1200, 1200, 1200]
        N = len(demand)
        T = list(range(1, N + 1))
        setup_cost = [5000] * N
        holding_cost = [5] * N
        production_cost = [100] * N
        M = [sum(demand[i:]) for i in range(len(demand))]
        ini_inv = 200
        return demand, N, T, setup_cost, holding_cost, production_cost, M, ini_inv
        
    elif example == 2:
        demand = [5, 7, 3, 6, 4]
        N = len(demand)
        T = list(range(1, N + 1))
        setup_cost = [3] * N
        holding_cost = [1] * N
        production_cost = [1, 1, 3, 3, 3]
        M = [sum(demand[i:]) for i in range(len(demand))]
        ini_inv = 2
        return demand, N, T, setup_cost, holding_cost, production_cost, M, ini_inv

    else:
        raise ValueError("The correct example should be selected")

# -----------------------------
# Define conshdlr
# -----------------------------
class LS_Cuts(Conshdlr):
    def __init__(self, T, demand, ini_inv):
        self.T = T
        self.demand = list(demand)
        self.ini_inv = ini_inv
        self.N = len(self.T)

    def generate_subsets(self):
        result = {}
        for l in range(1, self.N + 1):
            subsets = []
            for r in range(1, l + 1):
                for comb in combinations(range(1, l + 1), r):
                    subsets.append(list(comb))
            result[l] = subsets
        return result

    def compute_new_demand_total(self):
        C = [0] * (self.N + 1)
        for k in range(1, self.N + 1):
            C[k] = C[k - 1] + self.demand[k - 1]
        nd = {}
        for l in range(1, self.N + 1):
            for i in range(1, l + 1):
                nd[(i, l)] = C[l] - C[i - 1]
        if (1, 1) in nd:
            nd[(1, 1)] = nd[(1, 1)] - self.ini_inv
        return nd

    def createCons(self, name, x, inv, y):
        cons = self.model.createCons(self, name)
        subsets = self.generate_subsets()
        new_demand_total = self.compute_new_demand_total()
        cons.data = {
            "x": x,
            "inv": inv,
            "y": y,
            "subsets": subsets,
            "new_demand_total": new_demand_total,
            "T": list(self.T)
        }
        return cons

    def conscheck(self, constraints, solution, check_integrality,
                  check_lp_rows, print_reason, completely, **kwargs):
        tol = 1e-6
        model = self.model
        self.addedcuts = False

        for cons in constraints:
            x = cons.data["x"]
            inv = cons.data["inv"]
            y = cons.data["y"]
            subsets = cons.data["subsets"]
            new_demand_total = cons.data["new_demand_total"]
            T_local = cons.data["T"]

            for l in T_local:
                if l not in subsets:
                    continue
                for s_idx, S in enumerate(subsets[l]):
                    lhs = sum(model.getSolVal(solution, x[i]) for i in S)
                    rhs_terms = sum(new_demand_total[(i, l)] * model.getSolVal(solution, y[i]) for i in S)
                    inv_l_val = model.getSolVal(solution, inv[l])
                    rhs = rhs_terms + inv_l_val

                    if lhs > rhs + tol:
                        self.addedcuts = True
                        expr_lhs = quicksum(x[i] for i in S)
                        expr_rhs = quicksum(new_demand_total[(i, l)] * y[i] for i in S) + inv[l]
                        cons_name = f"ls_cuts_l{l}_s{s_idx}"
                        model.addCons(expr_lhs <= expr_rhs, name=cons_name)

        if self.addedcuts:
            return {"result": SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):    
        if self.addedcuts == False:
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass


def opt_model(demand, N, T, setup_cost, holding_cost, production_cost, M, ini_inv):
    model = Model("lot_sizing_lp_with_LS_cuts")
    # -----------------------------
    # Build model, and declare variables
    # -----------------------------
    
    x = {i: model.addVar(name=f"x_{i}", lb=0.0) for i in T}
    y = {i: model.addVar(name=f"y_{i}", vtype="B") for i in T}
    inv = {i: model.addVar(name=f"inv_{i}", lb=0.0) for i in T}
    
    # -----------------------------
    # Optimization model
    # -----------------------------
    model.setObjective(
        quicksum(setup_cost[t - 1] * y[t] + production_cost[t - 1] *
        x[t] + holding_cost[t - 1] * inv[t] for t in T), "minimize")
    
    for t in T:
        model.addCons(x[t] <= M[t - 1] * y[t], name=f"Setup({t})")
    
        if t == 1:
            model.addCons(ini_inv + x[1] == inv[1] + demand[0], name=f"FlowCons({1})")
        else:
            model.addCons(inv[t - 1] + x[t] == inv[t] + demand[t - 1], name=f"FlowCons({t})")
    
    model.data = x, inv, y
    return model

# -----------------------------
# Select the running example
# -----------------------------
example = 1

demand, N, T, setup_cost, holding_cost, production_cost, M, ini_inv = running_example(example)
model = opt_model(demand, N, T, setup_cost, holding_cost, production_cost, M, ini_inv)
x, inv, y = model.data

# -----------------------------
# Add conshdlr
# -----------------------------
def run_cut_hdlr(run):
    if run == True:
        cut_hdlr = LS_Cuts(T, demand, ini_inv)
        model.includeConshdlr(
            cut_hdlr, "ls_u_cuts", "(L,S)_lazy_cuts",
            sepapriority = -1,
            enfopriority = -1,
            chckpriority = -1,
            sepafreq = -1,
            propfreq = -1,
            eagerfreq = -1,
            maxprerounds = 0,
            delaysepa = False,
            delayprop = False,
            needscons = True,
            presoltiming=SCIP_PRESOLTIMING.FAST,
            proptiming=SCIP_PROPTIMING.BEFORELP
        )

        pycons = cut_hdlr.createCons("lazy_cons", x, inv, y)
        model.addPyCons(pycons)
    else:
        pass

# -----------------------------
# Run conshdlr
# -----------------------------
run = True
run_cut_hdlr(run)

# -----------------------------
# Solver parameters
# -----------------------------
model.redirectOutput()
model.printVersion()
model.setPresolve(SCIP_PARAMSETTING.OFF)
model.setSeparating(SCIP_PARAMSETTING.OFF)
model.setHeuristics(SCIP_PARAMSETTING.OFF)
model.setBoolParam("misc/allowstrongdualreds", 0)

# -----------------------------
# Solve The model
# -----------------------------
model.relax()
model.optimize()
model.printSol()