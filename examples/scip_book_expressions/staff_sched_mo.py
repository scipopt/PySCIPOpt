"""
staff_sched_mo.py:  multiobjective model for staff scheduling

Objectives:
    - minimize cost
    - minimize uncovered shifts

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import random
from pyscipopt import Model, quicksum, multidict

def staff_mo(I,T,N,J,S,c,b):
    """
    staff: staff scheduling
    Parameters:
        - I: set of members in the staff
        - T: number of periods in a cycle
        - N: number of working periods required for staff's elements in a cycle
        - J: set of shifts in each period (shift 0 == rest)
        - S: subset of shifts that must be kept at least consecutive days
        - c[i,t,j]: cost of a shit j of staff's element i on period t
        - b[t,j]: number of staff elements required in period t, shift j
    Returns a model with no objective function.
    """

    Ts = range(1,T+1)
    model = Model("staff scheduling -- multiobjective version")

    x,y = {},{}
    for t in Ts:
        for j in J:
            for i in I:
                x[i,t,j] = model.addVar(vtype="B", name="x(%s,%s,%s)" % (i,t,j))
            y[t,j] = model.addVar(vtype="C", name="y(%s,%s)" % (t,j))

    C = model.addVar(vtype="C", name="cost")
    U = model.addVar(vtype="C", name="uncovered")

    model.addCons(C >= quicksum(c[i,t,j]*x[i,t,j] for i in I for t in Ts for j in J if j != 0), "Cost")
    model.addCons(U >= quicksum(y[t,j] for t in Ts for j in J if j != 0), "Uncovered")

    for t in Ts:
        for j in J:
            if j == 0:
                continue
            model.addCons(quicksum(x[i,t,j] for i in I) >= b[t,j] - y[t,j], "Cover(%s,%s)" % (t,j))
    for i in I:
        model.addCons(quicksum(x[i,t,j] for t in Ts for j in J if j != 0) == N, "Work(%s)"%i)
        for t in Ts:
            model.addCons(quicksum(x[i,t,j] for j in J) == 1, "Assign(%s,%s)" % (i,t))
            for j in J:
                if j != 0:
                    model.addCons(x[i,t,j] + quicksum(x[i,t,k] for k in J if k != j and k != 0) <= 1,\
                                    "Require(%s,%s,%s)" % (i,t,j))
        for t in range(2,T):
            for j in S:
                model.addCons(x[i,t-1,j] + x[i,t+1,j] >= x[i,t,j], "SameShift(%s,%s,%s)" % (i,t,j))


    model.data = x,y,C,U
    return model



def optimize(model,cand,obj):
    """optimize: function for solving the model, updating candidate solutions' list
    Parameters:
        - model: Gurobi model object
        - cand: list of pairs of objective functions (for appending more solutions)
        - obj: name of a model's variable to setup as objective
    Returns the solver's exit status
    """
    # model.Params.OutputFlag = 0 # silent mode
    model.setObjective(obj,"minimize")

    model.optimize()
    x,y,C,U = model.data
    status = model.getStatus()
    if status == "optimal" or status == "bestsollimit": # todo GRB.Status.SUBOPTIMAL:
        sols = model.getSols()
        for sol in sols:
            cand.append((model.getVal(var=U,solution=sol),model.getVal(var=C,solution=sol)))

     #   for k in range(model.SolCount):
     #       model.Params.SolutionNumber = k
     #       cand.append(model.getVal(U),model.getVal(C))
    return status



def solve_segment(I,T,N,J,S,c,b):
    """
    solve_segment: segmentation for finding set of solutions for two-objective TSP
    Parameters:
        - I: set of members in the staff
        - T: number of periods in a cycle
        - N: number of working periods required for staff's elements in a cycle
        - J: set of shifts in each period (shift 0 == rest)
        - S: subset of shifts that must be kept at least consecutive days
        - c[i,t,j]: cost of a shit j of staff's element i on period t
        - b[t,j]: number of staff elements required in period t, shift j
    Returns list of candidate solutions
    """
    model = staff_mo(I,T,N,J,S,c,b)     # model for minimizing time
    x,y,C,U = model.data
    model.setRealParam("limits/time", 60)

    # store the set of solutions for plotting
    cand = []

    # first objective: cost
    stat1 = optimize(model,cand,C)
    stat2 = optimize(model,cand,U)

    if stat1 != "optimal" or stat2 != "optimal":
        return []

    ulist = [int(ui+.5) for (ui,ci) in cand]
    max_u = max(ulist)
    min_u = min(ulist)

    # add a time upper bound constraint, moving between min and max values
    UConstr = model.addCons(U <= max_u, "UConstr")


    for u_lim in range(max_u-1, min_u, -1):
        print("limiting u to",u_lim)
        UConstr.setAttr("RHS",u_lim)
        optimize(model,cand,C)

    return cand



if __name__ == "__main__":
    from pareto_front import pareto_front
    from staff_sched import make_data,make_data_trick
    # I,T,N,J,S,c,b = make_data()
    I,T,N,J,S,c,b = make_data_trick()
    cand_seg = solve_segment(I,T,N,J,S,c,b)

    print("candidate solutions:")
    for cand in cand_seg:
        print("\t",cand)

    front_seg = pareto_front(cand_seg)
    print("pareto front:",len(front_seg),"points out of",len(cand_seg))
    for cand in front_seg:
        print("\t",cand)


    try:
        import matplotlib.pyplot as P
    except:
        print("for graphics, install matplotlib")
        exit(0)

    P.clf()
    P.xlabel("uncovered shifts")
    P.ylabel("cost")
    P.title("Pareto front")

    ### # plot pareto front - scaling
    ### x = [xi for (xi,yi) in cand_seg]
    ### y = [yi for (xi,yi) in cand_seg]
    ### P.plot(x,y,"bo",c="grey")

    # plot pareto front - scaling
    x = [xi for (xi,yi) in front_seg]
    y = [yi for (xi,yi) in front_seg]
    P.plot(x,y,"bo",c="black")

    x = [xi for (xi,yi) in front_seg]
    y = [yi for (xi,yi) in front_seg]
    P.plot(x,y,c="black",lw=3,label="segmentation")
    P.legend()
    P.savefig("tsp_mo_pareto_staff.pdf",format="pdf")
    P.show()
