"""
tsp-mp.py:  solve the multi-objective traveling salesman problem

Approaches:
    - segmentation
    - ideal point
    - scalarization

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

import math
import random
from pyscipopt import Model, quicksum, multidict

def optimize(model,cand):
    """optimize: function for solving the model, updating candidate solutions' list
    Will add to cand all the intermediate solutions found, as well as the optimum
    Parameters:
        - model: Gurobi model object
        - cand: list of pairs of objective functions (for appending more solutions)
    Returns the solver's exit status
    """
    model.hideOutput()
    model.optimize()
    x,y,C,T = model.data
    status = model.getStatus()
    if status == "optimal":
        # collect suboptimal solutions
        solutions = model.getSols()
        for sol in solutions:
            cand.append((model.getSolVal(T, sol), model.getSolVal(C)))
    return status


def base_model(n,c,t):
    """base_model: mtz model for the atsp, prepared for two objectives
    Loads two additional variables/constraints to the mtz model:
        - C: sum of travel costs
        - T: sum of travel times
    Parameters:
        - n: number of cities
        - c,t: alternative edge weights, to compute two objective functions
    Returns list of candidate solutions
    """
    from atsp import mtz_strong
    model = mtz_strong(n,c)     # model for minimizing cost
    x,u = model.data

    # some auxiliary information
    C = model.addVar(vtype="C", name="C")       # for computing solution cost
    T = model.addVar(vtype="C", name="T")       # for computing solution time

    model.addCons(T == quicksum(t[i,j]*x[i,j] for (i,j) in x), "Time")
    model.addCons(C == quicksum(c[i,j]*x[i,j] for (i,j) in x), "Cost")

    model.data = x,u,C,T
    return model


def solve_segment_time(n,c,t,segments):
    """solve_segment: segmentation for finding set of solutions for two-objective TSP
    Parameters:
        - n: number of cities
        - c,t: alternative edge weights, to compute two objective functions
        - segments: number of segments for finding various non-dominated solutions
    Returns list of candidate solutions
    """

    model = base_model(n,c,t)   # base model for minimizing cost or time
    x,u,C,T = model.data

    # store the set of solutions for plotting
    cand = []
    # print("optimizing time"
    model.setObjective(T, "minimize")
    stat1 = optimize(model,cand)

    # print("optimizing cost"
    model.setObjective(C, "minimize")
    stat2 = optimize(model,cand)

    if stat1 != "optimal" or stat2 != "optimal":
        return []

    times = [ti for (ti,ci) in cand]
    max_time = max(times)
    min_time = min(times)
    delta = (max_time-min_time)/segments
    # print("making time range from",min_time,"to",max_time

    # add a time upper bound constraint, moving between min and max values
    TimeCons = model.addCons(T <= max_time, "TimeCons")

    for i in range(segments+1):
        time_ub = max_time - delta*i
        model.chgRhs(TimeCons, time_ub)
        # print("optimizing cost subject to time <=",time_ub
        optimize(model,cand)

    return cand


def solve_ideal(n,c,t,segments):
    """solve_ideal: use ideal point for finding set of solutions for two-objective TSP
    Parameters:
        - n: number of cities
        - c,t: alternative edge weights, to compute two objective functions
        - segments: number of segments for finding various non-dominated solutions
    Returns list of candidate solutions
    """
    model = base_model(n,c,t)   # base model for minimizing cost or time
    x,u,C,T = model.data

    # store the set of solutions for plotting
    cand = []
    # print("optimizing time"
    model.setObjective(T, "minimize")
    stat1 = optimize(model,cand)

    # print("optimizing cost"
    model.setObjective(C, "minimize")
    stat2 = optimize(model,cand)    #find the minimum cost routes

    if stat1 != "optimal" or stat2 != "optimal":
        return []

    times = [ti for (ti,ci) in cand]
    costs = [ci for (ti,ci) in cand]
    min_time = min(times)
    min_cost = min(costs)
    # print("ideal point:",min_time,",",min_cost

    #===============================================================
    # Objective function is f1^2 + f2^2 where f=Sum tx-min_time and g=Sum cx-min_cost
    f1 = model.addVar(vtype="C", name="f1")
    f2 = model.addVar(vtype="C", name="f2")

    model.addCons(f1 == T - min_time, "obj1")
    model.addCons(f2 == C - min_cost, "obj2")

    # print("optimizing distance to ideal point:"
    for i in range(segments+1):
        lambda_ = float(i)/segments
        # print(lambda_
        z = model.addVar(name="z")
        Obj = model.addCons(lambda_*f1*f1 + (1-lambda_)*f2*f2 == z)
        model.setObjective(z, "minimize")
        optimize(model, cand)    #  find the minimum cost routes
    return cand


def solve_scalarization(n,c,t):
    """solve_scalarization: scale objective function to find new point
    Parameters:
        - n: number of cities
        - c,t: alternative edge weights, to compute two objective functions
    Returns list of candidate solutions
    """

    model = base_model(n,c,t)   # base model for minimizing cost or time
    x,u,C,T = model.data

    def explore(C1,T1,C2,T2,front):
        """explore: recursively try to find new non-dominated solutions with a scaled objective
        Parameters:
            - C1,T1: cost and time of leftmost point
            - C1,T1: cost and time of rightmost point
            - front: current set of non-dominated solutions
        Returns the updated front
        """
        alpha = float(C1 - C2)/(T2 - T1)
        # print("%s,%s -- %s,%s  (%s)..." % (C1,T1,C2,T2,alpha)
        init = list(front)
        model.setObjective(quicksum((c[i,j] + alpha*t[i,j])*x[i,j] for (i,j) in x), "minimize")
        optimize(model,front)
        front = pareto_front(front)
        # print("... added %s points" % (len(front)-len(init))
        if front == init:
            # print("no points added, returning"
            return front

        CM = model.getVal(C)
        TM = model.getVal(T)
        # print("will explore %s,%s -- %s,%s and %s,%s -- %s,%s" % (C1,T1,CM,TM,CM,TM,C2,T2)
        if TM > T1:
            front = explore(C1,T1,CM,TM,front)
        if T2 > TM:
            front = explore(CM,TM,C2,T2,front)
        return front


    # store the set of solutions for plotting
    cand = []       # to store the set of solutions for plotting
    model.setObjective(T, "minimize")

    stat = optimize(model,cand)
    if stat != "optimal":
        return []
    C1 = model.getVal(C)
    T1 = model.getVal(T)

    # change the objective function to minimize the travel cost
    model.setObjective(C, "minimize")

    stat = optimize(model,cand)
    if stat != "optimal":
        return []
    C2 = model.getVal(C)
    T2 = model.getVal(T)

    front = pareto_front(cand)
    return explore(C1,T1,C2,T2,front)


def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def make_data(n):
    x,y = {},{} # positions in the plane
    c,t = {},{} # cost, time
    for i in range(1,n+1):
        x[i] = random.random()
        y[i] = random.random()
    for i in range(1,n+1):
        for j in range(1,n+1):
            c[i,j] = distance(x[i],y[i],x[j],y[j])
            t[i,j] = 1/(c[i,j]+1.0)+0.3*random.random()
    return c,t,x,y


if __name__ == "__main__":
    from pareto_front import pareto_front

    random.seed(7)
    n = 20
    c,t,x,y = make_data(n)


    print("\n\n\nmultiobjective optimization: segmentation with additional (time) constraint")
    segments = 6
    cand_seg_time = solve_segment_time(n,c,t,segments)
    print("candidate solutions:")
    for cand in cand_seg_time:
        print("\t",cand)

    front_seg_time = pareto_front(cand_seg_time)
    print("pareto front:",len(front_seg_time),"points out of",len(cand_seg_time))
    for cand in front_seg_time:
        print("\t",cand)


    print("\n\n\nmultiobjective optimization: min distance to ideal point")
    cand_ideal = solve_ideal(n,c,t,segments)
    print("candidate solutions:")
    for cand in cand_ideal:
        print("\t",cand)

    front_ideal = pareto_front(cand_ideal)
    print("pareto front:",len(front_ideal),"points out of",len(cand_ideal))
    for cand in front_ideal:
        print("\t",cand)


    print("\n\n\nmultiobjective optimization: scalarization strategy")
    front_scalarization = solve_scalarization(n,c,t)
    front_scalarization.sort()
    print("front solutions:")
    for cand in front_scalarization:
        print("\t",cand)
    assert front_scalarization == pareto_front(front_scalarization)


    try:
        import matplotlib.pyplot as P
    except:
        print("for graphics, install matplotlib")
        exit(0)

    P.clf()
    P.xlabel("cost")
    P.ylabel("time")
    P.title("Pareto front")

    # plot pareto front - scalarization
    t = [ti for (ti,ci) in front_scalarization]
    c = [ci for (ti,ci) in front_scalarization]
    P.plot(t,c,"bo",c="black")

    t = [ti for (ti,ci) in front_scalarization]
    c = [ci for (ti,ci) in front_scalarization]
    P.plot(t,c,c="black",lw=3,label="scalarization")

    # plot pareto front - segmentation (time)
    t = [ti for (ti,ci) in cand_seg_time]
    c = [ci for (ti,ci) in cand_seg_time]
    P.plot(t,c,"bo",c="cyan")

    allpoints = [(ti,ci) for (ti,ci) in cand_seg_time]
    t = [ti for (ti,ci) in front_seg_time]
    c = [ci for (ti,ci) in front_seg_time]
    P.plot(t,c,c="cyan",lw=3,label="segmentation (time)")

    # plot pareto front - -ideal point
    t = [ti for (ti,ci) in cand_ideal]
    c = [ci for (ti,ci) in cand_ideal]
    P.plot(t,c,"bo",c="red")

    allpoints = [(ti,ci) for (ti,ci) in cand_ideal]
    t = [ti for (ti,ci) in front_ideal]
    c = [ci for (ti,ci) in front_ideal]
    P.plot(t,c,c="red",lw=3,label="ideal point")

    P.legend()
    P.savefig("tsp_mo_pareto_ideal.pdf",format="pdf")
    P.show()
