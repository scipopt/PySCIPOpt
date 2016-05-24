"""
scheduling.py:  solve the one machine scheduling problem.
approaches:
    - linear ordering formulation
    - time-index formulation
    - disjunctive formulation
    - heuristics using cutting plane

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

from pyscipopt import Model, quicksum, multidict

def scheduling_linear_ordering(J,p,d,w):
    """
    scheduling_linear_ordering: model for the one machine total weighted tardiness problem

    Model for the one machine total weighted tardiness problem
    using the linear ordering formulation

    Parameters:
        - J: set of jobs
        - p[j]: processing time of job j
        - d[j]: latest non-tardy time for job j
        - w[j]: weighted of job j; the objective is the sum of the weighted completion time

    Returns a model, ready to be solved.
    """
    model = Model("scheduling: linear ordering")

    T,x = {},{} # tardiness variable; x[j,k] =1 if job j precedes job k, =0 otherwise
    for j in J:
        T[j] = model.addVar(vtype="C", name="T(%s)"%(j))
        for k in J:
            if j != k:
                x[j,k] = model.addVar(vtype="B", name="x(%s,%s)"%(j,k))

    for j in J:
        model.addCons(quicksum(p[k]*x[k,j] for k in J if k != j) - T[j] <= d[j]-p[j], "Tardiness(%r)"%(j))

        for k in J:
            if k <= j:
                continue
            model.addCons(x[j,k] + x[k,j] == 1, "Disjunctive(%s,%s)"%(j,k))

            for ell in J:
                if ell > k:
                    model.addCons(x[j,k] + x[k,ell] + x[ell,j] <= 2, "Triangle(%s,%s,%s)"%(j,k,ell))

    model.setObjective(quicksum(w[j]*T[j] for j in J), "minimize")


    model.data = x,T
    return model


def scheduling_time_index(J,p,r,w):
    """
    scheduling_time_index: model for the one machine total weighted tardiness problem

    Model for the one machine total weighted tardiness problem
    using the time index formulation

    Parameters:
        - J: set of jobs
        - p[j]: processing time of job j
        - r[j]: earliest start time of job j
        - w[j]: weighted of job j; the objective is the sum of the weighted completion time

    Returns a model, ready to be solved.
    """
    model = Model("scheduling: time index")
    T = max(r.values()) + sum(p.values())
    X = {}   # X[j,t]=1 if job j starts processing at time t, 0 otherwise
    for j in J:
        for t in range(r[j], T-p[j]+2):
            X[j,t] = model.addVar(vtype="B", name="x(%s,%s)"%(j,t))

    for j in J:
        model.addCons(quicksum(X[j,t] for t in range(1,T+1) if (j,t) in X) == 1, "JobExecution(%s)"%(j))

    for t in range(1,T+1):
        ind = [(j,t2) for j in J for t2 in range(t-p[j]+1,t+1) if (j,t2) in X]
        if ind != []:
            model.addCons(quicksum(X[j,t2] for (j,t2) in ind) <= 1, "MachineUB(%s)"%t)

    model.setObjective(quicksum((w[j] * (t - 1 + p[j])) * X[j,t] for (j,t) in X), "minimize")

    model.data = X
    return model


def scheduling_disjunctive(J,p,r,w):
    """
    scheduling_disjunctive: model for the one machine total weighted completion time problem

    Disjunctive optimization model for the one machine total weighted
    completion time problem with release times.

    Parameters:
        - J: set of jobs
        - p[j]: processing time of job j
        - r[j]: earliest start time of job j
        - w[j]: weighted of job j; the objective is the sum of the weighted completion time

    Returns a model, ready to be solved.
    """
    model = Model("scheduling: disjunctive")

    M = max(r.values()) + sum(p.values())       # big M
    s,x = {},{}      # start time variable, x[j,k] = 1 if job j precedes job k, 0 otherwise
    for j in J:
        s[j] = model.addVar(lb=r[j], vtype="C", name="s(%s)"%j)
        for k in J:
            if j != k:
                x[j,k] = model.addVar(vtype="B", name="x(%s,%s)"%(j,k))


    for j in J:
        for k in J:
            if j != k:
                model.addCons(s[j] - s[k] + M*x[j,k] <= (M-p[j]), "Bound(%s,%s)"%(j,k))

            if j < k:
                model.addCons(x[j,k] + x[k,j] == 1, "Disjunctive(%s,%s)"%(j,k))

    model.setObjective(quicksum(w[j]*s[j] for j in J), "minimize")

    model.data = s,x
    return model


def scheduling_cutting_plane(J,p,r,w):
    """
    scheduling_cutting_plane: heuristic to one machine weighted completion time based on cutting planes

    Use a cutting-plane method as a heuristics for solving the
    one-machine total weighted completion time problem with release
    times.

    Parameters:
        - J: set of jobs
        - p[j]: processing time of job j
        - r[j]: earliest start time of job j
        - w[j]: weighted of job j; the objective is the sum of the weighted completion time

    Returns a tuple with values of:
        - bestC: corresponding completion time of the best found solution
        - bestobj: best, best, hobj
    """

    model = Model("scheduling: cutting plane")

    C = {}   # completion time variable
    for j in J:
        C[j] = model.addVar(lb=r[j]+p[j], obj=w[j], vtype="C", name="C(%s)"%j)


    sumP = sum([p[j] for j in J])
    sumP2 = sum([p[j]**2 for j in J])
    model.addCons(C[j]*p[j] >= sumP2*0.5 + (sumP**2)*0.5, "AllJobs")

    model.setObjective(quicksum(w[j]*C[j] for j in J), "minimize")

    cut = 0
    bestobj = float("inf")
    while True:
        model.optimize()
        sol = sorted([(model.getVal(C[j]),j) for j in C])
        seq = [j for (completion,j) in sol]

        # print("Opt.value=",model.getObjVal()
        # print("current solution:", seq
        hC,hobj = evaluate(seq,p,r,w)
        if hobj < bestobj:
            # print("\t*** updating  best found solution:", bestobj, "--->", hobj, "***"
            bestC = hC
            bestobj = hobj
            best = list(seq)

        S,S_ = [],[]
        sumP,sumP2,f = 0,0,0
        for (C_,j) in sol:
            pj = p[j]
            delta = pj**2 + ((sumP+pj)**2 - sumP**2) - 2*pj*C_
            if f > 0 and delta <= 0:
                S = S_
                break
            f += delta/2.
            sumP2 += pj**2
            sumP += pj
            S_.append(j)

        if S == []:
            break

        cut += 1
        model.addCons(quicksum(C[j]*p[j] for j in S) >= sumP2*0.5 + (sumP**2)*0.5, "Cut(%s)"%cut)

    return bestC,bestobj,best



def evaluate(seq,p,r,w):
    C = 0       # completion time of previous job
    obj = 0
    for j in seq:
        s = max(r[j],C)
        C = max(r[j],C) + p[j]
        obj += w[j]*C
    return C,obj



def printsol(seq,p,r,w):
    print("Solution:",seq)
    C = 0       # completion time of previous job

    print("%12s%12s%12s%12s%12s%12s%12s" % ("job","p","r","w","start","completion","obj"))
    obj = 0
    for j in seq:
        s = max(r[j],C)
        C = max(r[j],C) + p[j]
        obj += w[j]*C
        print("%12s%12s%12s%12s%12s%12s%12s" % (j,p[j],r[j],w[j],s,C,w[j]*C))
    print("completion time:",C)
    print("objective:",obj)
    print
    return C



import random
def make_data(n):
    """
    Data generator for the one machine scheduling problem.
    """
    p,r,d,w = {},{},{},{}

    J = range(1,n+1)

    for j in J:
        p[j] = random.randint(1,4)
        w[j] = random.randint(1,3)

    T = sum(p)
    for j in J:
        r[j] = random.randint(0,5)
        d[j] = r[j] + random.randint(0,5)

    return J,p,r,d,w



def example(n):
    """
    Data generator for the one machine scheduling problem.
    """
    J,p,r,d,w = multidict({
        1:[1,4,0,3],
        2:[4,0,0,1],
        3:[2,2,0,2],
        4:[3,4,0,3],
        5:[1,1,0,1],
        6:[4,5,0,2],
        })
    return J,p,r,d,w



if __name__ == "__main__":
    random.seed(1)
    n = 5       # number of jobs
    # J,p,r,d,w = make_data(n)
    J,p,r,d,w = example(n)

    print("Linear ordering formulation")
    model = scheduling_linear_ordering(J,p,d,w)
    # model.write("scheduling-lo.lp")
    model.optimize()
    z = model.getObjVal()
    x,T = model.data
    for (i,j) in x:
        if model.getVal(x[i,j]) > .5:
            print("x(%s) = %s" % ((i,j), int(model.getVal(x[i,j])+.5)))
    for i in T:
        print("T(%s) = %s" % (i, int(model.getVal(T[i])+.5)))
    print("Opt.value by the linear ordering formulation=",z)

    print("Time index formulation")
    model = scheduling_time_index(J,p,r,w)
    model.optimize()
    X = model.data
    z = model.getObjVal() + sum([w[j]*p[j] for j in J])
    print("Optimal value by Time Index Formulation:",z)
    seq = [j for (t,j) in sorted([(t,j) for (j,t) in X if model.getVal(X[j,t]) > .5])]
    C1 = printsol(seq,p,r,w)

    print("Disjunctive formulation")
    model = scheduling_disjunctive(J,p,r,w)
    model.optimize()
    s,x = model.data
    z = model.getObjVal() + sum([w[j]*p[j] for j in J])
    print("Optimal value by Disjunctive Formulation:",z)
    seq = [j for (t,j) in sorted([(int(model.getVal(s[j])+.5),j) for j in s])]
    C2 = printsol(seq,p,r,w)
    assert C2 == C1

    print("Earliest start time rule:")
    tmp = sorted([(r[i],i) for i in J])
    seq = [i for (dummy,i) in sorted(tmp)]
    printsol(seq,p,r,w)

    print("Smith's rule: decreasing order of w/p")
    tmp = [(float(w[i])/p[i],i) for i in J]
    seq = [i for (dummy,i) in sorted(tmp,reverse=True)]
    printsol(seq,p,r,w)

    print("Scheduling heuristics with cutting planes")
    C,obj,seq = scheduling_cutting_plane(J,p,r,w)
    printsol(seq,p,r,w)
