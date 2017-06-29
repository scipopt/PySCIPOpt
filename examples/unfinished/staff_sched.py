"""
staff_sched.py:  model for staff scheduling

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
import random
from pyscipopt import Model, quicksum, multidict

def staff(I,T,N,J,S,c,b):
    """
    staff: staff scheduling
    Parameters:
        - I: set of members in the staff
        - T: number of periods in a cycle
        - N: number of working periods required for staff's elements in a cycle
        - J: set of shifts in each period (shift 0 == rest)
        - S: subset of shifts that must be kept at least consecutive days
        - c[i,t,j]: cost of a shift j of staff's element i on period t
        - b[t,j]: number of staff elements required in period t, shift j
    Returns a model, ready to be solved.
    """
    Ts = range(1,T+1)
    model = Model("staff scheduling")

    x = {}
    for t in Ts:
        for j in J:
            for i in I:
                x[i,t,j] = model.addVar(vtype="B", name="x(%s,%s,%s)" % (i,t,j))


    model.setObjective(quicksum(c[i,t,j]*x[i,t,j] for i in I for t in Ts for j in J if j != 0),
                       "minimize")

    for t in Ts:
        for j in J:
            if j == 0:
                continue
            model.addCons(quicksum(x[i,t,j] for i in I) >= b[t,j], "Cover(%s,%s)" % (t,j))
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


    model.data = x
    return model


def make_data():
    T = 7               # number of periods
    N = 5               # number of working periods of each staff element in a cycle
    J = range(4)        # shift set; 0 == rest
    S = [2,3]           # subset of shifts that must be kept at least consecutive days
    # staff set, base cost
    I,c_base = multidict({1:8000, 2:9000, 3:10000, 4:11000, 5:12000, 6:13000, 7:14000, 8:15000})
    c = {}
    for i in I:
        for t in range(1,T+1):
            for j in J:
                if j == 0:
                    continue
                c[i,t,j] = c_base[i]
                if j == 3:      # night shift, more expensive
                    c[i,t,j] *= 2
                if t == T-1 or t == T:    # weekend, more expensive
                    c[i,t,j] *= 1.5
    b = {
        (1,1):2, (1,2):3, (1,3):1,
        (2,1):2, (2,2):3, (2,3):1,
        (3,1):2, (3,2):2, (3,3):1,
        (4,1):1, (4,2):1, (4,3):1,
        (5,1):3, (5,2):3, (5,3):1,
        (6,1):4, (6,2):4, (6,3):2,
        (7,1):5, (7,2):5, (7,3):2,
        }
    return I,T,N,J,S,c,b


def make_data_trick():
    T = 7               # number of periods
    N = 5               # number of working periods of each staff element in a cycle
    J = range(4)        # shift set; 0 == rest
    S = [2,3]           # subset of shifts that must be kept at least consecutive days
    # staff set, base cost
    I,c_base = multidict({1:8000, 2:9000, 3:10000, 4:11000, 5:12000, 6:13000, 7:14000, 8:15000})
    c = {}
    for i in I:
        for t in range(1,T+1):
            for j in J:
                if j == 0:
                    continue
                c[i,t,j] = c_base[i]
                if j == 3:      # night shift, more expensive
                    c[i,t,j] *= 2
                if t == T-1 or t == T:    # weekend, more expensive
                    c[i,t,j] *= 1.5
    b = {
        (1,1):2, (1,2):2, (1,3):2,
        (2,1):2, (2,2):2, (2,3):2,
        (3,1):2, (3,2):2, (3,3):2,
        (4,1):2, (4,2):2, (4,3):2,
        (5,1):2, (5,2):2, (5,3):2,
        (6,1):2, (6,2):2, (6,3):2,
        (7,1):2, (7,2):2, (7,3):2,
        }
    return I,T,N,J,S,c,b


if __name__ == "__main__":
    I,T,N,J,S,c,b = make_data_trick()
    # I,T,N,J,S,c,b = make_data()
    model = staff(I,T,N,J,S,c,b)
    model.optimize()
    status = model.getStatus()

    if status == "optimal":
        x = model.data
        print("Optimum solution found")
        print("\n\nstaff schedule: (shift on each day)")
        for i in I:
            s = "worker %s:\t" % i
            for t in range(1,T+1):
                for j in J:
                    if model.getVal(x[i,t,j]) > .5:
                        s += str(j)
            print(s)
        print("\n\nuncovered shifts:")
        # for t in range(1,T+1):
            # s = "day %s:\t" % t
            # for j in J:
                # if y[t,j].X > .5:
                        # s += "%s:%s, " % (j,int(y[t,j].X+.5))
            # print(s)

    elif status == "infeasible":
        print("Infeasible instance...")
    #    model.computeIIS()
     #   for c in model.getConss():
      #      if c.IISConstr:
       #         print(c.ConstrName)

    elif status == "unbounded" or status == "infeasible":
        print("Unbounded instance")

    else:
        print("Error: Solver finished with non-optimal status",status)
