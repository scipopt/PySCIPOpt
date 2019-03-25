##@file rcs.py
#@brief model for the resource constrained scheduling problem
"""
Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

def rcs(J,P,R,T,p,c,a,RUB):
    """rcs -- model for the resource constrained scheduling problem
    Parameters:
        - J: set of jobs
        - P: set of precedence constraints between jobs
        - R: set of resources
        - T: number of periods
        - p[j]: processing time of job j
        - c[j,t]: cost incurred when job j starts processing on period t.
        - a[j,r,t]: resource r usage for job j on period t (after job starts)
        - RUB[r,t]: upper bound for resource r on period t
    Returns a model, ready to be solved.
    """
    model = Model("resource constrained scheduling")

    s,x = {},{}   # s - start time variable;  x=1 if job j starts on period t
    for j in J:
        s[j] = model.addVar(vtype="C", name="s(%s)"%j)
        for t in range(1,T-p[j]+2):
            x[j,t] = model.addVar(vtype="B", name="x(%s,%s)"%(j,t))

    for j in J:
        # job execution constraints
        model.addCons(quicksum(x[j,t] for t in range(1,T-p[j]+2)) == 1, "ConstrJob(%s,%s)"%(j,t))

        # start time constraints
        model.addCons(quicksum((t-1)*x[j,t] for t in range(2,T-p[j]+2)) == s[j], "ConstrJob(%s,%s)"%(j,t))

    # resource upper bound constraints
    for t in range(1,T-p[j]+2):
        for r in R:
            model.addCons(
                quicksum(a[j,r,t-t_]*x[j,t_] for j in J for t_ in range(max(t-p[j]+1,1),min(t+1,T-p[j]+2))) \
                <= RUB[r,t], "ResourceUB(%s)"%t)

    # time (precedence) constraints, i.e., s[k]-s[j] >= p[j]
    for (j,k) in P:
        model.addCons(s[k] - s[j] >= p[j], "Precedence(%s,%s)"%(j,k))

    model.setObjective(quicksum(c[j,t]*x[j,t] for (j,t) in x), "minimize")

    model.data = x,s
    return model


def make_1r():
    """creates example data set 1"""
    J, p = multidict({       # jobs, processing times
        1 : 1,
        2 : 3,
        3 : 2,
        4 : 2,
        })
    P = [(1,2), (1,3), (2,4)]
    R = [1]
    T = 6
    c = {}
    for j in J:
        for t in range(1,T-p[j]+2):
            c[j,t] = 1*(t-1+p[j])
    a = {
        (1,1,0):2,
        (2,1,0):2, (2,1,1):1, (2,1,2):1,
        (3,1,0):1, (3,1,1):1,
        (4,1,0):1, (4,1,1):2,
        }
    RUB = {(1,1):2, (1,2):2, (1,3):1, (1,4):2, (1,5):2, (1,6):2}
    return (J,P,R,T,p,c,a,RUB)

def make_2r():
    """creates example data set 2"""
    J, p = multidict({       # jobs, processing times
        1 : 2,
        2 : 2,
        3 : 3,
        4 : 2,
        5 : 5,
        })
    P = [(1,2), (1,3), (2,4)]
    R = [1,2]
    T = 6
    c = {}
    for j in J:
        for t in range(1,T-p[j]+2):
            c[j,t] = 1*(t-1+p[j])
    a = {
        # resource 1:
        (1,1,0):2, (1,1,1):2,
        (2,1,0):1, (2,1,1):1,
        (3,1,0):1, (3,1,1):1, (3,1,2):1,
        (4,1,0):1, (4,1,1):1,
        (5,1,0):0, (5,1,1):0, (5,1,2):1, (5,1,3):0, (5,1,4):0,
        # resource 2:
        (1,2,0):1, (1,2,1):0,
        (2,2,0):1, (2,2,1):1,
        (3,2,0):0, (3,2,1):0, (3,2,2):0,
        (4,2,0):1, (4,2,1):2,
        (5,2,0):1, (5,2,1):2, (5,2,2):1, (5,2,3):1, (5,2,4):1,
        }
    RUB = {(1,1):2, (1,2):2, (1,3):2, (1,4):2, (1,5):2, (1,6):2, (1,7):2,
           (2,1):2, (2,2):2, (2,3):2, (2,4):2, (2,5):2, (2,6):2, (2,7):2 }
    return (J,P,R,T,p,c,a,RUB)


if __name__ == "__main__":
    (J,P,R,T,p,c,a,RUB) = make_2r()
    model = rcs(J,P,R,T,p,c,a,RUB)
    model.optimize()
    x,s = model.data

    print("Optimal value:",model.getObjVal())
    for (j,t) in x:
        if model.getVal(x[j,t]) > 0.5:
            print(x[j,t].name,"=",model.getVal(x[j,t]))

    for j in s:
        if model.getVal(s[j]) > 0.:
            print(s[j].name,"=",model.getVal(s[j]))
