##@file prodmix_soco.py
#@brief product mix model using soco.
"""
Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

def prodmix(I,K,a,p,epsilon,LB):
    """prodmix:  robust production planning using soco
    Parameters:
        I - set of materials
        K - set of components
        a[i][k] -  coef. matrix
        p[i] - price of material i
        LB[k] - amount needed for k
    Returns a model, ready to be solved.
    """

    model = Model("robust product mix")

    x,rhs = {},{}
    for i in I:
        x[i] = model.addVar(vtype="C", name="x(%s)"%i)
    for k in K:
        rhs[k] = model.addVar(vtype="C", name="rhs(%s)"%k)

    model.addCons(quicksum(x[i] for i in I) == 1)
    for k in K:
        model.addCons(rhs[k] == -LB[k]+ quicksum(a[i,k]*x[i] for i in I) )
        model.addCons(quicksum(epsilon*epsilon*x[i]*x[i] for i in I) <= rhs[k]*rhs[k])

    model.setObjective(quicksum(p[i]*x[i] for i in I), "minimize")

    model.data = x,rhs
    return model


def make_data():
    """creates example data set"""
    a = { (1,1):.25, (1,2):.15, (1,3):.2,
          (2,1):.3,  (2,2):.3,  (2,3):.1,
          (3,1):.15, (3,2):.65, (3,3):.05,
          (4,1):.1,  (4,2):.05,  (4,3):.8
          }
    epsilon = 0.01
    I,p = multidict({1:5, 2:6, 3:8, 4:20})
    K,LB = multidict({1:.2, 2:.3, 3:.2})
    return I,K,a,p,epsilon,LB


if __name__ == "__main__":
    I,K,a,p,epsilon,LB  = make_data()
    model = prodmix(I,K,a,p,epsilon,LB)
    model.optimize()
    print("Objective value:",model.getObjVal())
    x,rhs = model.data
    for i in I:
        print(i,": ",model.getVal(x[i]))
