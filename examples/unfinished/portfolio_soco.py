"""
portfolio_soco.py:  modified markowitz model for portfolio optimization.

Approach: use second-order cone optimization.

Copyright (c) by Joao Pedro PEDROSO, Masahiro MURAMATSU and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict

import math
def phi_inv(p):
    """phi_inv: inverse of gaussian (normal) CDF
    Source:
        Handbook of Mathematical Functions
        Dover Books on Mathematics
        Milton Abramowitz and Irene A. Stegun (Editors)
        Formula 26.2.23.
        """
    if p < 0.5:
        t = math.sqrt(-2.0*math.log(p))
        return ((0.010328*t + 0.802853)*t + 2.515517)/(((0.001308*t + 0.189269)*t + 1.432788)*t + 1.0) - t
    else:
        t = math.sqrt(-2.0*math.log(1.0-p))
        return t - ((0.010328*t + 0.802853)*t + 2.515517)/(((0.001308*t + 0.189269)*t + 1.432788)*t + 1.0)


def p_portfolio(I,sigma,r,alpha,beta):
    """p_portfolio -- modified markowitz model for portfolio optimization.
    Parameters:
        - I: set of items
        - sigma[i]: standard deviation of item i
        - r[i]: revenue of item i
        - alpha: acceptance threshold
        - beta: desired confidence level
    Returns a model, ready to be solved.
    """

    model = Model("p_portfolio")

    x = {}
    for i in I:
        x[i] = model.addVar(vtype="C", name="x(%s)"%i)  # quantity of i to buy
    rho = model.addVar(vtype="C", name="rho")
    rhoaux = model.addVar(vtype="C", name="rhoaux")

    model.addCons(rho == quicksum(r[i]*x[i] for i in I))
    model.addCons(quicksum(x[i] for i in I) == 1)

    model.addCons(rhoaux == (alpha - rho)*(1/phi_inv(beta))) #todo
    model.addCons(quicksum(sigma[i]**2 * x[i] * x[i] for i in I) <=  rhoaux * rhoaux)

    model.setObjective(rho, "maximize")

    model.data = x
    return model



if __name__ == "__main__":
    # portfolio
    I,sigma,r = multidict(
        {1:[0.07,1.01],
         2:[0.09,1.05],
         3:[0.1,1.08],
         4:[0.2,1.10],
         5:[0.3,1.20]}
        )
    alpha = 0.95
    # beta = 0.1

    for beta in [0.1, 0.05, 0.02, 0.01]:
        print("\n\n\nbeta:",beta,"phi inv:",phi_inv(beta))
        model = p_portfolio(I,sigma,r,alpha,beta)
        model.optimize()

        x = model.data
        EPS = 1.e-6
        print("Investment:")
        print("%5s\t%8s" % ("i","x[i]"))
        for i in I:
            print("%5s\t%8g" % (i,model.getVal(x[i])))

        print("Objective:",model.getObjVal())
