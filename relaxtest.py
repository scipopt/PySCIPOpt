#! usr/bin/env python3
from pyscipopt       import Model, SCIP_PARAMSETTING, Expr, Relax, Term, ExprCons
from SONCrelaxator   import *
from constrained     import *
from Poem.polynomial import *

import re
import numpy as np

def example():
    m = Model()
    #m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.disablePropagation()
    #m.setIntParam('lp/disablecutoff', 1)
    #m.setBoolParam('nlp/disable', True)
    #m.setIntParam('lp/solvefreq', -1)

    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")
    #x2 = m.addVar(vtype = "C", name = "x2")
    #x3 = m.addVar(vtype = "C", name = "x3")

    #TODO: find SONC polynomial with linear term, to split into objective - constraint
    #TODO: find examples with more variables
    """
    #shows that relaxator is really used, time ~17 sec #4.15sec
    #p = 7.93823555685 + 1.80070743765*x0**6 + 4.40432092848*x1**6 + 0.950088417526*x0**2*x1**2 + 2.2408931992*x0**1*x1**2 + 1.86755799015*x0**1*x1**4 - 0.977277879876*x0**2*x1**1 - 0.151357208298*x0**3*x1**1  - 0.103218851794*x0**4*x1**1
    #m.addCons(p<= 0)
    #m.addCons(x1-x0>=0)
    m.addCons(x0**2*x1**2+x0*x1+x0**4+x1**4-x0<=10)
    m.setObjective(-2*x0+x1)
    """
    """
    #SONC leads to 'cutoff' in SCIP (lowerbound = 5.21), hence infeasible, SCIP alone does not terminate (spatial branch and bound), time ~ 11 sec
    m.addCons(7.10008969*x1**2 + 8.53135095*x0**6 + 1.9507754*x0**4 - 0.50965218*x0*x1 - 0.4380743*x0**3  <= 0)
    m.addCons(0.77749036*x0**2 - 1.61389785*x0**3*x1 - 0.21274028*x0**2*x1 <= 0)
    m.setObjective(5.24276483 - 1.25279536*x1)
    """
    """
    #SONC get some cutoff even though lowerbound < solution?? (leads to infeasibility even though SCIP without relaxator finds solution), time ~ 30 sec #6.28sec
    m.addCons(4.27619696*x1**2 + 8.87643816*x0**6 - 0.21274028*x0**4 + 0.3869025*x0*x1  <= 0)
    m.addCons(- 0.51080514*x0**3  + 0.06651722*x0**2*x1 <= 0)
    m.addCons(-0.02818223*x0**2 + 0.42833187*x0**3*x1 <= 0)
    m.setObjective(6.89037448 -0.89546656*x0 - 1.18063218*x1)
    """
    """
    #reasonable solution, time ~ 18 sec #5.34sec (LP first 0.00)
    m.addCons(1.16786015*x1**2 + 5.22927676*x0**2*x1**2 + 0.71805392*x0**4 <= 0)
    m.addCons( - 0.85409574*x0**2*x1 -2.55298982*x0*x1 <= 0)
    m.setObjective(1.55352131 +0.3130677*x1)
    """
    """
    #unbounded x1 since occurs only with exponent 1, hence SONC does not work, time ~ 0.1sec
    m.addCons(x0 <= 3)
    m.addCons(x1 <= 4)
    m.addCons(8*x0**3 - 2*x0**4 - 8*x0**2 + x1 >= 2)
    m.addCons(32*x0**3 - 4*x0**4 - 88*x0**2 + 96*x0 + x1 >= 36)
    m.setObjective(-x0-x1)
    """
    """
    #Problem only containing linear and quadratic constraints, using sonc anyway shows that relaxator is really used (if do not return if polycons==False), time ~ 21secs #3.91sec
    m.addCons(0.25*x1 - 0.0625*x1**2 - 0.0625*x0**2 + 0.5*x0 <= 1)
    m.addCons(0.0714285714285714*x1**2 + 0.0714285714285714*x0**2 - 0.428571428571429*x1 - 0.428571428571429*x0 <= -1)
    #m.addCons((1<=x0)<=5.5)
    #m.addCons((1<=x1)<=5.5)
    m.chgVarLb(x0,1)
    m.chgVarLb(x1,1)
    m.chgVarUb(x0,5.5)
    m.chgVarUb(x1,5.5)
    m.setObjective(x1)
    """
    """
    #terminates, gives reasonable lower bound, time ~ 10sec #3.81sec (LP first 0.00)
    m.addCons(0<=0.5+x0**2*x1**1-x0**6*x1**4-x0**3*x1**3)
    m.addCons(-x0<=1-x0**4-x0**2*x1**4)
    m.setObjective(3+x0)
    """
    """
    #(a) Test Polynomials modified, terminates, good lower bound (exact to 5sf.), time ~ 24sec #5.72sec
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    m.addCons(x0**2*x1 + 3*x1 >=0)
    m.setObjective(-x0)
    """

    #(b) Test Polynomials modified, reasonable lower bound, time ~ 7sec #4.92sec
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    m.addCons(-x0**2*x1 - 3*x1 +0.5*x0 >=0)
    m.setObjective(-0.5*x0)

    """
    #(c) Test Polynomials modified, terminates, time ~ 25sec #faster if using orthant=[(0,1)] in POEM #6.30sec
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    #m.addCons(x0**2*x1 >=0) #if added, time ~ 30sec
    m.setObjective(3*x1-x0)
    """
    """
    #Example 4.8 (6)i. in Lower Bounds for a polynomial (Ghasemi, Marshall), lower bound: -1.6, sol: 0.0, time ~ 15sec #4.53sec (LP first 0.00)
    m.addCons(1-2*x1-6*x0**2-x0**4>=0)
    m.addCons(-x0**3-x1**4>=0)
    m.setObjective(x0+x1)
    """
    """
    #Example 4.8 (8), good bound (-7.01e-05, sol: -9.06e-07), time ~ 14sec #only one sol (instead of 3) probably due to rounding error #2.99sec
    m.addCons(2*x0**2+x1>=0)
    m.addCons(x1-x0**4*x1+x1**5-x0**6-x1**6>=0)
    m.addCons(x1-5*x0**2+x0**4*x1-x0**6-x1**6>=0)
    m.setObjective(-x1-1)
    """
    """
    #terminates, reasonable lower bound, time ~ 8sec #3.04sec (LP first 0.00)
    m.addCons(1-x0**4-x1**4>=0)
    m.addCons(x0**3+2>=0)
    m.setObjective(x0-2)
    """
    """
    #ex14_1_1, unbounded point x3, 4 variables, time ~ 0.1sec
    m.addCons(-x3+x0 <= 0)
    m.addCons(-x3+x0 >= 0)
    m.addCons(x3**2 + x0**2 +x2**2 <= 10)
    m.addCons(2*x2**2 + 4*x1*x2 - 42*x1 + 4*x1**3 - x3 <= 14)
    m.addCons(-2*x2**2 - 4*x1*x2 + 42*x1 - 4*x1**3 - x3 <= -14)
    m.addCons(2*x1**2 + 4*x1*x2 - 26*x2 + 4*x2**3 - x3 <= 22)
    m.addCons(-2*x1**2 - 4*x1*x2 + 6*x2 - 4*x2**3 - x3 <= -54)
    m.addCons((-5<=x1)<=5)
    m.addCons((-5<=x2)<=5)
    m.setObjective(x0)
    """
    """
    #Example 3.2 Ghasemi, Marshall, extended, time ~ 30sec #7.38sec (LP first 0.00)
    m.addCons(x0**2-2*x0*x1+x1**2>=0)
    m.addCons(-x0**4+x1-x0**3-x1**4+20+x0**2*x1>=0)
    m.setObjective(x0+x1)
    """
    """
    #Example instances/MINLP/circle.cip changed #3.62sec
    m.addCons(-(x0+2.545724188)**2 + (x1-9.983058643)**2 <= -x2**2)
    m.addCons((x0-8.589400372)**2 - (x1-6.208600402)**2  <= x2**2)
    m.addCons((x0-5.953378204)**2 + 4*(x1-9.920197351)**2 <= x2**2)
    m.addCons((x0-3.710241136)**2 - (x1-7.860254203)**2 <= -x2**2)
    m.addCons(-8.88178419700125e-16 + (x0-3.629909053)**2 - (x1-2.176232347)**2  <= -x2**2)
    #m.addCons((x0-3.016475803)**2 + (x1-6.757468831)**2 <= x2**2)
    #m.addCons(8.88178419700125e-16+ (x0-4.148474536)**2 + (x1-2.435660776)**2 <= x2**2)
    #m.addCons((x0-8.706433123)**2 + (x1-3.250724797)**2 <= x2**2)
    #m.addCons((x0-1.604023507)**2 + (x1-7.020357481)**2 <= x2**2)
    #m.addCons((x0-5.501896021)**2 + (x1-4.918207429)**2 <= x2**2)
    m.setObjective(x2)
    """
    return m

if __name__=="__main__":
    m = example()
    relaxator = SoncRelax()
    #which frequency is reasionable/sufficient? maybe also change priority param?
    m.includeRelax(relaxator,"SONCRelaxator", "Relaxator using SONC decompositions", freq=1) #priority=-1
    m.optimize()
    var = m.getVars()
    for v in var:
        print(v,"=", m.getVal(v))
    print("objective = ", m.getObjVal())

    n = example()
    n.disablePropagation()
    n.optimize()
    va = n.getVars()
    for v in va:
        print(v, "=", n.getVal(v))
    print("objective = ", n.getObjVal())
