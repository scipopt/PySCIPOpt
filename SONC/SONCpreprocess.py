#! usr/bin/env python3
from pyscipopt  import SCIP_RESULT, Relax, Term, Expr
from POEM       import Polynomial, build_lagrangian_GP, OptimizationProblem, Constraint, InfeasibleError
from convert    import ExprToPoly

import numpy as np
import cvxpy as cvx
import re

#TODO: use flags to decide which preprocessing options should be used?
#TODO: is it better not to have the problem in SCIP and POEM, but only use one of them?
def preprocess(problem, Vars):
    """base function calling all preprocessing options"""
    #make sure that we have constraints '>= 0'
    problem = greaterConstraints(problem)
    x0found = re.search(r'x0',str(Vars))
    problem, Vars = rewriteObjective(problem, Vars, x0found) #index gives the variable that was deleted when rewriting the objective
    problem = boundNegativeTerms(problem, Vars)
    return problem

def greaterConstraints(problem):
    for con in problem.constraints:
        if con.operand == '<=':
            con.b *= -1
            con.operand = '>='
        elif con.operand == '==':
            con.operand = '>='
            c = -1*con.b
            problem.constraints.append(Constraint(Polynomial(con.A,c),'>='))
    return problem

#TODO: instead of deleting inside A,b maybe a better idea to just set the coefficients to 0 and add a function that deletes those (makes index unnecessary)
def rewriteObjective(problem, Vars, x0found):
    """function tries to find variables that was only added to make the objective linear and rewrites it back into a nonlinear objective, deletes the variable
     param: problem - optimization problem of class OptimizationProblem in POEM
    """
    obj = problem.objective
    cons = problem.constraints
    if np.count_nonzero(obj.A[1:,]) == 1:
        index = np.nonzero(obj.A[1:,])
        A = obj.A[1:,].copy()
        for con in cons:
            conA = con.A[1:,]
            i = np.nonzero(conA[index[0][0]])
            if np.count_nonzero(conA) != 1 and np.count_nonzero(conA[index[0][0]]) == 1 and conA[index[0][0]][i] == 1 and con.b[i] == 1 and con.operand=='>=':
                con.b*=-1
                cons = [c for c in cons if c!=-con and c != con and (np.count_nonzero(c.A[index[0]+1])!=1 or np.count_nonzero(c.A[1:,])!=1) ]
                for c in cons:
                    c.A = np.delete(c.A,index[0]+1,axis = 0)
                con.b[i] += 1
                con.A = np.delete(con.A, index[0]+1,axis=0)
                newProb = OptimizationProblem()
                newProb.setObjective(Polynomial(con.A,con.b))
                newProb.addCons(cons)
                Vars = np.delete(Vars,index[0][0])
                return newProb, Vars

    #return original problem if nothing could be rewritten
    return problem, Vars

def boundNegativeTerms(problem, Vars):
    """Try to use the bounds x<=u on variables given by branch and bound to find a valid cover for all negative terms in the polynomial.
    This is done by finding even exponents a with x^a <= u^a.
    returns problem with bound constraints added.
    params: - problem: problem of type OptimizationProblem in POEM
            - Vars: list of all variables in the problem
    """
    #TODO: use only when necessary and use only the necessary directions
    n=len(Vars)
    if problem.constraints !=[]:
        mu = cvx.Variable(len(problem.constraints), pos=True, name='mu')
        polyOrig,mu_constraints = build_lagrangian_GP(mu,problem)
    else:
        polyOrig = problem.objective
        polyOrig._normalise()
    try:
        #if all points are already covered, just add the bounds as constraints x^2<=u^2
        polyOrig._compute_cover()
        a = 2*np.ones(n,dtype=int)
        A = polyOrig.A[1:,]
        VarExist=np.zeros(n,dtype=int)
        for i in range(polyOrig.A.shape[1]):
            if np.count_nonzero(A[:,i])==1 and polyOrig.b[i]>=0:
                index = np.nonzero(A[:,i])
                if A[index,i]%2 == 0:
                    if not VarExist[index]:
                        VarExist[index] = 1
                        a[index] = A[index,i]
                    else:
                        a[index] = min(a[index],A[index,i])

    except: #InfeasibleError
        #if not all points are covered, 2*max{exponents of inner terms} will cover all points if bounds are given for the constraints
        A = polyOrig.A[1:]
        a = np.array([2*max(A[i,polyOrig.non_squares]) for i in range(n)])
    for i,y in enumerate(Vars):
        u = max(abs(y.getUbLocal()),abs(y.getLbLocal()))
        if u != problem.infinity and abs(u**a[i])<=10e6:
            #TODO: only works if variables are not reordered at some point
            boundconsA = np.zeros((n,2))
            boundconsA[i][1] = a[i]
            boundconsb = [-u**a[i],1]
            boundcons = Polynomial(boundconsA,boundconsb)
            problem.addCons(Constraint(boundcons, '<='))
    return problem
