#! usr/bin/env python3
from POEM       import Polynomial, build_lagrangian_GP, OptimizationProblem, Constraint, InfeasibleError

import numpy as np
import cvxpy as cvx

def preprocess(problem, Vars):
    """base function calling all preprocessing options"""
    #make sure that we have constraints '>= 0'
    problem = greaterConstraints(problem)
    
    problem, Vars = rewriteObjective(problem, Vars)
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

def rewriteObjective(problem, Vars):
    """
    function tries to find variable that was only added to make the objective linear and rewrites it back into a nonlinear objective, deletes the variable
     - param problem: optimization problem of class OptimizationProblem in POEM
     - param Vars: list of variables of given problem
    """
    obj = problem.objective
    cons = problem.constraints
    #find only variable occuring in objective
    if np.count_nonzero(obj.A[1:,]) == 1:
        index = np.nonzero(obj.A[1:,])
        A = obj.A[1:,].copy()

        #find constraint that replaces the objective variable
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

                #construct new problem with nonlinear objective, constraint deleted, variable deleted
                newProb = OptimizationProblem()
                newProb.setObjective(Polynomial(con.A,con.b))
                newProb.addCons(cons)
                
                #update list of variables
                Vars = np.delete(Vars,index[0][0])
                return newProb, Vars

    #return original problem if nothing could be rewritten
    return problem, Vars

def boundNegativeTerms(problem, Vars):
    """
    Try to use the bounds x<=u on variables given by branch and bound to find a valid cover for all negative terms in the polynomial.
    This is done by finding even exponents a with x^a <= u^a, where u is maximum of lower and upper bound of x.
    returns problem with bound constraints added.
     - param problem: problem of type OptimizationProblem in POEM
     - param Vars: list of all variables in the problem
    """
    n=len(Vars)
    if problem.constraints !=[]:
        #constrained case, build lagrangian
        mu = cvx.Variable(len(problem.constraints), pos=True, name='mu')
        polyOrig = build_lagrangian_GP(mu, problem) #mu_bounds
    else:
        #unconstrained case, use objective
        polyOrig = problem.objective
        polyOrig._normalise()

    try:
        #if all points are already covered, just add the bounds as constraints x^2<=u^2
        polyOrig._compute_zero_cover()
        A = polyOrig.A[1:,]
        coverST = [polyOrig.cover[0].copy()]
        for cover in polyOrig.cover[1:]:
            if 0 not in cover:
                cover.insert(0,0)
            added = False
            for el in coverST:
                monos = [e for e in el if e in polyOrig.monomial_squares]
                if (len(cover[:-1])==len(monos) and cover[:-1]==monos):
                    el.append(cover[-1])
                    added = True
                    break
            if added == False:
                coverST.append(cover.copy())
        #if all(np.count_nonzero(A[:,i])!=1 for i in range(polyOrig.A.shape[1])):
        if len(coverST)==1:
            a = 2*np.ones(n,dtype=int)
        else:
            if n%2==0:
                a = np.array([2*n*max(A[i,polyOrig.non_squares]) for i in range(n)])
            else:
                a = np.array([2*n*max(A[i,polyOrig.non_squares]) for i in range(n)])
            problem.neededBounds=a
        '''else:
            a = np.ones(n,dtype=int) #2*
            VarExist=np.zeros(n,dtype=int)
            for i in range(polyOrig.A.shape[1]):
                if np.count_nonzero(A[:,i])==1: #and polyOrig.b[i]>=0:
                    index = np.nonzero(A[:,i])
                    if i in polyOrig.non_squares: #A[index,i]%2==0
                        if not VarExist[index]:
                            VarExist[index] = 1
                            a[index] = A[index,i]
                        else:
                            a[index] = min(a[index],A[index,i])'''


    except InfeasibleError:
        #if not all points are covered, use suitable exponent to cover all
        A = polyOrig.A[1:]
        if n%2==0:
            a = np.array([2*n*max(A[i,polyOrig.non_squares]) for i in range(n)])
        else:
            a = np.array([2*n*max(A[i,polyOrig.non_squares]) for i in range(n)])
        problem.neededBounds=a

    l = len(problem.constraints)
    for i,y in enumerate(Vars):
        #do not add Variables that do not occur in the subproblem
        if not A[i,:].any():
            a[i] = 0
        if a[i]%2 == 0:
            for j in range(A.shape[1]):
                if A[i,j] == a[i] and np.count_nonzero(A[:,j]) == 1:
                    #monomial square already exists, need to reduce exponent to fulfill assumptions of theorem
                    if l != 0 and a[i] > 0 and type(polyOrig.b[j][1]) in {int,float} and polyOrig.b[j][1] == 0:
                        a[i] -= 1
                        break
                    elif l == 0 and a[i] > 0 and  polyOrig.b[j] > 0:
                        a[i] -= 1
                        break

        #add bounds as constraints x^a<=u^a to problem
        u = max(abs(y.getUbLocal()),abs(y.getLbLocal()))
        if a[i]!=0 and u != problem.infinity:#TODO: sometime runtime overflow because power is too large
            boundconsA = np.zeros((n,2))
            boundconsA[i][1] = a[i]
            if u == abs(y.getLbLocal()) and a[i]%2==1:
                boundconsb = [u**a[i],1]
            else:
                boundconsb = [u**a[i],-1]
            boundcons = Polynomial(boundconsA,boundconsb)
            problem.addCons(Constraint(boundcons, '>='))

    return problem
