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
    #print('before pre', problem, Vars)
    problem, Vars = rewriteObjective(problem, Vars, x0found) #index gives the variable that was deleted when rewriting the objective
    #print(problem)
    problem = boundNegativeTerms(problem, Vars)
    #print('after pre', problem, Vars)
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
        #print(index, A,Vars)
        #x0found = re.search(r'x0',str(cons))
        if x0found == None:
            #A = np.concatenate((np.zeros(A.shape[0]),A), axis=0)
            #print(A.shape)
            #print(A)
            A = np.concatenate(([np.zeros(A.shape[1])], A))
            #print('index',index)
        for con in cons:
            conA = con.A[1:,]
            if x0found == None:
                conA = np.concatenate(([np.zeros(conA.shape[1])], conA))
            #print('constraint', con, con.operand)
            i = np.nonzero(conA[index[0][0]])
            #print('i',i, conA[index[0][0]])
            if np.count_nonzero(conA) != 1 and np.count_nonzero(conA[index[0][0]]) == 1 and conA[index[0][0]][i] == 1 and con.b[i] == 1 and con.operand=='>=':
                #cons.remove(con)
                con.b*=-1
                #print('should be', [c.A for c in cons if np.count_nonzero(c.A[index[0]+1])!=1 or np.count_nonzero(c.A[1:,])!=1])
                cons = [c for c in cons if c!=-con and c != con and (np.count_nonzero(c.A[index[0]+1])!=1 or np.count_nonzero(c.A[1:,])!=1) ]
                for c in cons:
                    #print('negative',c,-con)
                    #print('remove', c.A[index[0]+1], c.A[1:,],len(cons))
                    #print([c.A for c in cons])
                    #if c == -con:
                    #    cons.remove(c)
                    #elif np.count_nonzero(c.A[index[0]+1])==1 and np.count_nonzero(c.A[1:,])==1:
                    #    cons.remove(c)
                    c.A = np.delete(c.A,index[0]+1,axis = 0)
                #print('new cons',[(c.A,c.b) for c in cons])
                con.b[i] += 1
                con.A = np.delete(con.A, index[0]+1,axis=0)
                #problem.obj = con
                #problem.obj._normalise()
                #problem.constraints = cons
                newProb = OptimizationProblem()
                newProb.setObjective(Polynomial(con.A,con.b))
                #TODO: do not just delete the last constraint, but make sure only the right ones are added
                newProb.addCons(cons)
                #print('new objective: ', newProb.objective.A,newProb.objective.b)
                #for c in newProb.constraints:
                #    print('new constraints: ',c.A,c.b)
                #print(newProb)
                Vars = [el for el in Vars if re.search('x'+str(index[0][0]), str(el))==None]
                #Vars = np.delete(Vars,index[0][0])
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
        #print(problem)
        #if all points are already covered, just add the bounds as constraints x^2<=u^2
        polyOrig._compute_cover()
        a = 2*np.ones(n,dtype=int) #TODO: use smallest exponent that already exists
        A = polyOrig.A[1:,]
        VarExist=np.zeros(n,dtype=int)
        for i in range(polyOrig.A.shape[1]):
            #print(polyOrig.A,polyOrig.b)
            #print(i, polyOrig.A[1:,], polyOrig.A.shape[1])
            #print('A', A[:,i])
            if np.count_nonzero(A[:,i])==1 and polyOrig.b[i]>=0:
                #a[i] = np.nonzero(A[i])[0]
                #print('index',np.nonzero(A[:,i]))
                index = np.nonzero(A[:,i])
                #print('element in A[index,i]',A[index,i])
                if A[index,i]%2 == 0:
                    if not VarExist[index]:
                        VarExist[index] = 1
                        a[index] = A[index,i]
                    else:
                        a[index] = min(a[index],A[index,i])
            #print('a',a)



    except:
        #if not all points are covered, 2*max{exponents of inner terms} will cover all points if bounds are given for the constraints
        A = polyOrig.A[1:]
        a = np.array([2*max(A[i,polyOrig.non_squares]) for i in range(n)])
        #print('a uncovered',a, n)
    for i,y in enumerate(Vars):
        u = max(abs(y.getUbLocal()),abs(y.getLbLocal()))
        #print('u',y,u, a)
        #print(polyOrig.non_squares, polyOrig.monomial_squares)
        if u != problem.infinity :#and abs(u**a[i])<=10e4:
            #TODO: only works if variables are not reordered at some point
            boundconsA = np.zeros((n,2))
            boundconsA[i][1] = a[i]
            boundconsb = [-u**a[i],1]
            boundcons = Polynomial(boundconsA,boundconsb)
            problem.addCons(Constraint(boundcons, '<='))
    return problem
