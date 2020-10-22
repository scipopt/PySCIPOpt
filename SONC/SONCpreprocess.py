#! usr/bin/env python3
from pyscipopt  import SCIP_RESULT, Relax, Term, Expr, Model
from POEM       import Polynomial, InfeasibleError, build_lagrangian_GP, constrained_scipy, solve_GP, OptimizationProblem, Constraint
from convert    import ExprToPoly, PolyToExpr

import numpy as np
import cvxpy as cvx

#base function, calling all preprocessing options
#TODO: use flags to decide which preprocessing options should be used?
def preprocess(problem, SCIPprob):
    problem, index = rewriteObjective(problem, SCIPprob) #index gives the variable that was deleted when rewriting the objective
    problem = boundNegativeTerms(problem, SCIPprob, index)
    return problem

def rewriteObjective(problem, SCIPprob):
    obj = problem.objective
    cons = problem.constraints
    #print('problem to be rewritten: ',problem)
    if np.count_nonzero(obj.A[1:,]) == 1:
        index = np.nonzero(obj.A[1:,])
        A = obj.A[1:,]
        #print('index',index)
        for con in cons:
            conA = con.A[1:,]
            #print('constraint', con, con.operand)
            i = np.nonzero(conA[index[0][0]])
            #print('i',i, conA[index[0][0]])
            if np.count_nonzero(conA) != 1 and np.count_nonzero(conA[index[0][0]]) == 1 and conA[index[0][0]][i] == 1 and con.b[i] == -1:
                #cons.remove(con)
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
                return newProb, index[0][0]
        #print(conA, conA[index])
    
    #print(newProb)
    return problem, -1

def boundNegativeTerms(problem, SCIPprob, index):
    """Try to use the bounds x<=u on variables given by branch and bound to find a valid cover for all negative terms in the polynomial.
    This is done by finding even exponents a with x^a <= u^a."""
    #TODO: use only when necessary and use only the necessary directions
    n=len(SCIPprob.getVars())
    if problem.constraints !=[]:
        mu = cvx.Variable(len(problem.constraints), pos=True, name='mu')
        polyOrig = build_lagrangian_GP(mu,problem)
    else:
        #print(problem)
        polyOrig = problem.objective
        polyOrig._normalise()

    m=len(polyOrig.monomial_squares)
    l=len(polyOrig.non_squares)
    var = SCIPprob.getVars()
    if index != -1:
        var = np.delete(var,index)
        n-=1
    #print(var,index, SCIPprob.getVars())
    varOrig = SCIPprob.getVars()
    a = np.zeros(n,dtype=int)
    A = polyOrig.A[1:]
    #print('polyOrig',polyOrig.non_squares, polyOrig.monomial_squares)
    #print(max(polyOrig.A[1,polyOrig.non_squares]), polyOrig.b,A)
    #minimal_exponent(polyOrig,n)
    for i in range(n):
        a[i] = 2*max(A[i,polyOrig.non_squares])


    for i,y in enumerate(var):
        u = max(abs(y.getUbLocal()),abs(y.getLbLocal()))
        #print(u, y.getUbLocal(),y.getLbLocal(), y.getLbGlobal())
        if u != SCIPprob.infinity():
            t = Term()
            for k in range(a[i]):
                t += Term(varOrig[i])
            #print(a[i],t,n)
            #if y.getUbLocal() != SCIPprob.infinity() and y.getLbLocal() != -SCIPprob.infinity():
            boundcons = ExprToPoly(Expr({Term(): -u**a[i], t:1.0}), n)
            problem.addCons(Constraint(boundcons, '<='))
    #print('problem',problem)
    return problem

#TODO: Do we really need the minimal exponent or is the feasible solution enough?
def minimal_exponent(polynomial,n):
    constraints = []
    c = cvx.Variable(len(polynomial.non_squares),pos=True,name='c')
    a = cvx.Variable(n,pos=True,name='a')
    for i,y in enumerate(polynomial.non_squares):
        print('polyA',polynomial.A,polynomial.non_squares,polynomial.A[1:,y])
        constraints.append(cvx.sum(a*polynomial.A[1:,y])==c[i])
        for non in polynomial.non_squares:
            if non != y:
                constraints.append(c[i]>=cvx.sum(polynomial.A[1:,non]))
    lp = cvx.Problem(cvx.Minimize(cvx.sum(c)),constraints)
    lp.solve()
    print('lp',lp,lp.value,a.value,c.value,n)
    return polynomial


