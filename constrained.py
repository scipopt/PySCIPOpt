#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

from Poem.polynomial import *
from Poem.exceptions import InfeasibleError

import scipy
import cvxpy            as cvx


def constrained_opt(objective, constraint_list, solved_instance):
    """Optimise objective under given constraints."""
    def build_lagrangian(lamb):
        """build the function obj-sum(lamb*gi), where obj is the objective and gi are the constraints, 
        for lamb fixed this function gives a lower bound for f on {x: gi(x)>=0}"""
        res = objective.copy()
        for i in range(len(constraint_list)):
            summand = constraint_list[i].copy()
            summand.b = np.array(summand.b * lamb[i], dtype = np.float)
            res -= summand
        return Polynomial(res.A, res.b)

    def lower_bound(lamb):
        """computes a lower bound of build_lagrangian"""
        lagrangian = build_lagrangian(lamb)
        lagrangian.run_sonc()
        return -lagrangian.lower_bound

    t0 = datetime.now()
    #number of constraints
    m = len(constraint_list)

    #return if instance was already solved
    poly = build_lagrangian(np.ones(m))
    if str(poly) in solved_instance.keys():
        return solved_instance[str(poly)], poly

    #optimize lower_bound(lamb) with respect to lamb, return InfeasibleError if SONC decomposition is not well-defined (i.e. if unbounded points exist)
    #TODO: make sure, that verify == 1
    try:
        data = scipy.optimize.minimize(lower_bound, np.ones(m), method = 'SLSQP', constraints = scipy.optimize.LinearConstraint(np.eye(m), aux.EPSILON, np.inf, keep_feasible = True), options = {'maxiter': 40})
    except InfeasibleError as err:
        success = err
        return success, poly
    print('Lower bound: %.2f\nMultipliers: %s' % (-data.fun, data.x))
    print('Time: %.2f' % aux.dt2sec(datetime.now() - t0))
    return data, poly 
 
#TODO: can we use these functions for anything?
def unite_matrices(A_list):
    res = []
    for A in A_list:
        for col in A.T:
            if col.tolist() not in res:
                res.append(col.tolist())
    res = np.array(res, dtype = np.int).T
    res = res[:,np.lexsort(np.flipud(res))]
    return res

def expand(p, A):
    b = np.zeros(A.shape[1])
    exponents = A.T.tolist()
    for i,col in enumerate(p.A.T.tolist()):
        ind = exponents.index(col)
        b[ind] = p.b[i]
    return b


if __name__ == "__main__":

    #n = 6
    #d = 10
    #t = 84

    #np.random.seed(2)
    #random.seed(5)

    #t0 = datetime.now()

    #A = gen._create_exponent_matrix(n, d, t)
    #b = np.random.randn(t)

    #p0 = Polynomial(A,b)

    f = Polynomial('x0^2*x1 + 3*x1 - x0')
    g1 = Polynomial('-x0^4-x1^4+42')
    g2 = Polynomial('-x0^4+3*x0-2*x1^2+1')

    data = constrained_opt(f,[g1,g2])
    print(-data.fun, data.x)
"""
    A = unite_matrices([f.A, g1.A, g2.A])
    B = np.array([expand(p,A) for p in [f,g1,g2]])

    p = f + g1 + g2
    p = Polynomial(p.A, p.b)
    
    A,b = p.A, p.relax().b

    n,t = A.shape
    X = cvx.Variable(shape = (t,t), name = 'X', nonneg = True)
    #lamb[k,i]: barycentric coordinate, using A[:,i] to represent A[:,k]
    lamb = cvx.Variable(shape = (t,t), name = 'lambda', nonneg = True)
    #we use both variables only for k >= monomial_squares

    ##X[k,i]: weight of b[i] in the k-th summand
    #X = cvx.Variable(shape = (t,t), name = 'X')
    ##lamb[k,i]: barycentric coordinate, using A[:,i] to represent A[:,k]
    #lamb = cvx.Variable(shape = (t,t), name = 'lambda')
    mu = cvx.Variable(shape = 2, name = 'mu', nonneg = True)
    gamma = cvx.Variable()

    constraints = []
    #constraints += [b_relax[i] == -2*X[i,i] + cvx.sum(X[:,i]) for i in self.non_squares]
    #constraints += [b_relax[i] == cvx.sum(X[:,i]) for i in self.monomial_squares[1:]]
    #b = B.sum(axis = 0)
    #constraints += [B[0,:] + B[1,:] + B[2,:] == cvx.sum(X[:,i]) for i in range(1, t)]
    constraints += [gamma + B[0,0] + mu[0]*B[1,0] + mu[1]*B[2,0] == cvx.sum(X[:,0]) - 2*X[0,0]]
    constraints += [B[0,i] + mu[0]*B[1,i] + mu[1]*B[2,i] == cvx.sum(X[:,i]) - 2*X[i,i] for i in range(1, t)]
    constraints += [2*lamb[k,k] == cvx.sum(lamb[k,:]) for k in range(t)]
    constraints += [cvx.sum([A[:,i] * lamb[k,i] for i in range(t) if i != k]) == A[:,k]*lamb[k,k] for k in range(t)]
    constraints += [cvx.sum(cvx.kl_div(lamb[k,:], X[k,:])[[i for i in range(t) if i != k]]) <= -2*X[k,k] + cvx.sum(X[k,:]) for k in range(t)]
    #constraints += [lamb[k,k] == -cvx.sum(lamb[k,[i for i in range(t) if i != k]]) for k in range(t)]
    #constraints += [cvx.sum([A[:,i] * lamb[k,i] for i in range(t)]) == np.zeros(n) for k in range(t)]
    #constraints += [cvx.sum([cvx.kl_div(lamb[k,i], X[k,i]) for i in range(t) if i != k]) <= cvx.sum(X[k,:]) for k in range(t)]
    #constraints += [lamb[k,i] >= 0 for k in range(t) for i in range(t) if i != k]

    objective = cvx.Minimize(gamma)
    prob_sage = cvx.Problem(objective, constraints)

    p.sage_opt_python()
    #C = p.solution['C'].round(5)
    #lamb = p.solution['lambda'].round(5)
    #for i in range(C.shape[0]):
    #   C[i,i] *= -1
    #   lamb[i,i] *= -1
    #lamb[p.monomial_squares] = 0

    #prob_sage.variables()[0].value = C
    #prob_sage.variables()[1].value = lamb
"""
