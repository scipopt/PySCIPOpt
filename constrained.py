#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

from Poem.polynomial import *
from Poem.exceptions import InfeasibleError
from Poem.aux import *

import scipy
import cvxpy            as cvx


def constrained_opt(objective, constraint_list, start=[]):
    """Optimise objective under given constraints.
    :param: objective: objective function given as polynomial
    :param: constraint_list: list of all (polynomial) constraints
    :param: start: start point for scipy.optimize.minimize (optional, default value: [1,...,1])

    returns scipy.optimize.OptimizeResult with status 0-9 from SLSQP or status == 10 if InfeasibleError occurs
    """
    
    def lower_bound(lamb):
        """computes a lower bound of build_lagrangian via SONC"""
        lagrangian = build_lagrangian(lamb, objective, constraint_list)

        #make sure that same polynomial is not computed twice
        if str(lagrangian) in lagrangian_list.keys():
            return lagrangian_list[str(lagrangian)]

        #TODO: only compares to poly with lamb = [1,...,1] but should compare to last computed polynomial (use lagrangian_list?)
        #can reuse cover if only the coeffs changed since cover only depends on matrix A, not on b
        if lagrangian.A.shape == poly.A.shape:
            lagrangian.prob_sonc = poly.prob_sonc
            lagrangian.cover = poly.cover
            lagrangian.lamb = poly.lamb
        else:
            lagrangian._compute_cover()
        try:
            lagrangian.sonc_opt_python()
        except:
            lagrangian_list[str(lagrangian)] = -np.inf
            return -np.inf

        #save computed polynomial with corresponding solution
        lagrangian_list[str(lagrangian)] = -lagrangian.lower_bound
        return -lagrangian.lower_bound

    t0 = datetime.now()
    #number of constraints
    m = len(constraint_list)
    if start == []:
        start = np.ones(m)
    poly = build_lagrangian(start, objective, constraint_list)
    #compute cover of polynomial once, to save time later (TODO: maybe change that), or return if InfeasibleError
    try:
        poly._compute_cover()
    except InfeasibleError as err:
        return scipy.optimize.OptimizeResult({'success': False, 'message':err, 'status': 10, 'fun':np.inf, 'x':['nan']*m, 'nfev':0, 'njev':0,'nit':0})
    lagrangian_list = dict()

    #optimize lower_bound(lamb) with respect to lamb, return InfeasibleError if SONC decomposition is not well-defined (i.e. if unbounded points exist)
    #constraints in minimize are to make sure that lamb>0
    #TODO: make sure, that verify == 1
    try:
        data = scipy.optimize.minimize(lower_bound, start, method = 'SLSQP', constraints = scipy.optimize.LinearConstraint(np.eye(m), aux.EPSILON, np.inf), options = {'maxiter': 40})
    except InfeasibleError as err:
        return scipy.optimize.OptimizeResult({'success': False, 'message':err, 'status': 10, 'fun':np.inf, 'x':['nan']*m, 'nfev':0, 'njev':0,'nit':0})
    print('Lower bound: %.6f\nMultipliers: %s' % (-data.fun, data.x))
    print('Time: %.2f' % aux.dt2sec(datetime.now() - t0))
    return data


#TODO: instead of computing new polynomial maybe just update the old one?
def build_lagrangian(lamb, objective, constraint_list):
    """build the function obj-sum(lamb*g_i), where obj is the objective and gi are the constraints,
    for lamb fixed this function gives a lower bound for obj on {x: g_i(x)>=0}
    :param: lamb: lagrangian multiplier lambda
    :param: objective: objective function
    :param: constraint_list: list of constraints"""
    res = objective.copy()
    for i in range(len(constraint_list)):
        summand = constraint_list[i].copy()
        summand.b = np.array(summand.b * lamb[i], dtype = np.float)
        res -= summand
    return Polynomial(res.A, res.b)



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

    data = constrained_opt(f,[g1,g2], dict())
    #print(-data.fun, data.x)
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
