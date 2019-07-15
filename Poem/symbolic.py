#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

from sympy.ntheory.continued_fraction import continued_fraction_iterator, continued_fraction_reduce
import sympy
from Poem.polynomial import *
from Poem.circuit_polynomial import *
from datetime import datetime

import Poem.aux as aux
from Poem.aux import to_fraction, symlog
from Poem.AGE_polynomial import AGEPolynomial

x = sympy.IndexedBase('x')

t0 = datetime.now()

#p = Polynomial(1272)
p = Polynomial('general', 3, 8, 12, 6, seed = 1)
p = Polynomial(p.A, np.array((100*p.b).round(), dtype = np.int))
p._compute_cover()
p.sonc_opt_python(reltol = aux.EPSILON/2)
C = p.solution['C']
M = sympy.Matrix(np.zeros(C.shape, dtype = np.int))
#rounding the b[i] will not be necessary, if they are considered exact integers
b = [to_fraction(sq, bound = -1) for sq in p.b[p.monomial_squares]]
#for i in range(p.monomial_squares):
#	if sum(M[:,i]) > b[i]:
#		print(i)
#	#M[:,i] *= b[i] / sum(M[:,i])
epsilon = aux.EPSILON/2/p.A.shape[0]/len(p.cover)
for k in range(len(p.cover)):
	if 0 in p.cover[k]:
		lamb = sympy.linsolve((sympy.Matrix(p.A[:,p.cover[k][:-1]]), sympy.Matrix(p.A[:,p.cover[k][-1]])), [x[i] for i in range(p.A.shape[0])])
		lamb = next(iter(lamb))
		for i in range(1,len(p.cover[k]) - 1):
			M[k,p.cover[k][i]] = to_fraction(C[k,p.cover[k][i]], eps = epsilon * lamb[0]/lamb[i], bound = -1)
		M[k,0] = to_fraction(lamb[0] * (to_fraction(-p.coefficient_distribution[k,p.cover[k][-1]], eps = epsilon, bound = 1) * sympy.prod([(lamb[r] / M[k,p.cover[k][r]]) ** lamb[r] for r in range(1,len(p.cover[k]) - 1)])) ** (1/lamb[0]), bound = 1)
	else:
		M[k,:] = sympy.Matrix([[to_fraction(C[k,j], bound = 1) for j in range(C.shape[1])]])
	M[k,p.cover[k][-1]] = to_fraction(p.b[p.cover[k][-1]])

decomposition = [CircuitPolynomial(p.A[:,p.cover[k]], np.array(M[k,p.cover[k]])[0]) for k in range(len(p.cover))]
for q in decomposition:
	q.circuit_number()

#for q in decomposition:
#	q.clean()
    
print('SONC time: %.2f seconds' % aux.dt2sec(datetime.now() - t0))


frac = np.vectorize(to_fraction, cache = True)
box_v = np.vectorize(aux.get_box, cache = True)

p.sage_opt_python()

t0 = datetime.now()

C = p.solution['C']
C_sy = np.zeros(C.shape, dtype = object)
lamb = p.solution['lambda']
for i in p.non_squares:
	C[i,i] *= -1
	C_sy[i,:] = frac(C[i,:])
	lamb[i,i] *= -1

lamb_sy = np.zeros(lamb.shape, dtype = object)

box_all = box_v(lamb, 16)
for i in p.non_squares:
	box = list(zip(box_all[0][i], box_all[1][i]))
	lamb_sy[i,:] = aux.LP_solve_exact(p.A, np.zeros(p.A.shape[0], dtype = np.int), box = box)

C_sy *= np.array(np.array(p.relax().b, dtype = np.int) / C_sy.sum(axis = 0))

C_sy[p.monomial_squares,0] = 0

decomp = []

for j in p.non_squares:
	idx = [i for i in range(1, C_sy.shape[0]) if i != j and lamb_sy[j,i] != 0]
	C_sy[j,0] = to_fraction(sympy.exp(to_fraction((symlog(lamb_sy[j,idx] / C_sy[j,idx]) * lamb_sy[j,idx] - lamb_sy[j,idx]).sum() + lamb_sy[j,0]*sympy.log(lamb_sy[j,0]) - lamb_sy[j,0] - C_sy[j,j], bound = 1) / lamb_sy[j,0]), bound = 1)
	decomp.append(AGEPolynomial(p.A, C_sy[j,:], lamb = lamb_sy[j,:], orthant = np.ones(p._variables, dtype = np.int)))

print('SAGE time: %.2f seconds' % aux.dt2sec(datetime.now() - t0))
