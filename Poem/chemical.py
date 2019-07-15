#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

from Poem.polynomial import *
import Poem.polytope as polytope

s = open('../instances/4.10_simplh5denomcleared_irr_nok3_6freevars.txt','r').read().replace('\n','')
repl = [('x9','x0'),('k1','x2'),('k5','x3'),('d1','x4'),('d2','x5')]
for k,v in repl:
	s = s.replace(k,v)

p0 = Polynomial(s)
p0.b = np.array(p0.b, dtype = np.int)
p1 = Polynomial(p0.A[1:,:], p0.b, orthant = np.array([1,1,1,1,1,1]))
p1.clean()
#eliminate all negative vertices
l = []
for i in p1.non_squares:
	v = p1.A[:,i]
	A = p1.A[:,p1.monomial_squares]
	if(polytope.is_in_convex_hull_cvxpy((A,v))):
		l.append(i)
p = Polynomial(np.concatenate((p1.A[:,p1.monomial_squares],p1.A[:,l]),axis = 1),list(p1.b[p1.monomial_squares]) + list(p1.b[l]))

idx = [i for i in range(p.A.shape[1]) if p.A[5,i] == 0]
q = Polynomial(p.A[[1,2,3,4,6],:][:,idx], p.b[idx])
q.clean()

idx = [i for i in range(p0.A.shape[1]) if p0.A[5,i] == 0]
A = p0.A[[1,2,3,4,6],:][:,idx]
#A *= 2
q0 = Polynomial(A, p0.b[idx])
q0.clean()
q0.b = np.array(q0.b, dtype = np.int)
