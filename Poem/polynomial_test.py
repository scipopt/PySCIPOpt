#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Test suite for polynomial.py."""

import numpy as np
import sympy
import cvxpy as cvx

try:
	import matlab
	import matlab.engine
	matlab_found = True
	matlab_inst = matlab.engine.start_matlab('-useStartupFolderPref -nosplash -nodesktop')
except:
	matlab_found = False

from Poem.polynomial import Polynomial
import Poem.aux as aux


#===Creation of polynomials===

def test_from_string():
	test_string = '8.5 + 2.2*x2^4 + 2.9*x1^4*x2^2 + 2.5*x1^6 + 7.6*x0^2*x1^2*x2^2 + 4.8*x0^4 - 0.2*x0^1*x1^1*x2^1 + 1.5*x0^2*x1^2*x2^1 + 1.5*x0^1*x1^2*x2^1'
	success = True
	try:
		p = Polynomial(test_string)
	except Exception as err:
		print(repr(err))
		success = False
	assert(success)

def test_from_matrix_vector():
	A = np.array([[0,0],[10,0],[0,10],[6,8],[8,6],[1,1],[6,6],[8,1],[2,1]]).T
	b = [1,1,1,1,1,-1,-1,-1,-1]
	success = True
	try:
		p = Polynomial(A, b)
	except Exception as err:
		print(repr(err))
		success = False
	assert(success)

def test_from_parameters():
	success = True
	try:
		p1 = Polynomial('standard_simplex', 6, 18, 20, seed = 19)
		p2 = Polynomial('simplex', 5, 12, 23, seed = 61)
		p3 = Polynomial('general', 7, 14, 24, 8, seed = 17)
	except Exception as err:
		print(repr(err))
		success = False
	assert(success)

def test_standard_simplex_properties():
	n = 7
	for seed in range(10):
		p = Polynomial('standard_simplex', n, 12, 20, seed = seed)
		p._normalise()
		assert(len(p.hull_vertices) == n+1)
		assert(len(p.monomial_squares) >= len(p.hull_vertices))
		assert(p.degenerate_points == [])
		assert(p.has_zero)
		assert(p.init_time > 0)

def test_simplex_properties():
	n = 5
	for seed in range(10):
		p = Polynomial('simplex', n, 22, 27, seed = seed)
		p._normalise()
		assert(len(p.hull_vertices) == n+1)
		assert(len(p.monomial_squares) >= len(p.hull_vertices))
		assert(p.degenerate_points == [])
		assert(p.has_zero)
		assert(p.init_time > 0)

def test_general_properties():
	n = 4
	terms = 27
	inner = 15
	for seed in range(5):
		p = Polynomial('general', n, 26, terms, inner, seed = seed)
		p._normalise()
		p._compute_convex_hull()
		assert(len(p.hull_vertices) <= terms - inner)
		assert(len(p.monomial_squares) >= len(p.hull_vertices))
		assert(p.has_zero)
		assert(p.init_time > 0)

#===Arithmetic===
def test_eq():
	p = Polynomial('1 + x0^6 - 2*x0^3')
	q = Polynomial('x0^6 - 2*x0^3 + 1')
	assert(p == q)
	p = Polynomial([[1, 1, 1, 1, 1, 1], [0, 0, 6, 0, 3, 1], [0, 6, 0, 4, 2, 2]], [1,1,1,-1,1,-1])
	q = Polynomial(str(p))
	assert(p == q)

#===Output===
def test_str():
	p = Polynomial('standard_simplex', 4,8,11,seed = 0)
	#drop symbolic part, once __eq__ is defined
	assert(p.to_symbolic() - Polynomial(str(p)).to_symbolic() == 0)

def test_get_solutions():
	p = Polynomial('general',4,8,30,20,seed = 1)
	p._compute_cover()
	p.sonc_opt_python()
	p.sage_opt_python()
	sols = p.get_solutions()
	assert(len(sols) == 2)
	assert(set(p.old_solutions.keys()) == set(e[0] for e in sols))

def test_print_solutions():
	#just call and make sure they don't throw an error
	p = Polynomial('general',4,8,30,20,seed = 1)
	#Use _compute_cover, so we don't get a warning.
	p._compute_cover()
	p.opt()
	p.sonc_opt_python()

	p.print_solutions()
	p.print_solutions(only_valid = True)
	p.print_solutions(form = 'latex')

def test_call():
	A = np.array([[0,0],[10,0],[0,10],[6,8],[8,6],[1,1],[6,6],[8,1],[2,1]]).T
	b = [1,1,1,1,1,-1,-1,-1,-1]
	p = Polynomial(A, b)
	#cann call on lists
	assert(p([1,1]) == 1)
	#cann call on np-arrays
	assert(p(np.array([1,2])) == 1276)

def test_run_all():
	p = Polynomial('standard_simplex',2,10,16,seed = 8)
	p.run_all(keep_alive = True)
	#assert all algorithms are called
	if matlab_found:
		assert(len(p.old_solutions) == 11)
	else:
		assert(len(p.old_solutions) == 5)
	for sol in p.old_solutions.values():
		#assert that SONC solves it correctly
		assert(sol['verify'] == 1 or sol['strategy'] == 'sos' or sol['language'] == 'matlab')
		assert(set(sol.keys()) <= {'status', 'params', 'verify', 'init_time', 'time', 'C', 'lambda', 'opt', 'modeler', 'solver', 'solver_time', 'language', 'strategy', 'problem_creation_time','solution_time','cover_time','solution','index','error'})
		assert({'status','verify', 'init_time', 'time', 'C', 'opt', 'modeler', 'solver', 'solver_time', 'language', 'strategy'} <= set(sol.keys()))
		if sol['language'] == 'python':
			assert('problem_creation_time' in sol.keys())

def test_coefficient_reallocation():
	A = np.array([[0,0],[10,0],[0,10],[6,8],[8,6],[1,1],[6,6],[8,1],[2,1]]).T
	b = [1,1,1,1,1,-1,-1,-1,-1]
	p = Polynomial(A, b)
	p._compute_cover()
	p._compute_zero_cover()
	p.set_cover([circ for cover in p.old_covers.values() for circ in cover])
	opts = p.sonc_realloc()
	#The values should improve.
	for i in range(1,len(opts)):
		assert(opts[i-1] > opts[i])

def test_sonc_simplex():
	#the general cover can be worse than the zero-cover
	p = Polynomial('standard_simplex',3,20,77,seed = 2)
	p.sonc_opt_python()
	opt = p.solution['opt']
	p._compute_cover()
	p.sonc_opt_python()
	assert(opt < p.solution['opt'])

def test_sos_decomposition():
	p = Polynomial('general',3,6,10,5,seed = 0)
	p.sos_opt_python(solver = cvx.CVXOPT, sparse = False)
	l = p.get_decomposition()
	difference = p.to_symbolic() + p.solution['opt'] - sum([q.to_symbolic()**2 for q in l]).expand()
	coeffs = sympy.Poly(difference).coeffs()
	assert(max([abs(c) for c in coeffs]) <= aux.EPSILON)
	diff_poly = Polynomial(str(difference))
	assert(max(abs(diff_poly.b)) <= aux.EPSILON)

def test_sonc_decomposittion():
	p = Polynomial('general',3,8,10,5,seed = 0)
	p.sonc_opt_python()
	l = p.get_decomposition()
	difference = p.relax().to_symbolic() + p.solution['opt'] - sum([q.to_symbolic() for q in l])
	diff_poly = Polynomial(str(difference))
	assert(diff_poly.is_sum_of_monomial_squares(eps = 5 * aux.EPSILON))

def test_sonc_no_constant():
	p = Polynomial('x1^2 - x1*x2 + x2^2')
	p.sonc_opt_python()
	l = p.get_decomposition()
	difference = p.relax().to_symbolic() + p.solution['opt'] - sum([q.to_symbolic() for q in l])
	diff_poly = Polynomial(str(difference))
	assert(diff_poly.is_sum_of_monomial_squares(eps = aux.EPSILON))

	p = Polynomial('x1^2 - x1*x2 + x2^2 + x3^2 - x2*x3')
	p.sonc_opt_python()
	l = p.get_decomposition()
	difference = p.relax().to_symbolic() + p.solution['opt'] - sum([q.to_symbolic() for q in l])
	diff_poly = Polynomial(str(difference))
	assert(diff_poly.is_sum_of_monomial_squares(eps = aux.EPSILON))

def test_detect_infinity():
	p = Polynomial('1*x0^2*x1^2 + x0^2 + 0.9 * x1^2 -2 * x0*x1^2 - 2*x0^2*x1 + 0')
	assert(p.detect_infinity() is not None)

def test_branch_bound_order():
	#for every node in the tree, we must have min >= self_bound >= lower bound (from branching)
	def check_bound_order(node):
		assert(node.min[0] > node.lower_bound - aux.EPSILON)
		assert(node.lower_bound >= max([-sol['opt'] for sol in node.old_solutions.values() if sol['verify'] == 1]))
		if node.child0 is not None:
			return check_bound_order(node.child0)
		if node.child1 is not None:
			return check_bound_order(node.child1)

	p = Polynomial('general',5,12,21,15,seed = 8)
	p.traverse('min', reltol = 1e-3)
	check_bound_order(p)
	p = Polynomial('general',3,16,17,10,seed = 5)
	p.traverse('min')
	check_bound_order(p)

def test_branch_bound_existence():
	def check_bound_existence(node):
		#if 2 children, then we have run SONC
		if node.child0 is not None and node.child1 is not None and node.child0.lower_bound > np.inf and node.child1.lower_bound > np.inf:
			if 'sonc' not in [key[1] for key in node.old_solutions.keys()]:
				return False
		res = True
		if node.child0 is not None:
			res = res and check_bound_existence(node.child0)
		if node.child1 is not None:
			res = res and check_bound_existence(node.child1)
		return res

	p = Polynomial('general', 6, 24, 25, 14, seed = 5)
	p._create_search_tree()
	p.traverse('min', reltol = 1e-3)
	assert(check_bound_existence(p))
	p = Polynomial('general',5,12,21,15,seed = 8)
	p._create_search_tree()
	p.traverse('min', reltol = 1e-3, max_size = None)
	assert(check_bound_existence(p))

#=== Test Cases for Matlab ===

if matlab_found:

	#This test should be removed, when tested by anyone who is not the author.
	def test_from_database():
		success = True
		try:
			p = Polynomial(734, matlab_instance = matlab_inst)
		except Exception as err:
			print(repr(err))
			success = False
		assert(success)
		assert(p.matlab == matlab_inst)

	#This test should be removed, when tested by anyone who is not the author.
	def test_from_database_faulty():
		success = True
		try:
			p = Polynomial(-734)
		except Exception as err:
			print(repr(err))
			success = False
		assert(not success)

	def test_from_string():
		test_string = '8.5 + 2.2*x2^4 + 2.9*x1^4*x2^2 + 2.5*x1^6 + 7.6*x0^2*x1^2*x2^2 + 4.8*x0^4 - 0.2*x0^1*x1^1*x2^1 + 1.5*x0^2*x1^2*x2^1 + 1.5*x0^1*x1^2*x2^1'
		success = True
		try:
			p = Polynomial(test_string, matlab_instance = matlab_inst)
		except Exception as err:
			print(repr(err))
			success = False
		assert(success)
		assert(p.matlab == matlab_inst)

	def test_from_matrix_vector():
		A = np.array([[0,0],[10,0],[0,10],[6,8],[8,6],[1,1],[6,6],[8,1],[2,1]]).T
		b = [1,1,1,1,1,-1,-1,-1,-1]
		success = True
		try:
			p = Polynomial(A, b, matlab_instance = matlab_inst)
		except Exception as err:
			print(repr(err))
			success = False
		assert(success)
		assert(p.matlab == matlab_inst)

	def test_from_parameters():
		success = True
		try:
			p1 = Polynomial('standard_simplex', 6, 18, 20, seed = 19, matlab_instance = matlab_inst)
			p2 = Polynomial('simplex', 5, 12, 23, seed = 61, matlab_instance = matlab_inst)
			p3 = Polynomial('general', 7, 14, 24, 8, seed = 17, matlab_instance = matlab_inst)
		except Exception as err:
			print(repr(err))
			success = False
		assert(success)
		assert(p1.matlab == matlab_inst)
		assert(p2.matlab == matlab_inst)
		assert(p3.matlab == matlab_inst)

	def test_sonc_simplex_python_vs_matlab():
		#taking monomial squares into account must improve the solution
		#want some interior monomial squares
		p = Polynomial('standard_simplex',3,20,77,seed = 2, matlab_instance = matlab_inst)
		p.sonc_opt_python()
		opt = p.solution['opt']
		p.opt(language = 'matlab', method = 'outer')
		#cannot have better accuracy, since Matlab violates constraints
		assert(abs(opt/p.solution['opt'] - 1) <= 4e-5)

	def test_sonc_even_python_vs_matlab():
		#compare even splitting between Matlab and Python
		for seed in range(5):
			p = Polynomial('general',5,22,27,15, seed = seed, matlab_instance = matlab_inst)
			p.opt()
			opt = p.solution['opt']
			p.opt(language='matlab', method='even')
			for sol in p.old_solutions.values():
				assert(abs(opt/sol['opt'] - 1) <= aux.EPSILON * 30)
