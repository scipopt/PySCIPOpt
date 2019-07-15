#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Provide functions to create polynomials in sparse notation."""

import random
import numpy as np
import scipy
from datetime import datetime

import Poem.aux as aux
from Poem.aux import binomial, dt2sec
from Poem.polytope import convex_hull, is_in_convex_hull_cvxpy

def motzkin():
	"""Create Motzkin-Polynomial in sparse notation.

	Call:
		A, b = motzkin()
	Output:
		A: matrix, containing the (affine) exponents
		b: coefficients
	"""
	A = np.array([[1,1,1,1],[0,2,4,2],[0,4,2,2]])
	b = [1,1,1,-3]
	return A,b

def _create_exponent_matrix(n, degree, terms = 0, seed = None):
	"""Create an exponent matrix in n variables, with given degree and number of terms.

	Call:
		A = _create_exponent_matrix(n, degree[, terms])
	Input:
		n: non-negative integer
		degree: non-negative integer
		terms (optional, default 0): non-negative integer
	Output:
		A: (`n` x `terms`) matrix, where the sum of each column is at most `degree`
	"""
	if seed is not None:
		random.seed(seed)
	size = binomial(n + degree, n)
	if terms > size or terms == 0:
		terms = size
	
	index_list = random.sample(range(1,size), terms)
	index_list.sort()
	return np.array([aux._index_to_vector(i, n, degree) for i in index_list]).T

def make_affine(A, zeros = True):
	"""Add a column of zeroes and a row of ones to input A.

	Call:
		res = make_affine(A)
	Input:
		A: an (`m` x `n`)-matrix of non-negative integers
	Output: 
		res: an (`m+1` x `n+1`)-matrix, which is A expanded by a zero-column to the left and a one-row on top
	"""
	if zeros:
		res = np.zeros((A.shape[0]+1, A.shape[1]+1), dtype=np.int)
		res[0,:] = np.ones(A.shape[1]+1, dtype=np.int)
		res[1:,1:] = A
	else:
		res = np.concatenate((np.ones((1,A.shape[1]), dtype=np.int), A), axis = 0)
	return res

def create_poly(n, degree, terms, A = None, hull_size = None, seed = None, negative = False, inner = 0):
	"""Create a multivariate polynomial in sparse notation.

	Call:
		A, b = create_poly(n, degree, terms[, A])
	Input:
		n: non-negative integer
		degree: non-negative integer
		terms: non-negative integer
		A (optional, default None): (`n` x `terms`)-array of integers
	Output:
		A: (`n+1` x `terms`)-array of integers, each column representing an exponent of the polynomial
		b: array of length `terms`, each element representing a coefficient of the poylnomial
	
	If no matrix is given, then a new one is created, otherwise the one given is used.
	It is reordered, such that the first columns form the convex hull.
	For these, the coefficients are positive, the other coefficients are negative.
	"""
	if aux.VERBOSE >= 2:
		print('Creating polynomial:\n\tn:\t%d\n\td:\t%d\n\tt:\t%s\n\tinner:\t%d' % (n,degree,terms,inner))
	if seed is not None:
		np.random.seed(seed)
		random.seed(seed)
	if A is None:
		A = np.zeros((n + 1, terms), dtype = np.int)
		A[0,:] = np.ones(terms, dtype = np.int)
		A[1:,1:terms - inner] = 2*_create_exponent_matrix(n, degree//2, terms - 1 - inner, seed)
		points = A[:,:terms - inner - 1].T.tolist()

		hull_vertices = convex_hull(A[:,:terms - inner])
		hull_size = len(hull_vertices)

		max_count = inner**2
		count = 1
		for _ in range(max_count):
			lamb = np.random.rand(terms - inner)
			v = np.array(np.dot(A[:,:terms - inner], lamb / lamb.sum()).round(), dtype = np.int)
			if is_in_convex_hull_cvxpy((A[:,hull_vertices],v)) and list(v) not in points:
				A[:,-count] = v
				points.append(list(v))
				count += 1
			if count > inner: break
		if count <= inner:
			#number_interior = polytope.number_interior_points(A, strict = False)
			#if number_interior < 8192: #random threshold
			#	candidates = polytope.interior_points(A, strict = False)
			#	A[:,terms - inner:] = np.array(random.sample(candidates, 5), dtype = np.int).T
			#else:
			#	raise Exception('Could not find enough interior points')
			raise Exception('Could not find enough interior points')

		#reorder exponent matrix
		A = np.concatenate((A[:,hull_vertices], A[:,[i for i in range(A.shape[1]) if i not in hull_vertices]]), axis = 1)

	if hull_size is None:
		#determine the vertices of the hull
		hull_vertices = convex_hull(A)
		hull_size = len(hull_vertices)

	#create coefficients, positive for the hull, negative in the interior
	#scale coefficients for the hull to avoid unreasonably large optima (like 1e30)
	positive_factor = terms / n
	if negative:
		coeff = np.concatenate((positive_factor*abs(np.random.normal(size = hull_size)), -abs(np.random.normal(size = terms - hull_size))))
	else:
		coeff = np.concatenate((positive_factor*abs(np.random.normal(size = hull_size)), np.random.normal(size = terms - hull_size)))
	return A, coeff

def create_standard_simplex_polynomial(n,degree,terms,seed = None, negative = False):
	"""Create a multivariate polynomial in sparse notation, whose convex hull is the standard simplex of "edge length = degree".

	Call:
		A,b = create_standard_simplex_polynomial(n, degree, terms[, seed])
	Input:
		n: non-negative integer, number of variables
		degree: non-negative integer, degree of the result
		terms: non-negative integer, number of terms
		seed (optional, default None): seed for the random number generator
	Output:
		A: (`n+1` x `terms`)-array of integers, each column representing an exponent of the polynomial
		b: array of length `terms`, each element representing a coefficient of the poylnomial
	
	The result is None, if the parameters do not allow a valid SONC-instance, e.g.
		- terms <= n+1 (no simplex)
		- degree <= n (no interior point)
		- degree odd (can always become negative)
	"""
	if terms <= n + 1 or degree <= n or degree % 2 != 0:
		return None
	if seed is not None:
		np.random.seed(seed)
		random.seed(seed)
	A = _create_exponent_matrix(n, degree - 1 - n, terms - 1 - n)
	A = A + np.ones(A.shape)
	A = np.concatenate((degree * np.eye(n),A),axis = 1)
	A = make_affine(A)
	return create_poly(n,degree,terms,A, hull_size = n + 1, negative = negative)

def create_simplex_polynomial(n,degree,terms,seed = None, negative = False):
	"""Create a multivariate polynomial in sparse notation, whose convex hull is a simplex.

	Note: Here we get an arbitrary simplex, while in create_standard_simplex_polynomial() we have the scaled standard simplex.

	Call:
		A,b = create_simplex_polynomial(n, degree, terms[, seed])
	Input:
		n: non-negative integer, number of variables
		degree: non-negative integer, degree of the result
		terms: non-negative integer, number of terms
		seed (optional, default None): seed for the random number generator
	Output:
		A: (`n+1` x `terms`)-array of integers, each column representing an exponent of the polynomial
		b: array of length `terms`, each element representing a coefficient of the poylnomial
	
	The result is None, if the parameters do not allow a valid SONC-instance, e.g.
		- terms <= n+1 (no simplex)
		- degree <= n (no interior point)
		- degree odd (can always become negative)
	"""
	if terms <= n + 1 or degree <= n or degree % 2 != 0:
		return None
	if seed is not None:
		np.random.seed(seed)
		random.seed(seed)
	d = degree // 2

	max_attempts = 500
	for _ in range(max_attempts):
		A = make_affine(2*_create_exponent_matrix(n, d, n))
		if np.linalg.matrix_rank(A) <= n:
			continue
		if abs(np.linalg.det(A)) < terms: continue

		inner = []
		Q,R = np.linalg.qr(A)
		for _ in range(terms**2):
			lamb = np.random.rand(n + 1)
			v = np.array((np.dot(A, lamb) / lamb.sum()).round(), dtype = np.int)
			new_lamb = scipy.linalg.solve_triangular(R, np.dot(Q.T,v))
			if all(new_lamb > aux.EPSILON) and all(new_lamb < 1 - aux.EPSILON) and list(v) not in inner:
				inner.append(list(v))
			if len(inner) + n + 1 == terms: break
		if len(inner) + n + 1 < terms: continue

		res = np.zeros((n + 1, terms), dtype = np.int)
		for i in range(n + 1, terms):
			res[:,i] = np.array(inner[i - (n+1)])
		res[:,:n+1] = A
		return create_poly(n, degree, terms, res, hull_size = n+1, negative  = negative)
	return None

if __name__ == "__main__":
	#main()
	#A, b = create_poly(4,8,10,seed = 2) # degenerate case

	n = 3
	degree = 20
	terms = 20
	seed = 0

	t0 = datetime.now()
	A,b = create_poly(n,degree,terms,seed = seed)

	print("time: %.2f seconds" % dt2sec(datetime.now() - t0))
