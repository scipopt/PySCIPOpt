#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Class for multivariate polynomials in sparse notation, focus on optimisation."""

import numpy as np
import sqlite3
import sympy
import scipy.optimize
import warnings
import re
import random
from datetime import datetime
import itertools
import os

import json_tricks as json
import cvxpy as cvx
from tabulate import tabulate
import sparse
try:
	import matlab
	import matlab.engine
	matlab_found = True
except:
	matlab_found = False

import aux
from aux import binomial, is_psd, linsolve
import polytope
from polytope import convex_hull
from generate_poly import create_standard_simplex_polynomial, create_simplex_polynomial, create_poly

x = sympy.Symbol('x')
sympy.init_printing();
np.set_printoptions(linewidth = 200)

class Polynomial(object):
	"""Class for multivariate polynomials in sparse notation, focus on optimisation."""

	# === Creating the object ===

	def __init__(self, *args, **kwargs):
		"""Create a new multivariate polynomial object for optimisation.

		Call:
			p = Polynomial(A, b)
			p = Polynomial(s)
			p = Polynomial(shape, variables, degree, terms[, inner])
			p = Polynomial(nr)
		Input:
			There are different possible inputs:
			---
			A - (n x t)-matrix or list of lists, representiong the exponents
			b - array-like of length t
			---
			s - string, which represents the polynomial, variables as 'x0' or 'x(0)'
			---
			shape - string, describes Newton polytope, can be 'simplex'/'standard_simplex'/'general'
			variables - int, maximal number of variables
			degree - int, maximal degree
			terms - int, number of terms
			inner [optional, default 0] - minimal number of interior points
			---
			nr - number, which tells the rowid of the database
			---

		Additional keywords
			seed [default None] - seed for the random number generator
			dirty [default True] - flag, whether the input is in an unclean state
				USE ONLY IF YOU KNOW WHAT YOU ARE DOING.
			matlab_instance [default newly created] - bridge to matlab, to avoid starting multiple instances
		"""
		if aux.VERBOSE > 2:
			print('number of args: %s' % len(args))
			for arg in args: print(arg)
		# -- setting some default values, so they are defined
		self.degenerate_points = None
		self.monomial_squares = None
		self.matlab = None
		self.cover_time = 0

		# -- initialise parameters from keywords --
		if 'dirty' in kwargs.keys():
			self.dirty = kwargs['dirty']
		else:
			self.dirty = True

		if 'matlab_instance' in kwargs.keys():
			self.matlab = kwargs['matlab_instance']
			self.matlab_start_time = 0

		if 'seed' in kwargs.keys():
			np.random.seed(kwargs['seed'])
			random.seed(kwargs['seed'])

		# -- distinguish possible input cases --
		if len(args) == 2:
			if (type(args[0]) == np.ndarray or type(args[0]) == list and all([type(l) == list for l in args[0]])) and type(args[1]) in [np.ndarray, list]:
				self.__read_matrix_vector(*args)
			else:
				raise Exception('Two inputs must be of type (A::matrix, b::vector).')
		elif len(args) == 1:
			if type(args[0]) == int:
				if args[0] < 0:
					raise Exception('Rowid of database must be non-negative integer.')
				self.__read_from_database(args[0])
			elif type(args[0]) == str:
				self.__read_string(args[0])
			else:
				raise Exception('Single input must be int (rowid of data base) or string.')
		elif len(args) in [4,5]:
			if type(args[0]) == str and all([type(e) == int for e in args[1:]]):
				self.__create_from_parameters(*args)
			else:
				raise Exception('Input with 4 or 5 parameters must be (shape::string, n::int, d::int, t::int[, inner::int]).')
		else:
			raise Exception('Wrong number of inputs.')

		# -- compute additional information -- 
		t0 = datetime.now()

		self._degree = max([sum(self.A[1:,i]) for i in range(self.A.shape[1])])

		#set further defaults
		self.cover = None
		self.solution = None
		self.old_solutions = {}
		self.old_covers = {}

		self.prob_sos = None 
		self.prob_sonc = None 

		self.init_time = aux.dt2sec(datetime.now() - t0)

	def __create_from_parameters(self, shape, n, d, t, inner = 0):
		"""Create instance with the given parameters.

		Call:
			p.__create_from_parameters(shape, n, d, t[, inner])
		Input:
			shape - string, describes Newton polytope, can be 'simplex'/'standard_simplex'/'general'
			n - int, maximal number of variables
			d - int, maximal degree
			t - int, number of terms
			inner [optional, default 0] - minimal number of interior points
		"""
		if aux.VERBOSE >= 2:
			print('Creating polynomial:\n\tshape:\t%s\n\tn:\t%d\n\td:\t%d\n\tt:\t%s\n\tinner:\t%d' % (shape,n,d,t,inner))
		##TODO: test, whether we can set these cases to 'clean'
		#self.dirty = False
		if shape == 'standard_simplex':
			self.degenerate_points = []
			self.__read_matrix_vector(*create_standard_simplex_polynomial(n,d,t))
			self.hull_size = n+1
		elif shape == 'simplex':
			self.degenerate_points = []
			self.__read_matrix_vector(*create_simplex_polynomial(n,d,t))
			self.hull_size = n+1
		elif shape == 'general':
			self.__read_matrix_vector(*create_poly(n,d,t, inner = inner))
		else:
			raise Exception('Unknown shape, possible: standard_simplex, simplex, general')

	def __read_matrix_vector(self, A, b):
		A = np.array(A)
		self.b = np.array(b)
		if all(A[0,:] == 1):
			self.A = np.array(A)
		else:
			self.A = np.concatenate((np.ones((1,A.shape[1]), dtype = np.int),A), axis = 0)
		if self.A.shape[1] != self.b.shape[0]:
			raise Exception('Dimensions of A and b do not match.')

	def __read_from_database(self, rowid, database = aux.SAVE_PATH + aux.DB_NAME):
		"""Obtain the polynomial (in sparse notation) with given row-id from the data base.

		Call:
			p.__read_from_database(rowid[, database])
		Input:
			rowid: non-negative integer
			database [optional, default given in aux.py]: location of the sqlite3-database
		"""
		conn = sqlite3.connect(database)
		cursor = conn.cursor()
		cursor.execute('select json from polynomial where rowid = ?;', (rowid,))
		polynomial_string = cursor.fetchone()[0]
		conn.close()
		self.__read_string(polynomial_string)

	def __read_string(self, s, reduce_vars = False):
		"""Read a polynomial from a given string.

		This is the inverse to the function __str__.

		Call:
			p.__read_string(s[, reduce_vars])
		Input:
			s: string, as printed e.g. by Matlab or SymPy, need '*',
					powers can be written with '^' or '**'
			reduce_vars [optional, default False]: whether to eliminate zero-rows in the beginning
				will destroy polynomial arithmetic if used
		"""
		if s == '':
			warnings.warn('Creating polynomial from empty string.')
			self.__read_matrix_vector(np.array([[1],[0]]),[0])
			return
		if s =='0': 
			self.__read_matrix_vector(np.array([[1],[0]]),[0])
			return
		#get number of variables, '+1' for x0
		n = max([int(i) for i in re.findall(r'x\(?([0-9]+)\)?', s)]) + 1
		if reduce_vars:
			n_min = min([int(i) for i in re.findall(r'x\(?([0-9]+)\)?', s)])
		else:
			n_min = 0

		#transform into some standard form
		pattern = re.compile(r'([^e])-')
		terms = pattern.sub(r'\1+-', s.replace(' ','')).replace('**','^').split('+')
		if not terms[0].find('x') == -1:
		    terms.insert(0,'0.0')
		t = len(terms)
		self.A = np.zeros((n + 1 - n_min, t), dtype = np.int)
		self.A[0,:] = np.ones(t, dtype = np.int)
		self.b = np.ones(t)
		for i in range(t):
			term = terms[i].split('*')
			#get the coefficient
			if term[0].find('x') == -1:
				self.b[i] = float(term[0])
				term = term[1:]
			elif term[0][0] == '-':
				self.b[i] = -1
			for var in term:
				if var.find('^') == -1:
					entry = (re.findall(r'x\(?([0-9]+)\)?', var)[0],1)
				else:
					entry = re.findall(r'x\(?([0-9]+)\)?\^([0-9]+)', var)[0]

				self.A[int(entry[0]) + 1 - n_min, i] += int(entry[1])
	
	def _add_zero(self):
		"""Add zero exponent with coefficient zero."""
		self.A = np.concatenate((np.zeros((self.A.shape[0],1), dtype = np.int), self.A), axis = 1)
		self.A[0,0] = 1
		self.b = np.concatenate(([0], self.b))

	# === Output === 

	def __dict__(self):
		"""Return polynomial as dictionary of type { exponent: coefficient }."""
		return {tuple(self.A[:,i]) : self.b[i] for i in range(self.A.shape[1])}

	def copy(self):
		"""Return a copy of itself."""
		return Polynomial(self.A, self.b)

	def __str__(self):
		"""Return the polynomial as string."""
		A = self.A[1:,1:]
		rows,cols = A.shape
		return ' + '.join([str(self.b[0])] + [str(self.b[j + 1]) + '*' + '*'.join(['x' + str(i) + '^' + str(A[i,j]) for i in range(rows) if A[i,j] != 0]) for j in range(cols) if self.b[j + 1] != 0]).replace('+ -','- ')

	def tex(self):
		"""Return the polynomial as string for LaTeX."""
		A = self.A[1:,:]
		rows,cols = A.shape
		return (' + '.join([str(self.b[j]) + '\cdot ' + ' '.join(['x_' + str(i) + '^{' + str(A[i,j]) + '}' for i in range(rows) if A[i,j] != 0]) for j in range(cols)])).replace('+ -','- ')

	def pip(self):
		"""Return the polynomial in PIP-format."""
		bounds = '\n'.join(['\t-inf <= x%d <= inf' % i for i in range(self.A.shape[1])])
		return 'Maximize\n\tobj: %s\nBound\n%s\nEnd' % (str(self).replace('*',' '), bounds)

	def symbolic(self):
		"""Return the polynomial as symbolic expression in sympy."""
		A = self.A[1:,:]
		rows,cols = A.shape
		return sum([self.b[j] * sympy.prod([x(i) ** A[i,j] for i in range(rows)]) for j in range(cols)])

	def get_solutions(self):
		"""Return a list of (solver, time, optimum) for all solutions found."""
		return [(key, self.old_solutions[key]['time'], self.old_solutions[key]['opt']) for key in self.old_solutions.keys()]

	def print_solutions(self, form = 'grid', only_valid = False):
		"""Print a table of all stored solutions.
	
		Call:
			p.print_solutions([only_valid, form])
		Input:
			only_valid [boolean, default False]: flag, whether to print only verified solutions
				i.e. those with <solution>['verify'] == 1
			form [string, default 'grid'] - tableformat for tabulate
		"""
		print(tabulate([list(key)[:-1] + [self.old_solutions[key][k] for k in ['time','opt','verify']] for key in self.old_solutions.keys() if (not only_valid) or (self.old_solutions[key]['verify'] == 1)],['language','strategy','solver','time','opt', 'verify'], tablefmt = form))

	def relax(self):
		"""Return a lower estimate for the polynomial.
		
		All potentially newgative terms are made negative at once.
		This function should be used, when chekcing the decomposition, obtained by get_decomposition(), since that functions works on the relaxation.
		"""
		return Polynomial(self.A.copy(), np.concatenate((self.b[:self.monomial_squares],-abs(self.b[self.monomial_squares:]))), degenerate_points = self.degenerate_points)

	# === Arithmetic ===
	def clean(self):
		"""Bring polynomial into clean state."""
		if self.dirty:
			self._normalise(zero = True)

	def _normalise(self, zero = False):
		"""Bring polynomial into normal form.

		Eliminate multiple occurrences of same monomials and zero coefficients.
		Rearrange order of summands.

		Call:
			p._normalise([zero])
		Parameters:
			zero [optional, default False]: flag, whether to ensure, that the first column of A is the zero-entry

		Ensure that summands appear in the following order:
		- zero (if zero = True)
		- other monomial squares
		- remaining terms (not monomial squares)
		"""
		##TODO:may leave the first part, if we know there are no double exponents
		#collect all same exponents
		base = {tuple(self.A[:,i]) : 0 for i in range(self.A.shape[1])}
		for i in range(self.A.shape[1]):
			base[tuple(self.A[:,i])] += self.b[i]
		#keep only non-zero terms, and possibly constant term
		base = { key : base[key] for key in base.keys() if base[key] != 0 or (all([entry == 0 for entry in key[1:]]) and zero) }
		l = list(base.keys())
		l.sort()
		self.A = np.zeros((self.A.shape[0],len(l)), dtype = np.int)
		self.b = np.zeros(len(l))
		for i in range(len(l)):
			self.A[:,i] = l[i]
			self.b[i] = base[l[i]]

		#put zero column to front
		if self.A[1:,0].any():
			zero_index = -1
			for i in range(self.A.shape[1]):
				if not self.A[1:,i].any():
					zero_index = i
					break
			if zero_index == -1:
				if zero:
					self._add_zero()
				self.has_zero = zero
			else:	#swap
				self.A[:,[0,zero_index]] = self.A[:,[zero_index,0]]
				self.b[[0,zero_index]] = self.b[[zero_index,0]]
				self.has_zero = True
		else:
			self.has_zero = True

		#find indices of monomial squares
		square_index = [i for i in range(self.A.shape[1]) if (self.b[i] > 0 and not (self.A[1:,i] % 2).any()) or not self.A[1:,i].any()]
		self.monomial_squares = len(square_index)
		#reorder the entries
		self.A[:,:self.monomial_squares], self.A[:,self.monomial_squares:] = self.A[:,square_index], self.A[:,[i for i in range(self.A.shape[1]) if i not in square_index]]
		self.b[:self.monomial_squares], self.b[self.monomial_squares:] = self.b[square_index], self.b[[i for i in range(self.A.shape[1]) if i not in square_index]]

		self.dirty = False
	
	def __add__(self,other):
		"""Return the sum of this polynomial with another one."""
		res = Polynomial(np.concatenate((self.A,other.A), axis = 1) ,np.concatenate((self.b, other.b), axis = 0))
		res._normalise()
		return res

	def __neg__(self):
		"""Return the negation of this polynomial."""
		res = self.copy()
		res.b = -res.b
		return res

	def __sub__(self, other):
		"""Return the difference between this polynomial and another one."""
		res = self.copy()
		return res.__add__(-other)

	def __call__(self,*x, dtype = 'float'):
		"""Evaluate the polynomial at point x."""
		#TODO: adjust docstring
		A = self.A[1:]
		if len(x) > A.shape[0]:
			x = x[:A.shape[0]]
		if len(x) < A.shape[0]:
			x = np.concatenate((x, np.zeros(A.shape[0] - len(x), dtype = np.int)))
		if dtype == 'float':
			return np.dot(np.prod(np.power(x,A.T),axis = 1),self.b)
		if dtype == 'int':
			b = [int(self.b[i]) for i in range(len(self.b))]
			res = 0
			for i in range(A.shape[1]):
				summand = b[i]
				for j in range(A.shape[0]):
					summand *= x[j]**A[j,i]
				res += summand
			return res
		raise Exception('dtype not understood; use \'float\' or \'int\'')

	def derive(self, index):
		#catch case, where index is higher than any occurring or variable does not occur
		if index >= self.A.shape[0] - 1 or not self.A[index + 1,:].any():
			return Polynomial('0')
		#multiply coefficients, subtract 1 from row, and clean the result
		A = self.A[1:,:].copy()
		b = self.b * A[index,:]
		A[index,:] -= 1
		res = Polynomial(A,b)
		res.clean()
		return res

	def prime(self, variables = None):
		if variables is None:
			variables = self.A.shape[0] - 1
		return tuple(self.derive(i) for i in range(variables))

	def __eq__(self, other):
		#TODO: may return false negatives
		self.clean()
		other.clean()
		return np.array_equal(self.A, other.A) and np.array_equal(self.b, other.b)

	# === Formulating the problems ===

	def _create_sos_opt_problem(self):
		"""Create the SOS-optimisation-problem in cvx for the polynomial given by (A,b).

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Note: This function does NOT call a solver. It only states the problem and does not solve it.

		Call:
			p._create_sos_opt_problem()
		Creates:
			p.prob_sos: cvx.Problem-instance
		"""
		t0 = datetime.now()
		#modify input and init variables
		A = self.A[1:,:]

		#support = np.array(polytope.interior_points(p.A, strict = False))[:,1:]
		#half_support = [tuple(v // 2) for v in support if not (v % 2).any()]
		#C = cvx.Variable((len(half_support),len(half_support)), PSD = True)
		#coeffs = {tuple(e): 0 for e in support}
		#for i in range(p.A.shape[1]):
		#    coeffs[tuple(p.A[1:,i])] += p.b[i]
		#lookup = {half_support[i] : i for i in range(len(half_support))}
		#constraints = []
		#for v,c in coeffs.items():
		#    if not any(v):
		#        constraints.append(C[0,0] == coeffs[v] + gamma)
		#        continue
		#    l = []
		#    for u in half_support:
		#        diff = tuple(v[i] - u[i] for i in range(len(v)))
		#        if diff in half_support:
		#            l.append((lookup[u],lookup[diff]))
		#    if l == []: break
		#    constraints.append(sum([C[i,j] for i,j in l]) == c)		


		n = A.shape[0]
		d = self._degree // 2
		size = binomial(n + 2*d, n)

		#create complete list of monomials, all coefficients initialised with 0
		coeffs = [(list(aux._index_to_vector(i,n,2*d)),0) for i in range(size)]
		#setting the coefficients occurring in A
		for i in range(A.shape[1]):
			index = aux._vector_to_index(A[:,i], 2*d)
			coeffs[index] = (coeffs[index][0],coeffs[index][1] + self.b[i])
		#declare semidefinite matrix C, aim: Z^T * C * Z = p where Z is vector of all monomials
		C = cvx.Variable((binomial(n + d, n), binomial(n + d, n)), PSD = True)
		gamma = cvx.Variable()

		#construct the constraints
		constraints = [C[0,0] == coeffs[0][1] + gamma]

		for alpha,c in coeffs[1:]:
			l = [np.array(beta) for beta in aux._smaller_vectors(alpha, sum(alpha)-d, d)]
			if aux.VERBOSE >= 2:
				eq = '%s: ' % alpha
				eq += ' + '.join(['C[%d,%d]' % (aux._vector_to_index(beta,d), aux._vector_to_index(alpha - beta, d)) for beta in l])
			constraints.append(sum([C[aux._vector_to_index(beta,d),aux._vector_to_index(alpha - beta, d)] for beta in l]) == c)
			if aux.VERBOSE >= 2:
				print(eq + ' == %f,' % c)
		
		#define the problem
		self.prob_sos = cvx.Problem(cvx.Minimize(gamma),constraints)

		self.sos_problem_creation_time = aux.dt2sec(datetime.now() - t0)

	def _create_sonc_opt_problem(self, B = None, split = True):
		"""Create the SONC-optimisation-problem in cvx for the polynomial given by (A,b).

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of non-negative circuit polynomials..
		
		Note: This function is for the general case.
			If the Newton polytope is a simplex, use p._create_sonc_problem().
		Note: This function does NOT call a solver. It only states the problem and does not solve it.
		Note: To obtain a DCP-problem in cvx, log was applied to every entry. 
			To get the proper solution this has to be mapped back.

		Call:
			p._create_sonc_opt_problem()
		Creates:
			p.prob_sonc: cvx.Problem-instance
		"""
		self.clean()
		if self.cover is None:
			self._compute_zero_cover(split)
		
		t0 = datetime.now()

		#default: evenly distribute the non-squares
		self._set_coefficient_distribution(B)

		X = cvx.Variable((len(self.cover), self.monomial_squares))

		constraints = []
		for i in range(1,self.monomial_squares):
			indices = [k for k in range(len(self.cover)) if i in self.cover[k]]
			if indices != []:
				constraints.append(cvx.log_sum_exp(X[indices, i]) <= cvx.log(self.b[i]))
		for k in range(len(self.cover)):
			lamb = self.lamb[k,self.cover[k][:-1]]
			constraints.append(cvx.log(-self.coefficient_distribution[k, self.cover[k][-1] - self.monomial_squares]) == cvx.sum(cvx.multiply(lamb, X[k, self.cover[k][:-1]]) - cvx.multiply(lamb, cvx.log(lamb))))

		if any([0 in c for c in self.cover]):
			objective = cvx.Minimize(cvx.log_sum_exp(X[[k for k in range(len(self.cover)) if 0 in self.cover[k]],0]))
		else:
			objective = cvx.Minimize(0)
		self.prob_sonc = cvx.Problem(objective, constraints)

		self.sonc_problem_creation_time = aux.dt2sec(datetime.now() - t0) + self.cover_time

	def _set_coefficient_distribution(self, B = None):
		if B is None:
			count = np.zeros(self.A.shape[1] - self.monomial_squares, dtype = np.int)
			for t in self.cover:
				for entry in t:
					if entry >= self.monomial_squares:
						count[entry - self.monomial_squares] += 1

			b_relax = -abs(self.b[self.monomial_squares:]) / count
			B = scipy.sparse.dok_matrix((len(self.cover), self.A.shape[1] - self.monomial_squares))
			for k in range(len(self.cover)):
				B[k, self.cover[k][-1] - self.monomial_squares] = b_relax[self.cover[k][-1] - self.monomial_squares]

		self.coefficient_distribution = B.copy()

	def _reallocate_coefficients(self):
		"""Given a solution, this function computes an improved distribution of the negative coefficients among the simplex polynomials.

		Note: This makes sense only for the non-simplex case.

		Call:
			B = p._reallocate_coefficients()
		Output:
			B - sparse matrix, where B[k,j] denoted how much of p.b[j + monomial_squares] goes into the k-th simplex polynomial.
		"""
		if self.solution is None:
			return

		#init
		try:
			C = self.solution['C'].todense()
		except:
			C = self.solution['C']
		circ = np.zeros(len(self.cover))

		cover_indices = [k for k in range(len(self.cover)) if 0 in self.cover[k]]
		
		#compute convex combinations and something-like-circuit-number
		for k in cover_indices:
			circ[k] = ((C[k,self.cover[k][1:-1]]/self.lamb[k,self.cover[k][1:-1]]) ** (self.lamb[k,self.cover[k][1:-1]] / (1 - self.lamb[k,0]))).prod()

		#compute common derivative of the b[:, j + monomial_square]
		diff = self.A.shape[1] - self.monomial_squares
		const = np.zeros(diff)
		for j in range(diff):
			relevant_indices = [k for k in cover_indices if j + self.monomial_squares in self.cover[k]]
			if len(relevant_indices) <= 1: continue
			f = (lambda a: np.sum([a**(self.lamb[k,0]/(1 - self.lamb[k,0])) * circ[k] for k in relevant_indices]) - abs(self.b[j + self.monomial_squares]))
			upper = max([(abs(self.b[j + self.monomial_squares]) / circ[k]) ** ((1 - self.lamb[k,0])/ self.lamb[k,0]) for k in relevant_indices]) + 1
			try:
				const[j] = scipy.optimize.brentq(f, 0, upper)
			except:
				const[j] = scipy.optimize.bisect(f, 0, upper)

		#compute output
		B = scipy.sparse.dok_matrix((len(self.cover), diff))
		for k in cover_indices:
			j = self.cover[k][-1] - self.monomial_squares
			if const[j] == 0:
				#in this case the above computation was not executed, but j in cover[k], so this is its only occurrence
				B[k,j] = -abs(self.b[j + self.monomial_squares])
			else:
				B[k,j] = - const[j] ** (self.lamb[k,0]/(1 - self.lamb[k,0])) * circ[k]

		#return scipy.sparse.coo_matrix(B)
		return B
	
	##TODO: idea can still be used, with some adjustments
	#def improve_sonc(self):
	#	"""Reduce numerical errors in the solution for the SONC-problem.

	#	Usually the solvers return a solution which violates the constraints.
	#	But due to the nature of the SONC-problem it is very easy to obtain a feasible solution.

	#	Note: This only update the variables (and thus the violation of the constraints).
	#		It does NOT update the optimum self.prob_sonc.value.

	#	Call:
	#		p.improve_sonc()
	#	"""
	#	b_relax = np.concatenate((self.b[:self.hull_size], -abs(self.b[self.hull_size:])))
	#	new_var = np.array(self.prob_sonc.variables()[0].value.copy())
	#	for i in range(1,self.hull_size):
	#		violation = self.prob_sonc.constraints[i-1].violation()
	#		if violation > 0:
	#			new_var[i,:] -= violation

	#	lamb = linsolve(self.A[:,:self.hull_size], self.A[:,self.hull_size:])
	#	diff = self.A.shape[1] - self.hull_size
	#	for j in range(diff):
	#		new_var[0,j] = np.log(lamb[0,j]) + (np.log(-b_relax[self.hull_size+j]) - np.sum((new_var[1:,j] - np.log(lamb[1:,j])) * lamb[1:,j])) / lamb[0,j]
	#	self.prob_sonc.variables()[0].value = new_var
	#	##cannot set the value
	#	#self.prob_sonc.value = np.sum(np.exp(self.prob_sonc.variables()[0].value[0,:]))

	# === Calling SOS solver ===

	def sos_opt_python(self, solver = 'CVXOPT', **solverargs):
		"""Optimise the polynomial given by (A,b) via SOS using cvx.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Note: SCS randomly runs VERY long on trivial instances. Usage is possible, but discouraged.

		Call:
			data = p.sos_opt_python(A,b,[solver],**solverargs)
		Input:
			solver [optional, default 'CVXOPT']: solver, to solve the problem, currenty possible: 'CVXOPT', 'MOSEK', 'SCS'
			solverargs: dictionary of keywords, handed to the solver
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: psd-matrix such that p = Z^T * C * Z, where Z is the vector of all monomials
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		if self.prob_sos is None: self._create_sos_opt_problem()

		t0 = datetime.now()

		if not 'verbose' in solverargs.keys(): solverargs['verbose'] = aux.VERBOSE
		try:
			if solver == 'SCS':
				#setting some defaults for SCS
				kwargs = {'eps': aux.EPSILON / 10, 'max_iters' : 20000}
				kwargs.update(solverargs)
				opt = self.prob_sos.solve(solver = solver, **kwargs)
			else:
				opt = self.prob_sos.solve(solver = solver, **solverargs)
			C = self.prob_sos.variables()[1].value
			#test, whether solution satisfies constraints
			if all([c.violation() <= aux.EPSILON for c in self.prob_sos.constraints]) and is_psd(C):
				verify = 1
			else:
				verify = -1
			data = {'opt': opt, 'C': C, 'status': self.prob_sos.status, 'verify': verify}
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)
		try:
			data['solver_time'] = self.prob_sos.solver_stats.solve_time + self.prob_sos.solver_stats.setup_time
		except:
			data['solver_time'] = 0

		self.solution_time = aux.dt2sec(datetime.now() - t0)

		data['time'] = self.solution_time
		data['language'] = 'python'
		data['solver'] = solver
		data['strategy'] = 'sos'
		data['params'] = solverargs
		self._store_solution(data)
		return data

	def sos_opt_matlab(self, solver = 'sedumi'):
		"""Optimise the polynomial given by (A,b) via SOS using cvx in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Call:
			data = p.sos_opt_matlab([solver])
		Input:
			solver [optional, default 'sedumi']: solver, to solve the problem, currenty possible: 'sedumi', 'sdpt3'
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: psd-matrix such that p = Z^T * C * Z, where Z is the vector of all monomials
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		#python formulation needed for verification
		if self.prob_sos is None: self._create_sos_opt_problem()

		self._ensure_matlab()

		t0 = datetime.now()
		self.matlab.cvx_solver(solver)
		matlab_result = self.matlab.sos_cvx(matlab.double(self.A.tolist()), matlab.double(self.b.tolist()), nargout = 5)
		try:
			data = {}
			data['opt'], data['C'], data['solver_time'], data['status'], data['verify'] = matlab_result
			data['C'] = np.array(data['C'])
			data['verify'] = 0
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['result'] = matlab_result
			data['error'] = repr(err)
		self.solution_time = aux.dt2sec(datetime.now() - t0)
		
		data['time'] = self.solution_time
		data['language'] = 'matlab'
		data['solver'] = solver
		data['strategy'] = 'sos'
		self._store_solution(data)
		return data

	def sostools_opt(self, sparse = False):
		"""Optimise the polynomial given by (A,b) via SOS using SOSTOOLS in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Call:
			data = p.sostools_opt()
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: psd-matrix such that p = Z^T * C * Z, where Z is the vector of all monomials
				- time: time to compute the solution
				- status: 1 = Solved, 0 = otherwise
		"""
		varlist = ['x' + str(i) for i in range(self.A.shape[0])]
		prog	= 'pvar ' + ' '.join(varlist) + ' gam;\n'
		prog += 'vartable = [' + ', '.join(varlist) + '];\n'
		prog += 'p = ' + str(self) + ';\n'
		prog += 'prog = sosprogram(vartable);\nprog = sosdecvar(prog,gam);\nprog = sosineq(prog,(p-gam)'
		if sparse:
			prog += ',\'sparse\''
		prog += ');\nprog = sossetobj(prog,-gam);\nprog = sossolve(prog);\nSOLgamma = -sosgetsol(prog,gam);\n'
		prog += 'opt = SOLgamma.coefficient(1,1) + 0;\ntime = prog.solinfo.info.cpusec;\nC = prog.solinfo.extravar.primal{1};\nnumerr = prog.solinfo.info.numerr;'

		self._ensure_matlab()

		t0 = datetime.now()

		try:
			self.matlab.evalc(prog)
			opt, time, C, numerr = (self.matlab.workspace[key] for key in ['opt','time','C','numerr'])

			if C is None: C = np.array([[]])
			status = int(1 - numerr)
			data = {'opt': opt, 'C': np.array(C), 'solver_time': time, 'status': status, 'verify': 0}
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)

		self.solution_time = aux.dt2sec(datetime.now() - t0)
		
		data['time'] = self.solution_time
		data['language'] = 'matlab'
		data['solver'] = 'sostools'
		data['strategy'] = 'sos'
		data['params'] = {'sparse': sparse}
		self._store_solution(data)
		return data

	def yalmip_opt(self):
		"""Optimise the polynomial given by (A,b) via SOS using YALMIP in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Call:
			data = p.yalmip_opt()
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: psd-matrix such that p = Z^T * C * Z, where Z is the vector of all monomials
				- time: time to compute the solution
				- status: 1 = Solved, 0 = otherwise
		"""
		varlist = ['x' + str(i) for i in range(self.A.shape[0])]
		prog	= 'sdpvar ' + ' '.join(varlist) + ' gam;\n'
		prog += 'p = ' + str(self) + ';\n'
		prog += 'prog = sos(p + gam)\n[sol,v,C] = solvesos(prog, gam, [], gam);\n'
		prog += 'opt = value(gam);\ntime = sol.yalmiptime + sol.solvertime;\nC = C{1};\nnumerr = sol.problem;'

		self._ensure_matlab()

		t0 = datetime.now()

		try:
			self.matlab.evalc(prog)
			opt, time, C, numerr = (self.matlab.workspace[key] for key in ['opt','time','C','numerr'])

			if C is None: C = np.array([[]])
			status = int(1 - numerr)
			data = {'opt': opt, 'C': np.array(C), 'solver_time': time, 'status': status, 'verify': 0}
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)

		self.solution_time = aux.dt2sec(datetime.now() - t0)
		
		data['time'] = self.solution_time
		data['language'] = 'matlab'
		data['solver'] = 'yalmip'
		data['strategy'] = 'sos'
		data['params'] = {}
		self._store_solution(data)
		return data

	def gloptipoly_opt(self):
		"""Optimise the polynomial given by (A,b) via SOS using globtipoly in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Call:
			data = p.gloptipoly_opt()
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: psd-matrix such that p = Z^T * C * Z, where Z is the vector of all monomials
				- time: time to compute the solution
				- status: 1 = Solved, 0 = otherwise
		"""
		rows,cols = self.A.shape
		varlist = ['x' + str(i) for i in range(rows)]
		prog	= 'mpol ' + ' '.join(varlist) + ' gam;\n'
		prog += 'p = ' + str(self) + ';\n'
		prog += '[status, opt_neg, M, dual, info] = msol(msdp(min(p)));\n'
		prog += 'm = sqrt(size(dual,1));\nC = reshape(dual,m,m);\ntime = info.cpusec;\nopt = -opt_neg;'

		self._ensure_matlab()

		t0 = datetime.now()

		try:
			self.matlab.evalc(prog)
			opt, time, C, status = (self.matlab.workspace[key] for key in ['opt','time','C','status'])
			data = {'opt': opt, 'C': np.array(C), 'solver_time': time, 'status': int(status), 'verify': 0}
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)

		self.solution_time = aux.dt2sec(datetime.now() - t0)

		data['time'] = self.solution_time
		data['language'] = 'matlab'
		data['solver'] = 'gloptipoly'
		data['strategy'] = 'sos'
		self._store_solution(data)
		return data

	# === calling SONC solver ===

	def sonc_opt_python(self, solver = 'ECOS', **solverargs):
		"""Optimise the polynomial given by (A,b) via SONC using cvx.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of non-negative circuit polynomials.

		Note: This function is for the general case.
			If the Newton polytope is a simplex, call p.sonc_opt_python_simplex().

		Call:
			data = p.sonc_opt_python([solver], **solverargs)
		Input:
			solver [optional, default 'ECOS']: solver, to solve the problem, currenty possible: 'ECOS', 'CVXOPT', 'SCS'
			solverargs: dictionary of keywords, handed to the solver
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: (m x (n-m))-matrix, coefficients for the decomposition
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		self.clean()
		#create problem instance
		if self.prob_sonc is None: 
			self._create_sonc_opt_problem(split = True)

		#parsing keywords
		if solver == 'SCS': 
			kwargs = {'eps': aux.EPSILON / 10, 'max_iters' : 20000}
		elif solver == 'ECOS': 
			kwargs = {'reltol': aux.EPSILON / 10, 'max_iters' : 500, 'feastol': aux.EPSILON / 10, 'abstol': aux.EPSILON / 10}
		else:
			kwargs = {}
		kwargs['verbose'] = (aux.VERBOSE > 0)
		kwargs.update(solverargs)

		#call the solver and handle the result
		t0 = datetime.now()
		try:
			self.prob_sonc.solve(solver = solver, **kwargs)

			#get exp-values, round for turning into sparse matrix
			Exp = np.exp(self.prob_sonc.variables()[0].value)
			
			#get coordinates and data for the coefficient matrix
			coords = []
			data = []
			for k in range(len(self.cover)):
				for i in range(len(self.cover[k]) - 1):
					coords.append((k, self.cover[k][i]))
					data.append(Exp[k, self.cover[k][i]])
			C = sparse.COO(np.array(coords).T,data,shape=(len(self.cover), self.monomial_squares))

			opt = np.sum(C[:,0]) - self.b[0]

			#test, whether solution satisfies constraints
			if all([c.violation() <= aux.EPSILON for c in self.prob_sonc.constraints]):
				verify = 1
			else:
				verify = -1
			data = {'opt': opt, 'C': C, 'solver_time': self.prob_sonc.solver_stats.solve_time + self.prob_sonc.solver_stats.setup_time, 'status': self.prob_sonc.status, 'verify': verify}
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)
			try:
				data['solver_time'] = self.prob_sonc.solver_stats.solve_time + self.prob_sonc.solver_stats.setup_time
			except:
				data['solver_time'] = 0

		self.solution_time = aux.dt2sec(datetime.now() - t0)

		solverargs['split'] = 'outer'
		data['time'] = self.solution_time
		data['language'] = 'python'
		data['solver'] = solver
		data['strategy'] = 'sonc'
		data['params'] = solverargs
		data['params']['cover'] = self.cover.copy()
		data['params']['distribution'] = self.coefficient_distribution.todense()
		self._store_solution(data)
		return data

	def trivial_solution(self):
		"""Compute a trivial solution to the SONC-problem.

		Let p be the polynomial given by (A,b). We want to quickly find a gamma such that p + gamma is SONC.

		Note: This function only works if the convex hull of the Newton polytope forms a simplex.

		Call:
			data = p.trivial_solution()
		Output:
			data: dictionary containing information about the solution
				- opt: feasible value
				- C: (m x (n-m))-matrix, coefficients for the decomposition
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		#TODO: make this work for non-degenerate case
		if not self.is_simplex:
			raise Exception('Trivial solution only in simplex case.')

		if self.prob_sonc is None: 
			self._create_sonc_opt_problem()

		t0 = datetime.now()

		self.prob_sonc.variables()[0].value = np.ones(self.prob_sonc.variables()[0].size)
		self.improve_sonc()
		C = np.exp(self.prob_sonc.variables()[0].value)
		opt = np.sum(C[0,:]) - self.b[0]
		if all([c.violation() <= aux.EPSILON for c in self.prob_sonc.constraints]):
			verify = 1
		else:
			verify = -1
		self.solution_time = aux.dt2sec(datetime.now() - t0)
		data = {'opt': opt, 'C': C, 'solver_time': self.solution_time, 'status': self.prob_sonc.status, 'verify': verify}

		data['time'] = self.solution_time
		data['language'] = 'python'
		data['solver'] = 'trivial'
		data['strategy'] = 'sonc'
		self._store_solution(data)
		return data

	# === General handling ===

	def _ensure_matlab(self):
		"""Create and start the Matlab engine and measure the required time."""
		if not matlab_found:
			raise Exception('Wanted to start Matlab, but Matlab engine not found.')
		if self.matlab is None:
			t1 = datetime.now()
			self.matlab = matlab.engine.start_matlab('-useStartupFolderPref -nosplash -nodesktop')
			self.matlab_start_time = aux.dt2sec(datetime.now() - t1)
		self.matlab.cd(os.getcwd())
		self.matlab.cd('../matlab')

	def _store_solution(self, data):
		"""Store new solution, but keep the previous ones for different methods."""
		if type(data['C']) == np.ndarray and len(data['C'].shape) == 1:
			data['C'] = np.array(np.matrix(data['C']).T)

		params = data['params'] if 'params' in data.keys() else {}

		data['status'] = aux.unify_status(data['status'])

		#summing up all computation times
		data['init_time'] = self.init_time
		data['time'] += self.init_time
		if data['language'] == 'python' and data['strategy'] != 'trivial':
			if data['strategy'] == 'sos': 
				data['problem_creation_time'] = self.sos_problem_creation_time
			elif data['strategy'] == 'sonc':
				data['problem_creation_time'] = self.sonc_problem_creation_time
			data['time'] += data['problem_creation_time']

		#Verify solution if verify = 0
		if data['verify'] == 0:
			try:
				if data['strategy'] == 'sos':
					if self.prob_sos is None: self._create_sos_opt_problem()

					self.prob_sos.variables()[1].value = data['C']
					self.prob_sos.variables()[0].value = data['opt']
					#test, whether solution satisfies constraints
					if all([c.violation() <= aux.EPSILON for c in self.prob_sos.constraints]) and is_psd(data['C']):
						data['verify'] = 1
					else:
						data['verify'] = -1
				elif data['strategy'] == 'sonc':
					warnings.warn('verfication for SONC-Matlab not implemented, Python is verified')
				#	if p.cover is None:
				#		if self.prob_sonc is None: self._create_sonc_opt_problem()
				#		self.prob_sonc.variables()[0].value = self.solution['C']
				#		if all([c.violation() <= aux.EPSILON for c in self.prob_sonc.constraints]):
				#			data['verify'] = 1
				#		else:
				#			data['verify'] = -1
				#	else:
				#		if self.prob_sonc is None: self._create_sonc_opt_problem_cover()


						#stuff
				else:
					raise Exception('Unknown strategy')
			except Exception as err:
				warnings.warn('Cannot verirfy solution: %s' % repr(err))

		self.solution = data

		key = (self.solution['language'], self.solution['strategy'], self.solution['solver'], json.dumps(params))
		if not key in self.old_solutions.keys():
			self.old_solutions[key] = self.solution.copy()

	def run_all(self, keep_alive = False):
		"""Run all optimisation methods which are currently implemented.

		The results are stored in self.old_solutions.
		SOS is called only if <= 400 monomials

		Current list of methods:
		- sostools
		- gloptipoly
		- sos-cvx in Matlab, using sedumi 
		- sos-cvx in Matlab, using sdpt3
		- sos-cvx in Python using CVXOPT (if matrix-size <= 120) 

		- sonc-cvx in Matlab, using sedumi
		- sonc-cvx in Matlab, using sdpt3
		- sonc-cvx in Python using ECOS

		If the Matkab engine was not found, only Python is run.
		"""
		#Handle trivial case of only monomial squares
		if self.monomial_squares == self.A.shape[1]:
			data = { 'time': 0, 'language': 'python', 'solver': 'trivial', 'strategy': 'trivial', 'status': 1, 'verify': 1, 'params': {}, 'C': np.array([]), 'opt': -self.b[0] }
			self._store_solution(data)
			return

		sos_size = binomial(self.A.shape[0] - 1 + self._degree//2, self._degree//2)
		sos_size_constraints = binomial(self.A.shape[0] - 1 + self._degree, self._degree)
		#saveguards for sos, where problem would crash RAM
		if sos_size <= 400 and matlab_found:
			self.sostools_opt()
			self.sostools_opt(sparse = True)
			self.gloptipoly_opt()
			self.yalmip_opt()
			if sos_size_constraints <= 8000:
				self.sos_opt_matlab()
				self.sos_opt_matlab(solver = 'sdpt3')
		#Python takes more RAM, so stricter saveguard
		if sos_size <= 120:
			self.sos_opt_python()
		self.opt()
		self.sonc_opt_python()
		if matlab_found:
			self.opt(method = 'even', language = 'matlab')
			if np.prod(self.A.shape) < 3000:
				self.opt(method = 'outer', language = 'matlab')
		if not keep_alive and matlab_found:
			self.matlab.exit()

	def _get_sos_decomposition(self):
		"""Return a certificate that the polynomial is SOS.

		Note: This function might fail for SOSTOOLS or Gloptipoly, since these do not always return a full-size matrix.
		
		Call:
			cert = p._get_sos_decomposition()
		Output:
			cert: a list of Polynomial, such that 
				sum([q**2 for q in cert]) == p (up to rounding)
		"""
		if self.solution is None: return None
		if self.solution['strategy'] != 'sos': return None

		C = self.solution['C']
		lambda0 = min(np.linalg.eigvalsh(C))
		if lambda0 < 0:
			C += np.eye(C.shape[0]) * (lambda0 + aux.EPSILON)
		L = np.linalg.cholesky(C)

		n = self.A.shape[0] - 1
		d = self._degree // 2
		size = binomial(n+d,d)

		A = np.array([aux._index_to_vector(i,n,d) for i in range(size)]).T
		A = np.concatenate((np.ones((1,size), dtype = np.int),A), axis = 0)
		return [Polynomial(A,L[:,i]) for i in range(size)]

	def get_decomposition(self):
		"""Return a decomposition into non-negative polynomials, according to currently stored solution."""
		if self.solution is None: return None
		if self.solution['strategy'] == 'sos':
			return self._get_sos_decomposition()
		if self.solution['strategy'] == 'sonc':
			cert = []
			for k in range(len(self.cover)):
				cert.append(Polynomial(self.A[:, self.cover[k]], np.concatenate((self.solution['C'][k, self.cover[k][:-1]], [self.coefficient_distribution[k, self.cover[k][-1] - self.monomial_squares]])), hull_size = len(self.cover[k]) - 1, degenerate_points = []))
			return cert
	
	def is_sum_of_monomial_squares(self, eps = 0):
		"""Check whether the polynomial is a sum of monomial squares.

		Call:
			res = p.is_sum_of_monomial_squares([strict])
		Input:
			strict [optional, default True]: boolean flag, whether the testing should be strict.
				If False, this allows errors up to aux.EPSILON in the coefficients.
		Output:
			res: bool, whether the polynomial is a sum of monomial squares
		"""
		idx = [i for i in range(len(self.b)) if abs(self.b[i]) > eps]
		A = self.A[1:, idx]
		b = self.b[idx]
		return not (A % 2).any() and (b > 0).all()

	# === Covering the polynomial ===

	def _compute_convex_hull(self):
		"""Compute the convex hull, store its size and reorder A and b."""
		hull_vertices = convex_hull(self.A)
		self.hull_size = len(hull_vertices)
		
		self.A = np.concatenate((self.A[:,hull_vertices], self.A[:,[i for i in range(self.A.shape[1]) if i not in hull_vertices]]), axis = 1)
		self.b = np.concatenate((self.b[hull_vertices], self.b[[i for i in range(len(self.b)) if i not in hull_vertices]]))

	def _compute_degenerate_points(self):
		"""Determine vertices which are not covered by 0 in the current cover.

		By default, using self._compute_zero_cover(), this gives all points which lie on an outer face non-adjacent to zero.
		"""		
		if aux.VERBOSE > 1:
			print('Computing degenerate points.')
		if self.cover is None:
			self._compute_zero_cover()
		self.degenerate_points = set([c[-1] for c in self.cover if c[0] != 0]) - set([c[-1] for c in self.cover if c[0] == 0])

		#if self.hull_size is None:
		#	self.compute_convex_hull()
		##setup LP
		##only test non-squares, restrict to combinations of vertices, to have smaller LP
		#A_eq = self.A[:,:self.hull_size]
		#self.degenerate_points = []
		##check each point, whether it is a convex combination using zero
		#for i in range(self.monomial_squares, self.A.shape[1]):
		#	lamb = cvx.Variable(self.hull_size)
		#	prob = cvx.Problem(cvx.Minimize(-lamb[0]), [A_eq*lamb == self.A[:,i], lamb >= 0])
		#	prob.solve()
		#	if prob.value > -aux.EPSILON:
		#		self.degenerate_points.append(i)

	def _compute_zero_cover(self, split = True):
		"""Compute a complete covering of A using simplices, if possible include 0.

		Call:
			p._compute_zero_cover()
		Result stored in:
			p.cover: list of coverings, each covering is a list of integers (as column indices), 
				such that these columns of A form a simplex with some interior nodes
				Each node, if possible, is covered using the origin.
		"""
		self.clean()

		t0 = datetime.now()
		#create short names
		n,t = self.A.shape
		m = self.monomial_squares
		#init
		U_index = {i for i in range(m,len(self.b))}
		T = []
		#setting up the LP
		c = [-1] + [0 for _ in range(m-1)]
		A_eq = self.A[:,:m]
		X = cvx.Variable(self.monomial_squares)

		#continue, until all points are covered
		while U_index != set():
			ui = U_index.pop()
			if aux.VERBOSE > 1:
				print('covering index %d' % ui)
			#find vertices covering u
			prob = cvx.Problem(cvx.Minimize(-X[0]), [X >= 0, A_eq * X == self.A[:,ui]])
			res = prob.solve(solver = 'GLPK')
			if res == np.inf:
				raise Exception('Polynomial is unbounded at point %d.' % ui)
			T_index = [i for i in range(m) if X.value[i] > 1e-15]
			#update target vector
			#get all points covered by T_index
			T_index = polytope._get_inner_points(self.A, range(m, t), T_index)
			T_index.sort()
			#mark covered points
			U_index -= set(T_index)
			T.append(T_index)
		self.cover_time = aux.dt2sec(datetime.now() - t0)
		self.set_cover(T, split)
	
	def _compute_cover(self, split = True):
		"""Compute a complete covering of A using simplices.

		Call:
			p._compute_cover()
		Result stored in:
			p.cover: list of coverings, each covering is a list of integers (as column indices), 
				such that these columns of A form a simplex with some interior nodes
				Also each node appears in some cover.
		"""
		self.clean()
		t0 = datetime.now()
		#create short names
		n,t = self.A.shape
		m = self.monomial_squares
		#init. V are the monomial squares
		#V_index = {i for i in range(t) if self.b[i] > 0 and not (self.A[1:,i] % 2).any()}
		V_index = set(range(m))
		U_index = set(range(t)) - V_index
		T = []
		#setting up the LP
		A_eq = self.A[:,:m]
		X = cvx.Variable(self.monomial_squares)
		c = np.ones(m)
		
		#cover U
		while U_index != set():
			ui = U_index.pop()
			if aux.VERBOSE > 1:
				print('covering point: ',u)
			#find vertices covering u
			prob = cvx.Problem(cvx.Minimize(-cvx.sum(cvx.multiply(c,X))), [X >= 0, A_eq * X == self.A[:,ui]])
			res = prob.solve(solver = 'GLPK')
			if res == np.inf:
				raise Exception('Polynomial is unbounded.')
			T_index = [i for i in range(m) if X.value[i] > aux.EPSILON**2]
			for i in T_index: c[i] = 0
			#get all points covered by T_index
			T_index = [ui] + polytope._get_inner_points(self.A, U_index, T_index)
			T_index.sort()
			#mark covered points
			U_index -= set(T_index)
			V_index -= set(T_index)
			T.append(T_index)
		#cover V
		U_index = {i for i in range(t) if self.b[i] < 0 or (self.A[1:,i] % 2).any()}
		while V_index != set():
			if aux.VERBOSE > 1:
				print('Still to use: ', V_index)
			change = False
			for ui in U_index:
				#find vertices covering u
				prob = cvx.Problem(cvx.Minimize(-cvx.sum(cvx.multiply(c,X))), [X >= 0, A_eq * X == self.A[:,ui]])
				prob.solve(solver = 'GLPK')
				if prob.value < -aux.EPSILON:
					T_index = [i for i in range(m) if X.value[i] > aux.EPSILON**2]
					for i in T_index: c[i] = 0
					if set(T_index) & V_index == set(): continue
					#get all points covered by T_index
					T_index = polytope._get_inner_points(self.A, U_index, T_index)
					T_index.sort()
					#mark covered points
					V_index -= set(T_index)
					T.append(T_index)
					change = True
					if V_index == set(): break
			if not change:
				#print('Unnecessary points: ', V_index)
				break
		self.cover_time = aux.dt2sec(datetime.now() - t0)
		self.set_cover(T, split)

	def set_cover(self, T, split = True):
		"""Set a new cover and store the previous one.

		Call:
			p.set_cover(T)
		Input:
			T: list of list of integers, where all entries are column indices of p.A
				For each l in T we need that p.A[:,l] describes a simplex with some interior points.
				If there already was a cover, it is appended to p.old_covers.
		"""
		self.clean()
		if self.cover is not None:
			warnings.warn('Overwriting cover.')
		if split:
			new_cover = []
			for c in T:
				squares = [e for e in c if e < self.monomial_squares]
				for e in c[len(squares):]:
					new_cover.append(squares + [e])
			self.old_covers[hash(str(new_cover))] = new_cover
			self.cover = new_cover
		else:
			self.old_covers[hash(str(T))] = T.copy()
			self.cover = T

		#store convex combinations
		self.lamb = scipy.sparse.dok_matrix((len(self.cover), self.monomial_squares))
		for k in range(len(self.cover)):
			self.lamb[k,self.cover[k][:-1]] = linsolve(self.A[:,self.cover[k][:-1]], self.A[:,self.cover[k][-1]])
		self.lamb = self.lamb.toarray()

		#ensure that coefficient_distribution is defined
		self._set_coefficient_distribution()

		#update optimisation problem
		if self.prob_sonc is not None:
			self._create_sonc_opt_problem()

	# === Constrained case ===



	# === SONC for general case ===

	def _matrix_list_to_sparse(self, C_list):
		C = np.zeros((len(self.cover), self.monomial_squares, self.A.shape[1] - self.monomial_squares))
		for k in range(len(self.cover)):
			C[np.ix_([k], [e for e in self.cover[k] if e < self.monomial_squares], [e - self.monomial_squares for e in self.cover[k] if e >= self.monomial_squares])] = C_list[k]
		return sparse.COO(C)

	def opt(self, method = 'even', T = None, language = 'python', solver = None):
		"""Optimise the polynomial given by (A,b) via SONC using cvx in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of non-negative circuit polynomials.

		Call:
			data = p.opt([method, T, solver, check])
		Input:
			method [optional, default 'even']: solving method, currently allows the following
				- even: get a cover, where each simplex contains zero,
					split the interior points evenly and opt each simplex separately
				- outer: get a cover with simplices, trying to use each summand
					solve one big problem, with variable splitting of the coefficients
				- cascade: get a cover with simplices, trying to use each summand
					Then optimise each simplex, starting with those far away from zero and update the remainig coefficients.
			solver [optional, default 'sedumi'/'ECOS']: solver, to solve the problem, currenty possible: 
				Matlab: 'sedumi', 'sdpt3'
				Python: 'ECOS', 'SCS', 'CVXOPT'
			T [optional, default None]: a covering of the interior points by simplices
				if none is given, a cover is computed
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: 3D-matrix, coefficients for the decomposition
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		#check whether instance is valid for the problem
		self.clean()

		#compute cover if none is given
		if T is None:
			if self.cover is None:
				self._compute_zero_cover()
		else:
			self.set_cover(T)
			self.cover_time = 0

		#Give warning, if not all positive summands were used
		if {i for i in range(self.monomial_squares)} - set([entry for l in self.cover for entry in l]) != set():
			warnings.warn('Unused positive points.')

		t0 = datetime.now()
		if language == 'python':
			if method == 'even':
				if solver is None: solver = 'ECOS'
				counter = [0 for _ in self.b]
				for t in self.cover:
					for entry in t: counter[entry] += 1
				splitb = self.b / counter
				subpoly = [Polynomial(self.A[:,t], splitb[t], hull_size = len([e for e in t if e < self.monomial_squares]), degenerate_points = []) for t in self.cover]
				for p in subpoly:
					p.sonc_opt_python(solver = solver)
					#if p.solution['status'] == -1:
					#	#try first other solver of the list
					#	p.sonc_opt_python_simplex(solver = [s for s in solver_list if s != solver][0])

				data = {'solution': [p.solution for p in subpoly]}
				
				try:
					data['C'] = scipy.sparse.dok_matrix((len(self.cover), self.monomial_squares))
					for k in range(len(self.cover)):
						data['C'][k, self.cover[k][:-1]] = subpoly[k].solution['C']
					data['C'] = sparse.COO(data['C'], shape = data['C'].shape)
				except Exception as err:
					data['C'] = aux.FAULT_DATA['C']
					data['error'] = repr(err)
				data['language'] = 'python'
				data['opt'] = sum([entry['opt'] for entry in data['solution']])
				data['solver_time'] = sum([entry['solver_time'] for entry in data['solution']])
				data['verify'] = min([entry['verify'] for entry in data['solution']])
				#set status from statuses of sub-problems
				data['status'] = [sol['status'] for sol in data['solution']]

				self.sonc_problem_creation_time = sum([p0.sonc_problem_creation_time for p0 in subpoly])
			elif method == 'cascade':
				raise Exception('Python-Cascade not implemented, yet.')
			elif method == 'outer':
				return self.sonc_opt_python()
			else:
				raise Exception('Unknown method')

		elif language == 'matlab':
			self._ensure_matlab()
			if solver is None: solver = 'sedumi'
			b_relax = np.concatenate((self.b[:self.monomial_squares], -abs(self.b[self.monomial_squares:])))

			self.matlab.cvx_solver(solver)
			if method == 'even':
				matlab_result = self.matlab.opt_sonc_split_even(matlab.double(self.A.tolist()), matlab.double(b_relax.tolist()), [[e + 1 for e in l] for l in self.cover], nargout = 5)
			elif method == 'outer':
				matlab_result = self.matlab.opt_sonc_split_outer(matlab.double(self.A.tolist()), matlab.double(b_relax.tolist()), [[e + 1 for e in l] for l in self.cover], matlab.double(self.coefficient_distribution.todense().tolist()), nargout = 5)
			else:
				raise Exception('Unknown method')

			try:
				data = {}
				data['opt'], data['C'], data['solver_time'], data['status'], data['verify'] = matlab_result
				if method == 'even':
					data['C'] = self._matrix_list_to_sparse([np.array(CC) for CC in data['C']])
				else:
					C = np.array(data['C']).round(aux.DIGITS)
					data['C'] = self._matrix_list_to_sparse([C[k, :len([e for e in self.cover[k] if e < self.monomial_squares]), :(len([e for e in self.cover[k] if e >= self.monomial_squares]))] for k in range(len(self.cover))])
			except Exception as err:
				data = aux.FAULT_DATA.copy()
				data['result'] = matlab_result
				data['error'] = repr(err)
			
			data['language'] = 'matlab'
		else:
			raise Exception('Unknown language')

		self.solution_time = aux.dt2sec(datetime.now() - t0)
		data['solution_time'] = self.solution_time
		data['cover_time'] = self.cover_time

		data['time'] = self.solution_time + self.cover_time
		data['solver'] = solver
		data['strategy'] = 'sonc'
		if not 'params' in data.keys():
			data['params'] = { 'split' : method }
		else:
			data['params']['split'] = method
		data['params']['cover'] = self.cover.copy()
		data['params']['distribution'] = self.coefficient_distribution.todense()
		self._store_solution(data)
		return data

	def _upper_bound(self, max_iters = None):
		"""Give an upper bound to the minimum by guessing starting values and computing the local minima.

		Call:
			fmin, xmin = p._upper_bound()
		Output:
			fmin - float, least value, that was found for polynomial p
			xmin - float-array, argument where fmin is attained
		"""
		#self.__call__ is clearer than just 'self'
		xmin = scipy.optimize.fmin(self.__call__, np.zeros(self.A.shape[0] - 1), disp = False)
		fmin = self.__call__(xmin)
		if max_iters is None:
			max_iters = np.prod(self.A.shape)

		for _ in range(max_iters):
			x0 = self.b.max() * np.random.randn(self.A.shape[0] - 1)
			xmin_tmp = scipy.optimize.fmin(self.__call__, x0, disp = False)
			val = self.__call__(xmin_tmp)
			if val < fmin:
				fmin = val
				xmin = xmin_tmp
		return fmin, xmin

	def bound_min(self):
		"""Compute and print upper and lower bound for the polynomial."""
		fmin, xmin = self._upper_bound()
		if self.old_solutions == {}:
			self.run_all()
		min_key = min(self.old_solutions, key=(lambda k: self.old_solutions[k]['opt']))
		print('Lower bound with: %s' % str(min_key))
		print('Upper bound at: %s' % str(xmin.round(3)))
		print('Lower bound: %.3f' % -self.old_solutions[min_key]['opt'])
		print('Upper bound: %.3f' % fmin)

	def fork(self):
		"""Create a list of worst cases, ranging over all orthants.

		This function is supposed to work together with forked_bound().

		Call:
			poly_list = p.fork()
		Output:
			poly_list: list of Polynomial, all exponents are doubled, so we correspond to ranging over the positive orthant
				the coefficients have the same absolute values, but their signs range over all worst-case scenarios, that can actually occur;
		"""
		base_sign = np.array(self.b[self.monomial_squares:] < 0, dtype=np.int)
		sign_list = [(np.dot(np.array(comb), self.A[1:,self.monomial_squares:]) + base_sign) % 2 for comb in itertools.product([0,1], repeat=self.A.shape[0] - 1)]
		maximal_signs = aux.maximal_elements(sign_list)
		return [Polynomial(2 * self.A[1:,:], np.concatenate((self.b[:self.monomial_squares], (-1)**signs * abs(self.b[self.monomial_squares:])))) for signs in maximal_signs]

	def forked_bound(self):
		"""Give an improved bound by computing a bound for each worst-case orthant.

		Call:
			bound = p.forked_bound()
		Output: 
			bound: float, the smallest bound found over the orthants
		"""
		self.clean()
		polys = self.fork()
		for q in polys:
			q.sonc_opt_python()
		return min([-q.solution['opt'] for q in polys])

	def sonc_realloc(self, max_iters = 10):
		"""Print several solutions for SONC, with sucessively improved distribution of negative coefficients.

		Call:
			p.sonc_realloc(max_iters)
		Input:
			max_iters [optional, default 10]: number of iterations
		"""
		if self.cover is None:
			self._compute_cover()
		self._create_sonc_opt_problem()
		opts = []
		for _ in range(max_iters):
			self.sonc_opt_python()
			opts.append(self.solution['opt'])
			B = self._reallocate_coefficients()
			self._create_sonc_opt_problem(B)
		return opts

	def detect_infinity(self):
		"""Quick check, whether we can verify that the polynomial is unbounded.

		This method iterates over all faces with degenerate points and checks whether the polynomial restricted to that face is negative.

		Call:
			res = p.detect_infinity()
		Output:
			res: 'None' if no negative face was found.
				Otherwise it is a pair consisting of:
				x_min - argument, where neagtive value is obtained
				indices - list of indices, which vertices form that negative face
		"""
		if self.degenerate_points is None:
			self._compute_degenerate_points()
		for deg in self.degenerate_points:
			l = []
			for m in range(self.monomial_squares):
				c = np.zeros(self.monomial_squares)
				c[m] = -1
				res = scipy.optimize.linprog(c, A_eq = self.A[:,:self.monomial_squares], b_eq = self.A[:,deg])
				if res.fun < -aux.EPSILON:
					l.append(m)	
			indices = [np.array_equal(np.dot(self.A[:,l], linsolve(self.A[:,l], self.A[:,i])).round(aux.DIGITS), self.A[:,i]) for i in range(self.A.shape[1])]
			q = Polynomial(self.A[:,indices], self.b[indices])
			x_min = scipy.optimize.fmin(q, np.sign(self.b[deg]) * (-1)**(q.A[1:,:].sum(axis = 1) % 2), disp = False)
			if q(x_min) < -aux.EPSILON:
				return x_min, [i for i in range(self.A.shape[1]) if indices[i]]
		return None

if __name__ == "__main__":
	pass
	#p = Polynomial('standard_simplex',30, 60, 100, seed = 0)
	##collecting further nice examples
	#example4_2 = Polynomial(str(8*x(0)**6 + 6*x(1)**6 + 4*x(2)**6+2*x(3)**6 -3*x(0)**3*x(1)**2 + 8*x(0)**2*x(1)*x(2)*x(3) - 9*x(1)*x(3)**4 + 2*x(0)**2*x(1)*x(3) - 3*x(1)*x(3)**2 + 1))
	#example_small = Polynomial('general',4,8,8,3,seed = 0)
	#ex1 = Polynomial('general',10,20,100,80,seed = 1)
