#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Class for multivariate polynomials in sparse notation, focus on optimisation."""

import numpy as np
import sqlite3
import sympy
import sys
import warnings
import re
import random
from datetime import datetime

import Poem.aux as aux
from Poem.generate_poly import create_standard_simplex_polynomial, create_simplex_polynomial, create_poly
from Poem.polytope import convex_hull

x = sympy.IndexedBase('x')
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
			orthant [default (0,...,0)] - restriction to some orthants, one entry for each variable
				0 - unknown sign
				1/-1 - positive/negative half space
		"""
		if aux.VERBOSE > 2:
			print('number of args: %s' % len(args))
			for arg in args: print(arg)
		# -- setting some default values, so they are defined
		self.degenerate_points = None
		self.monomial_squares = []
		self.hull_vertices = None

		# -- initialise parameters from keywords --
		if 'dirty' in kwargs.keys():
			self.dirty = kwargs['dirty']
		else:
			self.dirty = True

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

		#default for orthant need to know number of variables, hence not earlier
		if 'orthant' in kwargs.keys():
			self.orthant = np.array(kwargs['orthant'].copy(), dtype = np.int)
		else:
			self.orthant = np.zeros(self.A.shape[0] - 1, dtype = np.int)

		if 'is_symbolic' in kwargs.keys():
			self.is_symbolic = kwargs['is_symbolic']
		else:
			self.is_symbolic = all([type(coeff) in [sympy.Rational, sympy.Integer, int, np.int] for coeff in self.b])

		self._degree = max([sum(self.A[1:,i]) for i in range(self.A.shape[1])])
		self._variables = self.A[1:,:].any(axis = 1).sum()

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
			self.hull_vertices = list(range(n+1))
		elif shape == 'simplex':
			self.degenerate_points = []
			self.__read_matrix_vector(*create_simplex_polynomial(n,d,t))
			self.hull_vertices = list(range(n+1))
		elif shape == 'general':
			self.__read_matrix_vector(*create_poly(n,d,t, inner = inner))
		else:
			raise Exception('Unknown shape, possible: standard_simplex, simplex, general')

	def __read_matrix_vector(self, A, b):
		A = np.array(A)
		if len(b) == 0:
			self.b = np.zeros(1, dtype = np.int)
		else:
			self.b = np.array(b, dtype = aux.get_type(b))
		if 0 in A.shape:
			A = np.array([[1],[0]], dtype = np.int)
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
		cursor.execute('select string from polynomial where rowid = ?;', (rowid,))
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
		if s == '' and aux.VERBOSE:
			warnings.warn('Creating polynomial from empty string.')
			self.__read_matrix_vector(np.array([[1],[0]]),[0])
			return
		if s =='0': 
			self.__read_matrix_vector(np.array([[1],[0]]),[0])
			return
		#get number of variables, '+1' for x0
		s = s.replace('(','').replace(')','')
		n = max([int(i) for i in re.findall(r'x\[?([0-9]+)\]?', s)]) + 1
		if reduce_vars:
			n_min = min([int(i) for i in re.findall(r'x\[?([0-9]+)\]?', s)])
		else:
			n_min = 0

		#transform into some standard form
		pattern = re.compile(r'([^e])-')
		terms = pattern.sub(r'\1+-', s.replace(' ','')).replace('**','^').replace('/','*1/').split('+')
		t = len(terms)
		A = np.zeros((n - n_min, t), dtype = np.int)
		b = np.ones(t, dtype = object)
		for i in range(t):
			term = terms[i].split('*')
			#get the coefficient
			if term[0][0] == '-':
				b[i] *= -1
				term[0] = term[0][1:]
			for var in term:
				if var.find('x') == -1:
					b[i] *= aux.parse(var)
				else:
					if var.find('^') == -1:
						entry = (re.findall(r'x\[?([0-9]+)\]?', var)[0],1)
					else:
						entry = re.findall(r'x\[?([0-9]+)\]?\^([0-9]+)', var)[0]

					A[int(entry[0]) - n_min, i] = int(entry[1])
		self.__read_matrix_vector(A,b)
	
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

	def to_symbolic(self):
		"""Return the polynomial as symbolic expression in sympy."""
		A = self.A[1:,:]
		rows,cols = A.shape
		return sum([self.b[j] * sympy.prod([x[i] ** A[i,j] for i in range(rows)]) for j in range(cols)])

	def __sizeof__(self):
		"""Return bit-size of the instance."""
		return aux.bitsize(self.A) + aux.bitsize(self.b)

	def scaleround(self, factor):
		"""Scale polynomial and round coefficients to integer.

		Call:
			p.scaleround(factor)
		Input:
			factor [number] - scale all coefficients by 'factor', then round to integer

		Note: This function changes the coefficients in place and sets the 'dirty' flag to 'True'.
		"""
		self.b = np.array((factor * np.array(self.b, dtype = np.float)).round(), dtype = np.int)
		self.dirty = True

	# === Canonical Form ===
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

		Ensure that summands appear in lexicographic order.
		"""
		##TODO:may leave the first part, if we know there are no double exponents
		#collect all same exponents
		base = {tuple(self.A[:,i]) : 0 for i in range(self.A.shape[1])}
		for i in range(self.A.shape[1]):
			base[tuple(self.A[:,i])] += self.b[i]
		#keep only non-zero terms, and possibly constant term
		base = { key : base[key] for key in base.keys() if base[key] != 0 or (all([entry == 0 for entry in key[1:]]) and zero) }
		if base == {}: base = { tuple(0 for _ in range(self.A.shape[0])) : 0 }
		l = list(base.keys())
		l.sort()
		self.A = np.zeros((self.A.shape[0],len(l)), dtype = np.int)
		self.b = np.zeros(len(l), dtype = self.b.dtype)
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

		#reorder the entries
		self.A = self.A[:,np.lexsort(np.flipud(self.A))]
		#find indices of monomial squares
		sign = np.array(np.sign(self.b), dtype = np.int)
		#square_index = [i for i in range(self.A.shape[1]) if (self.b[i] > 0 and not (self.A[1:,i] % 2).any()) or not self.A[1:,i].any()]
		self.monomial_squares = [i for i in range(self.A.shape[1]) if sign[i] * np.prod(self.orthant ** (self.A[1:,i] % 2)) == 1 or not self.A[1:,i].any()]
		self.non_squares = [i for i in range(self.A.shape[1]) if i not in self.monomial_squares]

			#index = ((self.A[1:,self.monomial_squares:] % 2).sum(axis = 1) * (1 - abs(self.orthant))).argmax()
		##rearrange order of variables
		#var_indices = list(range(1, self.A.shape[0]))
		#var_indices.sort(key = lambda i: (self.A[i,self.monomial_squares:] % 2).sum(), reverse = True)
		#self.A[1:,:] = self.A[var_indices, :]
		#self.orthant = self.orthant[[idx - 1 for idx in var_indices]]

		self.dirty = False

	def _compute_convex_hull(self):
		"""Compute the convex hull, store indices in self.hull_vertices."""
		if self.hull_vertices is None:
			self.hull_vertices = convex_hull(self.A)

	#=== Arithmetic ===
	
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

	def __eq__(self, other):
		"""Check equality of polynomials."""
		self.clean()
		other.clean()
		return np.array_equal(self.A, other.A) and np.array_equal(self.b, other.b)

	def __call__(self,x, dtype = 'float'):
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

	#=== Derivatives ===

	def derive(self, index):
		"""Compute the derivative with respect to the given index.

		Call:
			res = p.derive(index)
		Input:
			index [integer] - index of variable, by which we derive p, starting with zero
		Output:
			res - Polynomial, derivative of p by x_index
		"""
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
		"""Compute full derivative of the polynomial.

		Call:
			pprime = p.prime([variables])
		Input:
			variables [optional, default: all occurring] - number of variables, by which we derive
		Output:
			pprime - Polynomial, derivative of p
		"""
		if variables is None:
			variables = self.A.shape[0] - 1
		return tuple(self.derive(i) for i in range(variables))

if __name__ == "__main__":
	pass
	#p = Polynomial('standard_simplex',30, 60, 100, seed = 0)
	##collecting further nice examples
	#example4_2 = Polynomial(str(8*x(0)**6 + 6*x(1)**6 + 4*x(2)**6+2*x(3)**6 -3*x(0)**3*x(1)**2 + 8*x(0)**2*x(1)*x(2)*x(3) - 9*x(1)*x(3)**4 + 2*x(0)**2*x(1)*x(3) - 3*x(1)*x(3)**2 + 1))
	#example_small = Polynomial('general',4,8,8,3,seed = 0)
	#ex1 = Polynomial('general',10,20,100,80,seed = 1)
