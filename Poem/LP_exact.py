#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Define constants and basic functions."""

import numpy as np
import sympy
import math
import subprocess
import shutil

#LP_solver_found is True is soplex or cdd is found.
LP_solver_found = (shutil.which('soplex') is not None)
try:
	import cdd
	cdd_found = True
	LP_solver_found = True
except ImportError:
	cdd_found = False

def LP_solve_exact(A,b,c = None, box = None):
	"""Exactly solve an LP min{cx : Ax <= b, x in box}.

	Call:
		opt = LP_solve_exact(A,b[,c,box])
	Input:
		A - `m x n`-matrix
		b - array of length m
		c [optional] - array of size n
		box [optional] - list of pairs, with length n;
			(low_i, up_i) implies constraint low_i <= x_i <= up_i
	Output:
		opt - array of length n; optimal (or just feasible if c = None) solution
	"""
	if not LP_solver_found:
		raise ModuleNotFoundError('No exact LP solver found. Install Soplex or cdd.')
	LP_tmp = LP(A, b, c = c, box = box)
	if cdd_found:
		return LP_tmp.solve_cdd()
	else:
		return LP_tmp.solve_soplex()

def get_box(value, digits):
	"""Obtain enclosing interval for given value and accuracy.

	Call:
		box = get_box(value, digits)
	Input:
		value [number]
		digits [positive integer] - accuracy in binary digits
	Output:
		box: pair of sympy-fractions, giving lower and upper bound for value
			such that upper - lower <= 2^-digits
	"""
	eps = sympy.Rational(1,2**digits)
	low = sympy.sympify(int(math.floor(value)))
	up = sympy.sympify(int(math.ceil(value)))
	while up - low > eps:
		mid = (up+low)/2
		if mid > value:
			up = mid
		else:
			low = mid
	return low,up

class LP(object):
	"""Class to solve an LP exactly."""

	def __init__(self, A, b, c = None, box = None):
		"""Create LP problem min{cx : Ax <= b, x in box}."""
		self.A = np.array(A, dtype = type(A[0,0]))
		self.b = np.array(b, dtype = type(b[0]))
		if c is None:
			self.c = np.zeros(A.shape[1] + 1, dtype = np.int)
		else:
			self.c = np.zeros(len(c) + 1, dtype = type(c[0]))
			self.c[1:] = c
		self.box = box

	def solve_cdd(self):
		"""Solve the LP via cdd."""
		#Meaning of cdd-Matrix: A * (1,x) >= 0
		n,m = self.A.shape
		if self.box is None:
			ineq_size = m
		else:
			ineq_size = 2 * m

		mat = np.zeros((2*n + ineq_size, m + 1), dtype = object)
		mat[:n,0] = -self.b
		mat[n:2*n,0] = self.b
		mat[:n,1:] = self.A
		mat[n:2*n,1:] = -self.A
		mat[2*n:2*n + m,1:] = np.eye(m, dtype = np.int)
		if self.box is not None:
			#must write fractions as string; otherwise they are treated as zero
			mat[2*n:2*n+m,0] = np.array([str(-bound[0]) for bound in self.box])
			mat[2*n + m:,1:] = -np.eye(m, dtype = np.int)
			mat[2*n+m:,0] = np.array([str(bound[1]) for bound in self.box])
		
		mat = cdd.Matrix(mat, number_type = 'fraction')
		mat.obj_type = cdd.LPObjType.MAX
		mat.obj_func = self.c
		lp = cdd.LinProg(mat)
		lp.solve()
		if lp.status == cdd.LPStatusType.OPTIMAL:
			return np.array([sympy.sympify(entry) for entry in lp.primal_solution])

	def _write_CPLEX_LP(self, filename = 'LP.tmp'):
		"""Write the LP in CPLEX-LP format."""
		n,m = self.A.shape
		writer = open('LP.tmp','w')
		writer.write('Minimize\n\tobj: %s\nSubject To\n' % (' + '.join(['%d x%d' % (self.c[i], i) for i in range(m)]) if self.c is not None else '0'))

		for i in range(n):
			writer.write('\tc%d: %s = %d\n' % (i, ' + '.join(['%d x%d' % (self.A[i,j], j) for j in range(m) if self.A[i,j] != 0]), self.b[i]))

		#write inequalities
		if self.box is None:
			writer.write('Bounds\n')
			for j in range(m):
				writer.write('\t0 <= x%d\n' % j)
		else:
			for i, (low, up) in enumerate(self.box):
				if isinstance(low, sympy.Rational):
					scale_low = low.q
					low = low.p
				elif type(low) == int:
					scale_low = 1
				else:
					raise Exception('Unexpected type: %s' % str(type(low)))
				if isinstance(up, sympy.Rational):
					scale_up = up.q
					up = up.p
				elif type(up) == int:
					scale_up = 1
				else:
					raise Exception('Unexpected type: %s' % str(type(up)))
				writer.write('\tbl%d: %d x%d >= %d\n' % (i,scale_low, i, low))
				writer.write('\tbu%d: %d x%d <= %d\n' % (i,scale_up, i, up))
			writer.write('Bounds\n' + '\n'.join(['\tx%d free' % i for i in range(m)]))

		writer.write('\nEnd')
		writer.close()

	def solve_soplex(self):
		"""Solve the LP: min c*x s.t. A*x = b, x >= 0."""
		self._write_CPLEX_LP()

		res = subprocess.check_output(['soplex','-X','-o0','-f0','LP.tmp'], universal_newlines = True)
		values = [line.split('\t') for line in res.splitlines() if line != '' and line[0] == 'x']
		res = np.zeros(self.A.shape[1], dtype = object)
		for variables, value in values:
			res[int(variables[1:])] = sympy.sympify(value)
		return res
