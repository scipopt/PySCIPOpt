#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Class for Circuit Polynomials, intended for SONC."""

import numpy as np
import sympy
import sys

import cvxpy as cvx

import Poem.polynomial_base as polynomial_base
from Poem.exceptions import WrongModelError
from Poem.aux import symlog, bitsize, get_type

x = sympy.IndexedBase('x')

class AGEPolynomial(polynomial_base.Polynomial):
	"""Class for AGE-polynomials.
	
	Arithmetic-geometric-mean-exponentials (AGE) are polynomials where at most 
	one terms is not a monomial square.
	"""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._normalise(zero = False)
		if len(self.non_squares) > 1:
			raise WrongModelError('More than one negative point: %s' % str(self.non_squares))
		if self.non_squares != []:
			self.inner = self.non_squares[0]

		if 'lamb' in kwargs.keys():
			self.lamb = np.array(kwargs['lamb'], dtype = get_type(kwargs['lamb']))
		else:
			if self.non_squares == []:
				self.lamb = np.zeros(self.A.shape[1], dtype = np.int)
			else:
				lamb = cvx.Variable(self.A.shape[1], nonneg = True)
				prob = cvx.Problem(cvx.Minimize(0), [self.A[:,self.monomial_squares] * lamb == self.A[:,self.inner]])
				self.lamb = np.zeros(self.A.shape[1])
				self.lamb[self.monomial_squares] = prob.variables()[0].value

	def non_negative(self):
		"""Check, whether polynomial is non-negative."""
		e = sympy.E if self.is_symbolic else np.e
		idx = [i for i in self.monomial_squares if self.lamb[i] > 0]
		lamb = self.lamb[idx]
		return (lamb * symlog(lamb / self.b[idx]) - lamb).sum() <= self.b[self.inner]

	def __sizeof__(self):
		"""Return bit-size of the instance, uncluding convex combination lambda."""
		return super().__sizeof__() + bitsize(self.lamb)
