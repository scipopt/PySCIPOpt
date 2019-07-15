#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Class for Circuit Polynomials, intended for SONC."""

import numpy as np
import sympy

import Poem.polynomial_base as polynomial_base
from Poem.exceptions import WrongModelError
from Poem.aux import linsolve, get_type

x = sympy.IndexedBase('x')

class CircuitPolynomial(polynomial_base.Polynomial):
	"""Class for circuit polynomials.
	
	Circuit polynomials in n variables are polynomials such that the Newton 
	polytope is a simplex, formed by monomial squares, and one exponent is a 
	non-square and lies in the relative interior.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.A.shape[1] > self._variables + 2:
			raise WrongModelError('Too many terms for circuit.')
		self._normalise(zero = False)
		if len(self.non_squares) > 1:
			raise WrongModelError('More than one negative point.')
		if self.non_squares != []:
			self.inner = self.non_squares[0]
			self.hull_vertices = self.monomial_squares.copy()
		else:
			self._compute_convex_hull()
			if len(self.hull_vertices) == self.A.shape[1]:
				raise WrongModelError('No interior point.')
			self.inner = [i for i in range(self.A.shape[1]) if i not in self.hull_vertices][0]

		self.theta = None

		if 'lamb' in kwargs.keys():
			self.lamb = np.array(kwargs['lamb'], dtype = get_type(kwagrs['lamb']))
		else:
			self._compute_convex_combination(self.is_symbolic)

	def _compute_convex_combination(self, symbolic = False):
		if symbolic:
			if self.non_squares != []:
				self.lamb = sympy.linsolve((sympy.Matrix(self.A[:,self.monomial_squares]), sympy.Matrix(self.A[:,self.inner])), [x[i] for i in range(self.A.shape[0])])
			else:
				self.lamb = sympy.linsolve((sympy.Matrix(self.A[:,self.hull_vertices]), sympy.Matrix(self.A[:,self.inner])), [x[i] for i in range(self.A.shape[0])])
			self.lamb = np.array(next(iter(self.lamb)))
		else:
			if self.non_squares != []:
				self.lamb = linsolve(self.A[:, self.monomial_squares], self.A[:,self.inner])
			else:
				self.lamb = linsolve(self.A[:, self.hull_vertices], self.A[:,self.inner])
		if (self.lamb <= 0).any():
			raise WrongModelError('Negative point lies outside.')

	def minimiser(self):
		"""Computer the minimiser of the polynomial."""
		Aq = (self.A[1:,self.hull_vertices[1:]].T - self.A[1:,self.inner])
		#Circuit polynomials are meant for the positive orthant, so we apply abs() on the coefficients
		s = linsolve(Aq, np.log(self.lamb[1:] / abs(self.b[self.hull_vertices[1:]] * self.b[self.inner])))
		return np.exp(s)

	def circuit_number(self):
		"""Compute the circuit number, stored in self.theta."""
		self.theta = ((self.b[self.hull_vertices] / self.lamb) ** self.lamb).prod()
		return self.theta

	def non_negative(self):
		"""Check, whether polynomial is non-negative."""
		if self.theta is None:
			self.circuit_number()
		return self.theta > abs(self.b[self.inner]) or self.non_squares == []
