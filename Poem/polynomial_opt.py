#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Class for multivariate polynomials in sparse notation, focus on optimisation."""

import numpy as np
import sympy
import scipy.optimize
import warnings
import random
from datetime import datetime
import os

import json_tricks as json
import cvxpy as cvx
from tabulate import tabulate
import sparse
import pymp
try:
	import z3
	z3_found = True
except ImportError:
	z3_found = False
try:
	import matlab
	import matlab.engine
	matlab_found = True
except ImportError:
	matlab_found = False

import Poem.aux as aux
from Poem.aux import binomial, is_psd, linsolve
import Poem.polytope as polytope
from Poem.exceptions import *
import Poem.polynomial_base as polynomial_base
from Poem.circuit_polynomial import CircuitPolynomial
from Poem.AGE_polynomial import AGEPolynomial

x = sympy.IndexedBase('x')
sympy.init_printing();
np.set_printoptions(linewidth = 200)

class Polynomial(polynomial_base.Polynomial):
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
		super().__init__(*args, **kwargs)
		t0 = datetime.now()

		# -- initialise parameters from keywords --
		if 'matlab_instance' in kwargs.keys():
			self.matlab = kwargs['matlab_instance']
			self.matlab_start_time = 0
		else:
			self.matlab = None

		# -- set further defaults --
		self.cover_time = 0
		self.cover = None
		self.solution = None
		self.old_solutions = {}
		self.old_covers = {}
		self.min = (np.inf, np.array([np.inf for _ in range(self.A.shape[0] - 1)]))
		self.lower_bound = -np.inf
		#optimisation problems
		self.clear()

		self.init_time += aux.dt2sec(datetime.now() - t0)

	def clear(self):
		"""Delete all problem associated with the polynomial, to save memory."""
		self.prob_sos = None
		self.prob_sos_sparse = None
		self.prob_sos_full = None
		self.prob_sonc = None
		self.prob_sage = None

	# === Output ===

	def get_solutions(self):
		"""Return a list of (solver, time, optimum) for all solutions found."""
		return [(key, self.old_solutions[key]['time'], self.old_solutions[key]['opt']) for key in self.old_solutions.keys()]

	def get_solution(self, index):
		"""Return the solution given by the index."""
		#TODO: make more efficient by lookup table
		for sol in self.old_solutions.values():
			if sol['index'] == index:
				return sol

	def print_solutions(self, form = 'grid', only_valid = False, params = False):
		"""Print a table of all stored solutions.

		You can obtain the solution with a given index by p.get_solution(<index>).

		Call:
			p.print_solutions([only_valid, form])
		Input:
			only_valid [boolean, default False]: flag, whether to print only verified solutions
				i.e. those with <solution>['verify'] == 1
			form [string, default 'grid'] - tableformat for tabulate
			params [boolean, default False]: flag, whether to print the parameters
		"""
		if params:
			print(tabulate([[self.old_solutions[key]['index']] + list(key) + [self.old_solutions[key][k] for k in ['time','opt','verify']] for key in self.old_solutions.keys() if (not only_valid) or (self.old_solutions[key]['verify'] == 1)], ['index','language','strategy','modeler','solver','params','time','opt', 'verify'], tablefmt = form))
		else:
			print(tabulate([[self.old_solutions[key]['index']] + list(key)[:-1] + [self.old_solutions[key][k] for k in ['time','opt','verify']] for key in self.old_solutions.keys() if (not only_valid) or (self.old_solutions[key]['verify'] == 1)], ['index','language','strategy','modeler','solver','time','opt', 'verify'], tablefmt = form))

	def relax(self):
		"""Return a lower estimate for the polynomial.

		All potentially newgative terms are made negative at once.
		This function should be used, when checking the decomposition, obtained by get_decomposition(), since that functions works on the relaxation.
		"""
		self.clean()
		b_relax = self.b.copy()
		b_relax[self.monomial_squares] = abs(self.b[self.monomial_squares])
		b_relax[self.non_squares] = -abs(self.b[self.non_squares])
		return Polynomial(self.A.copy(), b_relax, degenerate_points = self.degenerate_points)

	# === Formulating the problems ===

	def _create_sos_opt_problem(self, sparse = True):
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
		gamma = cvx.Variable()

		if sparse:
			support = np.array(polytope.interior_points(self.A, strict = False))[:,1:]
			half_support = [tuple(v // 2) for v in support if not (v % 2).any()]
			C = cvx.Variable((len(half_support),len(half_support)), PSD = True)
			coeffs = {tuple(e): 0 for e in support}
			for i in range(self.A.shape[1]):
				coeffs[tuple(self.A[1:,i])] += self.b[i]
			#create lookup table: vector -> index
			lookup = {half_support[i] : i for i in range(len(half_support))}
			constraints = []
			for v,c in coeffs.items():
				if not any(v):
					#constant term gets special treatment
					constraints.append(C[0,0] == coeffs[v] + gamma)
					continue
				#list all (indices of) pairs in half_support, that add up to v
				l = []
				for u in half_support:
					diff = tuple(v[i] - u[i] for i in range(len(v)))
					if diff in half_support:
						l.append((lookup[u],lookup[diff]))
				constraints.append(cvx.Zero(cvx.sum([C[i,j] for i,j in l]) - cvx.expressions.constants.Constant(c)))
			#define the problem
			self.prob_sos_sparse = cvx.Problem(cvx.Minimize(gamma),constraints)
			self.prob_sos = self.prob_sos_sparse
		else:
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
			self.prob_sos_full = cvx.Problem(cvx.Minimize(gamma),constraints)
			self.prob_sos = self.prob_sos_full

		self.sos_problem_creation_time = aux.dt2sec(datetime.now() - t0)

	def _create_sage_opt_problem(self):
		"""Create the SAGE-optimisation-problem in cvx for the polynomial given by (A,b).

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a SAGE.

		Note: This function does NOT call a solver. It only states the problem and does not solve it.

		Call:
			p._create_sage_opt_problem()
		Creates:
			p.prob_sage: cvx.Problem-instance
		"""
		self.clean()
		t0 = datetime.now()
		#define short notation
		A = self.A[1:,:]
		n,t = A.shape

		b_relax = self.b.copy()
		b_relax[self.monomial_squares] = abs(self.b[self.monomial_squares])
		b_relax[self.non_squares] = -abs(self.b[self.non_squares])

		X = cvx.Variable(shape = (t,t), name = 'X', nonneg = True)
		#lamb[k,i]: barycentric coordinate, using A[:,i] to represent A[:,k]
		lamb = cvx.Variable(shape = (t,t), name = 'lambda', nonneg = True)
		#we use both variables only for k >= monomial_squares

		constraints = []
		constraints += [b_relax[i] == -2*X[i,i] + cvx.sum(X[:,i]) for i in self.non_squares]
		constraints += [b_relax[i] == cvx.sum(X[:,i]) for i in self.monomial_squares[1:]]
		constraints += [2*lamb[k,k] == cvx.sum(lamb[k,:]) for k in self.non_squares]
		constraints += [cvx.sum([A[:,i] * lamb[k,i] for i in range(t) if i != k]) == A[:,k]*lamb[k,k] for k in self.non_squares]
		constraints += [cvx.sum(cvx.kl_div(lamb[k,:], X[k,:])[[i for i in range(t) if i != k]]) <= -2*X[k,k] + cvx.sum(X[k,:]) for k in self.non_squares]

		objective = cvx.Minimize(cvx.sum(X[:,0]))
		self.prob_sage = cvx.Problem(objective, constraints)

		self.sage_problem_creation_time = aux.dt2sec(datetime.now() - t0)

	def _create_sonc_opt_problem(self, B = None, split = True):
		"""Create the SONC-optimisation-problem in cvx for the polynomial given by (A,b).

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of non-negative circuit polynomials..

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
		b = np.array(self.b, dtype = np.float)

		X = cvx.Variable((len(self.cover), self.A.shape[1]))

		constraints = []
		for i in self.monomial_squares[1:]:
			indices = [k for k in range(len(self.cover)) if i in self.cover[k]]
			if indices != []:
				constraints.append(cvx.log_sum_exp(X[indices, i]) <= np.log(abs(b[i])))
		for k in range(len(self.cover)):
			lamb = self.lamb[k,self.cover[k][:-1]]
			constraints.append(np.log(abs(self.coefficient_distribution[k, self.cover[k][-1]])) == cvx.sum(cvx.multiply(lamb, X[k, self.cover[k][:-1]]) - (lamb * np.log(lamb))))

		if any([0 in c for c in self.cover]):
			objective = cvx.Minimize(cvx.log_sum_exp(X[[k for k in range(len(self.cover)) if 0 in self.cover[k]],0]))
		else:
			objective = cvx.Minimize(0)
		self.prob_sonc = cvx.Problem(objective, constraints)

		self.sonc_problem_creation_time = aux.dt2sec(datetime.now() - t0) + self.cover_time

	def _set_coefficient_distribution(self, B = None):
		if B is None:
			count = np.zeros(self.A.shape[1], dtype = np.int)
			for t in self.cover:
				count[t[-1]] += 1
			idx = [i for i in range(len(count)) if count[i] > 0]

			#b_relax = -abs(self.b)
			#b_relax[idx] /= count[idx]
			B = scipy.sparse.dok_matrix((len(self.cover), self.A.shape[1]))
			for k in range(len(self.cover)):
				B[k, self.cover[k][-1]] = -abs(self.b[self.cover[k][-1]]) / count[self.cover[k][-1]]

		self.coefficient_distribution = B.copy()

	# === Reallocation of SONC coefficients ===

	def _reallocate_coefficients(self):
		"""Given a solution, this function computes an improved distribution of the negative coefficients among the simplex polynomials.

		Note: This makes sense only for the non-simplex case.

		Call:
			B = p._reallocate_coefficients()
		Output:
			B - sparse matrix, where B[k,j] denoted how much of p.b[j] goes into the k-th simplex polynomial.
		"""
		if self.solution is None:
			return

		#init
		try:
			C = self.solution['C'].todense()
		except AttributeError:
			C = self.solution['C']
		circ = np.zeros(len(self.cover))

		cover_indices = [k for k in range(len(self.cover)) if 0 in self.cover[k]]

		#compute convex combinations and something-like-circuit-number
		for k in cover_indices:
			circ[k] = ((C[k,self.cover[k][1:-1]]/self.lamb[k,self.cover[k][1:-1]]) ** (self.lamb[k,self.cover[k][1:-1]] / (1 - self.lamb[k,0]))).prod()

		#compute common derivative of the b[:, j]
		const = np.zeros(self.A.shape[1])
		for j in self.non_squares:
			relevant_indices = [k for k in cover_indices if j in self.cover[k]]
			if len(relevant_indices) <= 1: continue
			f = (lambda a: np.sum([a**(self.lamb[k,0]/(1 - self.lamb[k,0])) * circ[k] for k in relevant_indices]) - abs(self.b[j]))
			upper = max([(abs(self.b[j]) / circ[k]) ** ((1 - self.lamb[k,0])/ self.lamb[k,0]) for k in relevant_indices]) + 1
			try:
				const[j] = scipy.optimize.brentq(f, 0, upper)
			except:
				const[j] = scipy.optimize.bisect(f, 0, upper)

		#compute output
		B = scipy.sparse.dok_matrix(C.shape)
		for k in cover_indices:
			j = self.cover[k][-1]
			if const[j] == 0:
				#in this case the above computation was not executed, but j in cover[k], so this is its only occurrence
				B[k,j] = -abs(self.b[j])
			else:
				B[k,j] = - const[j] ** (self.lamb[k,0]/(1 - self.lamb[k,0])) * circ[k]

		#return scipy.sparse.coo_matrix(B)
		return B

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

	def sos_opt_python(self, solver = 'CVXOPT', sparse = True, **solverargs):
		"""Optimise the polynomial given by (A,b) via SOS using cvx.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Note: scs randomly runs VERY long on trivial instances. Usage is possible, but discouraged.

		Call:
			data = p.sos_opt_python(A,b,[solver],**solverargs)
		Input:
			solver [optional, default 'CVXOPT']: solver, to solve the problem, currenty possible: 'CVXOPT', 'Mosek', 'SCS'
			solverargs: dictionary of keywords, handed to the solver
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: psd-matrix such that p = Z^T * C * Z, where Z is the vector of all monomials
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		sos_size = binomial(self.A.shape[0] - 1 + self._degree//2, self._degree//2)
		if self.prob_sos is None or (sparse and self.prob_sos_sparse is None) or (not sparse and self.prob_sos_full is None):
			self._create_sos_opt_problem(sparse = sparse)
		if sparse:
			self.prob_sos = self.prob_sos_sparse
		else:
			self.prob_sos = self.prob_sos_full

		t0 = datetime.now()

		if not 'verbose' in solverargs.keys(): solverargs['verbose'] = aux.VERBOSE
		if solver == cvx.SCS:
			#setting some defaults for scs
			kwargs = {'eps': aux.EPSILON / 10, 'max_iters' : 20000}
			kwargs.update(solverargs)
			params = kwargs
		else:
			params = solverargs
		try:
			opt = self.prob_sos.solve(solver = solver, **params)
			C = self.prob_sos.variables()[1].value
			#test, whether solution satisfies constraints
			if C is not None and all([c.violation() <= aux.EPSILON for c in self.prob_sos.constraints]) and is_psd(C):
				verify = 1
			else:
				verify = -1
			data = {'opt': opt, 'C': C, 'status': self.prob_sos.status, 'verify': verify}
		except cvx.SolverError as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)
		try:
			data['solver_time'] = self.prob_sos.solver_stats.solve_time + self.prob_sos.solver_stats.setup_time
		except (TypeError, AttributeError):
			data['solver_time'] = 0

		self.solution_time = aux.dt2sec(datetime.now() - t0)

		params['sparse'] = sparse
		data['time'] = self.solution_time
		data['language'] = 'python'
		data['modeler'] = 'cvxpy'
		data['solver'] = solver
		data['strategy'] = 'sos'
		data['params'] = params
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
		data['modeler'] = 'cvx'
		data['solver'] = solver
		data['strategy'] = 'sos'
		self._store_solution(data)
		return data

	def sostools_opt(self, solver = 'sedumi', sparse = False):
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
		#TODO: include the given solver in the argument
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
		data['solver'] = solver
		data['modeler'] = 'sostools'
		data['strategy'] = 'sos'
		data['params'] = {'sparse': sparse}
		self._store_solution(data)
		return data

	def yalmip_opt(self, method = 'solvesos', solver = 'sedumi'):
		"""Optimise the polynomial given by (A,b) via SOS using YALMIP in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of squares.
		This is the case iff there is some psd-matrix C such that p = Z^T * C * Z, where Z is the vector of all monomials.

		Call:
			data = p.yalmip_opt()
		Input:
			method [optional, default 'solvesos']: which method in Yalmip to use
				- solvesos
				- sparsepop
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
		prog += 'ops = sdpsettings(\'solver\',%s);' % solver
		if method == 'solvesos':
			prog += 'prog = sos(p + gam)\n[sol,v,C] = solvesos(prog, gam, [], gam, ops);\n'
			prog += 'opt = value(gam);\nC = C{1};\n'
		else:
			if method == 'sparsepop':
				prog += 'ops = sdpsettings(ops,\'savesolveroutput\',1,\'solver\',\'sparsepop\');\n'
			elif method == 'optimize':
				raise Exception('Extraction of solution not implemented.')
			else:
				raise Exception('Unknown method for Yalmip')
			prog += 'sol = optimize([],p,ops);\n'
			prog += 'opt = -sol.solveroutput.SDPobjValue;\n'
			prog += 'n = sqrt(size(sol.solveroutput.SDPinfo.x,1));\n'
			prog += 'C = reshape(sol.solveroutput.SDPinfo.x,n,n);\n'
		prog += 'time = sol.yalmiptime + sol.solvertime;\nnumerr = sol.problem;'

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
		data['solver'] = solver
		data['modeler'] = 'yalmip'
		data['strategy'] = 'sos'
		data['params'] = {'method': method}
		self._store_solution(data)
		return data

	def gloptipoly_opt(self, solver = 'sedumi'):
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
		#TODO: include the solver given in the argument
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
		data['modeler'] = 'gloptipoly'
		data['solver'] = solver
		data['strategy'] = 'sos'
		self._store_solution(data)
		return data

	# === calling SAGE solver ===

	def sage_opt_python(self, solver = 'ECOS', **solverargs):
		"""Optimise the polynomial given by (A,b) via SAGE using cvxpy.

		Let p be the polynomial given by (A,b), where A is an (n x t)-matrix and b in R^t.
		We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of arithmetic-geometric-mean exponentials (SAGE).

		Call:
			data = p.sage_opt_python([solver], **solverargs)
		Input:
			solver [optional, default 'ECOS']: solver, to solve the problem, currenty possible: 'ECOS', 'cvxopt', 'scs'
			solverargs: dictionary of keywords, handed to the solver
		Output:
			data: dictionary containing information about the solution
				- opt: optimal value
				- C: (t x t)-matrix, coefficients for the decomposition,
				- lambda: (t x t)-matrix, barycentri c coordinates
				- time: time to compute the solution
				- verify: 1 = Solved, -1 = error, 0 = otherwise/unchecked
				- status: status message of the solver
		"""
		self.clean()
		#create problem instance
		if self.prob_sage is None:
			self._create_sage_opt_problem()

		#parsing keywords
		if solver == 'scs':
			kwargs = {'eps': aux.EPSILON / 10, 'max_iters' : 20000}
		elif solver == 'ECOS':
			kwargs = {'reltol': aux.EPSILON / 10, 'max_iters' : 1000, 'feastol': aux.EPSILON / 10, 'abstol': aux.EPSILON / 10}
		else:
			kwargs = {}
		kwargs['verbose'] = (aux.VERBOSE > 0)
		kwargs.update(solverargs)

		#call the solver and handle the result
		t0 = datetime.now()
		try:
			self.prob_sage.solve(solver = solver, **kwargs)

			C = self.prob_sage.variables()[0].value
			#eliminate zeros, but do not violate other constraints
			C += (C == 0) * aux.EPSILON / self.A.shape[1]

			opt = np.sum(C[:,0]) - self.b[0]

			#test, whether solution satisfies constraints
			if max(aux.flatten([c.violation() for c in self.prob_sage.constraints])) < aux.EPSILON:
				verify = 1
			else:
				verify = -1
			data = {'opt': opt, 'C': C, 'lambda': self.prob_sage.variables()[1].value, 'solver_time': self.prob_sage.solver_stats.solve_time + self.prob_sage.solver_stats.setup_time, 'status': self.prob_sage.status, 'verify': verify}
		except Exception as err:
			data = aux.FAULT_DATA.copy()
			data['error'] = repr(err)
		try:
			data['solver_time'] = self.prob_sage.solver_stats.solve_time + self.prob_sage.solver_stats.setup_time
		except (TypeError, AttributeError):
			data['solver_time'] = 0

		self.solution_time = aux.dt2sec(datetime.now() - t0)

		data['time'] = self.solution_time
		data['language'] = 'python'
		data['solver'] = solver
		data['modeler'] = 'cvxpy'
		data['strategy'] = 'sage'
		data['params'] = solverargs
		self._store_solution(data)
		return data

	# === calling SONC solver ===

	def sonc_opt_python(self, solver = 'ECOS', **solverargs):
		"""Optimise the polynomial given by (A,b) via SONC using cvxpy.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of non-negative circuit polynomials.

		Call:
			data = p.sonc_opt_python([solver], **solverargs)
		Input:
			solver [optional, default 'ECOS']: solver, to solve the problem, currenty possible: 'ECOS', 'cvxopt', 'scs'
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
		if solver == 'scs':
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
			C = sparse.COO(np.array(coords).T,data,shape=(len(self.cover), self.A.shape[1]))

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
		except (TypeError, AttributeError):
			data['solver_time'] = 0

		self.solution_time = aux.dt2sec(datetime.now() - t0)

		solverargs['split'] = 'outer'
		data['time'] = self.solution_time
		data['language'] = 'python'
		data['solver'] = solver
		data['modeler'] = 'cvxpy'
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
		#TODO: probably fails, if the class is imported from another directory
		self.matlab.cd(os.getcwd())
		self.matlab.cd('../matlab')

	def _store_solution(self, data):
		"""Store new solution, but keep the previous ones for different methods."""
		if data['strategy'] in ['traverse', 'fork']:
			data['C'] = np.array([[]])
			data['status'] = 1
		if type(data['C']) == np.ndarray and len(data['C'].shape) == 1:
			data['C'] = np.array(np.matrix(data['C']).T)

		params = data['params'] if 'params' in data.keys() else {}

		data['status'] = aux.unify_status(data['status'])

		#summing up all computation times
		data['init_time'] = self.init_time
		data['time'] += self.init_time
		if data['language'] == 'python' and data['strategy'] not in ['trivial', 'traverse', 'fork']:
			if data['strategy'] == 'sos':
				data['problem_creation_time'] = self.sos_problem_creation_time
			elif data['strategy'] == 'sonc':
				data['problem_creation_time'] = self.sonc_problem_creation_time
			elif data['strategy'] == 'sage':
				data['problem_creation_time'] = self.sage_problem_creation_time
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
				elif data['strategy'] == 'sonc' and aux.VERBOSE:
					warnings.warn('Verification for SONC-Matlab not implemented, Python is verified.')
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
				elif data['strategy'] == 'sage':
					warnings.warn('SAGE solution should be verified.')
				else:
					raise Exception('Unknown strategy')
			except Exception as err:
				warnings.warn('Cannot verify solution: %s' % repr(err))

		self.solution = data

		#update lower bound
		#if (data['status'] == 1 or data['verify'] == 1) and self.lower_bound < -data['opt']:
		if data['verify'] == 1 and self.lower_bound < -data['opt']:
			self.lower_bound = -data['opt']
		#print('params = ', params)
		key = (self.solution['language'], self.solution['strategy'], self.solution['modeler'], self.solution['solver'], json.dumps(params))
		if not key in self.old_solutions.keys():
			self.solution['index'] = len(self.old_solutions)
			self.old_solutions[key] = self.solution.copy()

	def opt(self, T = None, language = 'python', solver = None):
		"""Optimise the polynomial given by (A,b) via SONC using cvx in Matlab.

		Let p be the polynomial given by (A,b). We want to find min{ p(x) : x in R^n} by asking, what is the minimal gamma such that p + gamma is a sum of non-negative circuit polynomials.

		Call:
			data = p.opt([T, solver, check])
		Input:
			solver [optional, default 'sedumi'/'ECOS']: solver, to solve the problem, currenty possible:
				Matlab: 'sedumi', 'sdpt3'
				Python: 'ECOS', 'scs', 'cvxopt'
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
		if set(self.monomial_squares) - set(self.non_squares) != set() and aux.VERBOSE:
			warnings.warn('Unused positive points.')

		t0 = datetime.now()
		if language == 'python':
			return self.sonc_opt_python()

		elif language == 'matlab':
			self._ensure_matlab()
			if solver is None: solver = 'sedumi'
			b_relax = self.b.copy()
			b_relax[monomial_squares] = abs(self.b[self.monomial_squares])
			b_relax[non_squares] = -abs(self.b[self.non_squares])

			self.matlab.cvx_solver(solver)
			matlab_result = self.matlab.opt_sonc_split_outer(matlab.double(self.A.tolist()), matlab.double(b_relax.tolist()), [[e + 1 for e in l] for l in self.cover], matlab.double(self.coefficient_distribution.todense().tolist()), nargout = 5)

			try:
				data = {}
				data['opt'], data['C'], data['solver_time'], data['status'], data['verify'] = matlab_result
				#C = np.array(data['C']).round(aux.DIGITS)
				#data['C'] = self.__matrix_list_to_sparse([C[k, :len([e for e in self.cover[k] if e in self.monomial_squares]), :(len([e for e in self.cover[k] if e >= self.monomial_squares]))] for k in range(len(self.cover))])
			except Exception as err:
				data = aux.FAULT_DATA.copy()
				data['result'] = matlab_result
				data['error'] = repr(err)

			data['modeler'] = 'cvx'
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
			data['params'] = {}
		data['params']['cover'] = self.cover.copy()
		data['params']['distribution'] = self.coefficient_distribution.todense()
		self._store_solution(data)
		return data

	def run_sos(self):
		"""Run all available SOS-methods to compute a lower bound.

		If Matlab is available (if matrix-size <= 400):
			* Sostools: with and without 'sparse' flag
			* Gloptipoly
			* Yalmip: using 'solvesos' and 'sparsepop'
			* own implementation (only if <= 8000 constraints): using 'sedumi' and 'sdpt3'
		Python (if matrix-size <= 120):
			* cvxopt
			* MOSEK
		"""
		sos_size = binomial(self.A.shape[0] - 1 + self._degree//2, self._degree//2)
		sos_size_constraints = binomial(self.A.shape[0] - 1 + self._degree, self._degree)
		#saveguards for sos, where problem would crash RAM
		if sos_size <= 400 and matlab_found:
			self.sostools_opt()
			self.sostools_opt(sparse = True)
			self.gloptipoly_opt()
			self.yalmip_opt(method = 'solvesos')
			self.yalmip_opt(method = 'sparsepop')
			if sos_size_constraints <= 8000:
				self.sos_opt_matlab()
				self.sos_opt_matlab(solver = 'sdpt3')
		#Python takes more RAM, so stricter saveguard
		if sos_size <= 120:
			self.sos_opt_python(solver = cvx.CVXOPT)
			self.sos_opt_python(solver = cvx.MOSEK)

	def run_sonc(self):
		"""Run SONC with two different cover strategies.

		* cover, that maximises use of constant term
		* cover, that uses all monomial squares

		If both strategies yields the same cover, we run SONC only once.
		"""
		if self.trivial_check():
			return
		self._compute_zero_cover()
		self.sonc_opt_python()
		self._compute_cover()
		self.sonc_opt_python()

	def run_all(self, keep_alive = False, call_sos = True, clear = True):
		"""Run all optimisation methods which are currently implemented.

		The results are stored in self.old_solutions.
		SOS is called only if <= 400 monomials

		Current list of methods:
		- sostools
		- gloptipoly
		- sos-cvx in Matlab, using sedumi
		- sos-cvx in Matlab, using sdpt3
		- sos-cvx in Python using cvxopt (if matrix-size <= 120)

		- sonc-cvx in Matlab, using sedumi
		- sonc-cvx in Matlab, using sdpt3
		- sonc-cvx in Python using ECOS

		If the Matlab engine was not found, only Python is run.
		"""
		#Handle trivial case of only monomial squares
		self.clean()
		if call_sos:
			self.run_sos()
		#Call SONC/SAGE
		#self.opt()
		self.run_sonc()
		self.sage_opt_python()
		if matlab_found:
			self.opt(method = 'even', language = 'matlab')
			if np.prod(self.A.shape) < 3000:
				self.opt(method = 'outer', language = 'matlab')
		if not keep_alive and matlab_found:
			self.matlab.exit()
		if clear:
			self.clear()

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

	def get_decomposition(self, symbolic = False):
		"""Return a decomposition into non-negative polynomials, according to currently stored solution.

		Note: SAGE not implemented, yet.
		Note: If in SONC, there are points, which are not covered with the constant term,
			the solution in exact arithmetic may still be slightly negative.

		Call:
			decomposition = p.get_decomposition([symbolic])
		Input:
			symbolic [optional, default False]: boolean, whether to create a solution in exact arithmetic.
				Currently only implemented for SONC.
		Output:
			decomposition: list of polynomials
		"""
		if self.solution is None: return None
		if self.solution['strategy'] == 'sos':
			if symbolic:
				raise NotImplementedError('Symbolic SOS decomposition not implemented.')
			else:
				return self._get_sos_decomposition()
		elif self.solution['strategy'] == 'sonc':
			if symbolic:
				return self._symbolic_sonc_decomposition()
			else:
				cert = []
				for k in range(len(self.cover)):
					cert.append(CircuitPolynomial(self.A[:, self.cover[k]], np.concatenate((self.solution['C'][k, self.cover[k][:-1]].todense() * (-1)**(self.b[self.cover[k][:-1]] < 0), [self.coefficient_distribution[k, self.cover[k][-1]]])), orthant = self.orthant))
				return cert
		elif self.solution['strategy'] == 'sage':
			if symbolic:
				return self._symbolic_sage_decomposition()
			else:
				C = self.solution['C'].round(aux.DIGITS)
				lamb = self.solution['lambda'].round(aux.DIGITS)
				for i in range(C.shape[0]):
					C[i,i] *= -1
					lamb[i,i] *= -1
				return [AGEPolynomial(self.A, C[k,:], lamb = lamb[k,:], is_symbolic = False, orthant = np.ones(self._variables, dtype = np.int)) for k in range(self.A.shape[1])]
		else:
			raise Exception('Unknown strategy')

	def _symbolic_sonc_decomposition(self):
		"""Return a list of Circuit-Polynomials which form a decompositoin of the relaxed polynomial.

		All of these summands have rational coefficients.
		They are continued-fraction-approximations of the numerical results.

		Note: May still fail, if there are degenerate points.
		"""
		if self.solution is None or self.solution['strategy'] != 'sonc':
			return None
		C = self.solution['C']
		M = sympy.Matrix(np.zeros(C.shape, dtype = np.int))
		#rounding the b[i] will not be necessary, if they are considered exact integers
		b = [aux.to_fraction(sq, bound = -1) for sq in self.b[self.monomial_squares]]
		epsilon = aux.EPSILON/2/self.A.shape[0]/len(self.cover)
		for k in range(len(self.cover)):
			M[k,self.cover[k][-1]] = -aux.to_fraction(-self.coefficient_distribution[k,self.cover[k][-1]], eps = epsilon, bound = 1)
			if 0 in self.cover[k]:
				lamb = sympy.linsolve((sympy.Matrix(self.A[:,self.cover[k][:-1]]), sympy.Matrix(self.A[:,self.cover[k][-1]])), [x[i] for i in range(self.A.shape[0])])
				lamb = next(iter(lamb))
				for i in range(1,len(self.cover[k]) - 1):
					M[k,self.cover[k][i]] = aux.to_fraction(C[k,self.cover[k][i]], eps = epsilon * lamb[0]/lamb[i], bound = -1)
				M[k,0] = aux.to_fraction(lamb[0] * (-M[k,self.cover[k][-1]] * sympy.prod([(lamb[r] / M[k,self.cover[k][r]]) ** lamb[r] for r in range(1,len(self.cover[k]) - 1)])) ** (1/lamb[0]), bound = 1)
			else:
				for i in self.cover[k][:-1]:
					M[k,i] = aux.to_fraction(C[k,i], bound = 1)

		return [CircuitPolynomial(self.A[:,self.cover[k]], np.array(M[k,self.cover[k]])[0], orthant = self.orthant) for k in range(len(self.cover))]

	def _symbolic_sage_decomposition(self):
		"""Return a list of Circuit-Polynomials which form a decomposition of the relaxed polynomial.

		All of these summands have rational coefficients.
		They are continued-fraction-approximations of the numerical results.

		Note: May still fail, if there are degenerate points.
		"""
		if self.solution is None or self.solution['strategy'] != 'sage':
			return None
		frac = np.vectorize(aux.to_fraction, cache = True)
		box_v = np.vectorize(aux.get_box, cache = True)

		C = self.solution['C']
		C_sy = np.zeros(C.shape, dtype = object)
		lamb = self.solution['lambda']
		for i in self.non_squares:
			C[i,i] *= -1
			C_sy[i,:] = frac(C[i,:])
			lamb[i,i] *= -1

		lamb_sy = np.zeros(lamb.shape, dtype = object)

		box_all = box_v(lamb, 16)
		for i in self.non_squares:
			box = list(zip(box_all[0][i], box_all[1][i]))
			lamb_sy[i,:] = aux.LP_solve_exact(self.A, np.zeros(self.A.shape[0], dtype = np.int), box = box)

		C_sy[self.monomial_squares,0] = 0

		C_sy *= np.array(np.array(self.relax().b, dtype = np.int) / C_sy.sum(axis = 0))

		decomp = []

		for j in self.non_squares:
			idx = [i for i in range(1, C_sy.shape[0]) if i != j and lamb_sy[j,i] != 0]
			C_sy[j,0] = aux.to_fraction(sympy.exp(aux.to_fraction((aux.symlog(lamb_sy[j,idx] / C_sy[j,idx]) * lamb_sy[j,idx] - lamb_sy[j,idx]).sum() + lamb_sy[j,0]*sympy.log(lamb_sy[j,0]) - lamb_sy[j,0] - C_sy[j,j], bound = 1) / lamb_sy[j,0]), bound = 1)
			decomp.append(AGEPolynomial(self.A, C_sy[j,:], lamb = lamb_sy[j,:], orthant = np.ones(self._variables, dtype = np.int)))
		return decomp

	def trivial_check(self):
		"""Check whether p is a sum of monomial squares."""
		self.clean()
		if self.is_sum_of_monomial_squares():
			self.min = (self.b[0], np.zeros(self.A.shape[0] - 1))
			data = { 'time': 0, 'language': 'python', 'solver': 'trivial', 'modeler': 'trivial', 'strategy': 'trivial', 'status': 1, 'verify': 1, 'params': {}, 'C': np.array([]), 'opt': -self.b[0] }
			self._store_solution(data)
			return True
		else:
			return False

	def is_sum_of_monomial_squares(self, eps = 0):
		"""Check whether the polynomial is a sum of monomial squares.

		Call:
			res = p.is_sum_of_monomial_squares([eps = 0])
		Input:
			eps [optional, default 0]: accuracy, up to which value a coefficient is treated as zero.
		Output:
			res: bool, whether the polynomial is a sum of monomial squares (up to given accuracy)
		"""
		self.clean()
		return self.non_squares == [] or not (abs(self.b[self.non_squares]) > eps).any()

	# === Covering the polynomial ===

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
			covering_vertices = []
			for m in range(len(self.monomial_squares)):
				c = np.zeros(len(self.monomial_squares))
				c[m] = -1
				res = scipy.optimize.linprog(c, A_eq = self.A[:,self.monomial_squares], b_eq = self.A[:,deg])
				if res.fun < -aux.EPSILON:
					covering_vertices.append(self.monomial_squares[m])
			#for each point, check whether it lies in the face given by covering_vertices
			#This happens iff adding the vector does not increase the rank.
			rank = np.linalg.matrix_rank(self.A[:,covering_vertices])
			indices = [np.linalg.matrix_rank(self.A[:, covering_vertices + [i]]) == rank for i in range(self.A.shape[1])]
			q = Polynomial(self.A[:,indices], self.b[indices])
			x_min = scipy.optimize.fmin(q, np.sign(self.b[deg]) * (-1)**(q.A[1:,:].sum(axis = 1) % 2), disp = False)
			if q(x_min) < -aux.EPSILON:
				return x_min, [i for i in range(self.A.shape[1]) if indices[i]]
		return None

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
		#init
		U_index = set(self.non_squares)
		T = []
		#setting up the LP
		A_eq = self.A[:,self.monomial_squares]
		X = cvx.Variable(len(self.monomial_squares), nonneg = True)

		#continue, until all points are covered
		while U_index != set():
			ui = U_index.pop()
			if aux.VERBOSE > 1:
				print('covering index %d' % ui)
			#find vertices covering u
			prob = cvx.Problem(cvx.Minimize(-X[0]), [A_eq * X == self.A[:,ui]])
			res = prob.solve(solver = 'GLPK_MI')
			if res == np.inf:
				raise InfeasibleError('Polynomial is unbounded at point %d.' % ui)
			T_index = [self.monomial_squares[i] for i in range(len(self.monomial_squares)) if X.value[i] > aux.EPSILON**2]
			#update target vector
			#get all points covered by T_index
			T_index = polytope._get_inner_points(self.A, self.non_squares, T_index)
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
		#init. V are the monomial squares
		V_index = set(self.monomial_squares)
		U_index = set(self.non_squares)
		T = []
		#setting up the LP
		A_eq = self.A[:,self.monomial_squares]
		X = cvx.Variable(len(self.monomial_squares), nonneg = True)
		c = np.ones(len(self.monomial_squares))

		#cover U
		while U_index != set():
			ui = U_index.pop()
			if aux.VERBOSE > 1:
				print('covering point: ', ui)
			#find vertices covering u
			prob = cvx.Problem(cvx.Minimize(-cvx.sum(cvx.multiply(c,X))), [A_eq * X == self.A[:,ui]])
			res = prob.solve(solver = 'GLPK_MI')
			if res == np.inf:
				raise InfeasibleError('Polynomial is unbounded at point %d.' % ui)
			T_index = []
			for i in range(len(self.monomial_squares)):
				if X.value[i] > aux.EPSILON**2:
					T_index.append(self.monomial_squares[i])
					c[i] = 0
			#get all points covered by T_index
			T_index = [ui] + polytope._get_inner_points(self.A, U_index, T_index)
			T_index.sort()
			#mark covered points
			U_index -= set(T_index)
			V_index -= set(T_index)
			T.append(T_index)
		#cover V
		U_index = set(self.non_squares)
		while V_index != set():
			if aux.VERBOSE > 1:
				print('Still to use: ', V_index)
			change = False
			for ui in U_index:
				#find vertices covering u
				prob = cvx.Problem(cvx.Minimize(-cvx.sum(cvx.multiply(c,X))), [A_eq * X == self.A[:,ui]])
				prob.solve(solver = 'GLPK_MI')
				if prob.value < -aux.EPSILON:
					T_index = []
					for i in range(len(self.monomial_squares)):
						if X.value[i] > aux.EPSILON**2:
							T_index.append(self.monomial_squares[i])
							c[i] = 0
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
		if self.cover is not None and aux.VERBOSE:
			warnings.warn('Overwriting cover.')
		if split:
			new_cover = []
			for c in T:
				squares = [e for e in c if e in self.monomial_squares]
				for e in c:
					if e in self.non_squares:
						new_cover.append(squares + [e])
			self.old_covers[hash(str(new_cover))] = new_cover
			self.cover = new_cover
		else:
			self.old_covers[hash(str(T))] = T.copy()
			self.cover = T

		#self._reduce_cover()

		#store convex combinations
		self.lamb = scipy.sparse.dok_matrix((len(self.cover), self.A.shape[1]))
		for k in range(len(self.cover)):
			self.lamb[k,self.cover[k][:-1]] = linsolve(self.A[:,self.cover[k][:-1]], self.A[:,self.cover[k][-1]])
		self.lamb = self.lamb.toarray()

		#ensure that coefficient_distribution is defined
		self._set_coefficient_distribution()

		#update optimisation problem
		if self.prob_sonc is not None:
			self._create_sonc_opt_problem()

	def _reduce_cover(self):
		counter = np.zeros(self.A.shape[1])
		for circuit in self.cover:
			for node in circuit:
				counter[node] += 1
		i = 0
		while i < len(self.cover):
			if (counter[self.cover[i]] > 1).all():
				circuit = self.cover.pop(i)
				counter[circuit] -= 1
			else:
				i += 1

	# === Constrained case ===




	# === Computation of Local Minima ===

	def local_min(self, method = 'sonc', **kwargs):
		"""Compute a local minimum, according to the given method.

		Call:
			fmin, xmin = p.local_min([method,**kwargs])
		Input:
			method: which method to use
				- random: gradient method with random starting points
					see p._local_min_random()
				- sonc: single call of gradient method, starting form barycentre of SONC-minima
					see p._local_min_sonc()
				- differential_evolution: scipy
					see p._local_min_differential_evolution()
			**kwargs: dictionary of keyword-arguments, handled to the minimisation method
		Output:
			fmin: minimal value found
			xmin: argument, where fmin is attained
		"""
		self.clean()
		if self.trivial_check():
			return self.min
		if method == 'all':
			self._local_min_random()
			self._local_min_sonc()
			self._local_min_differential_evolution()
			return self.min
		elif method == 'random':
			return self._local_min_random(**kwargs)
		elif method == 'sonc':
			return self._local_min_sonc()
		elif method == 'differential_evolution':
			return self._local_min_differential_evolution(**kwargs)
		else:
			raise Exception('Unknown method for minimisation')

	def _local_min_differential_evolution(self, **kwargs):
		"""Compute a minimum via differential-evolution from SciPy."""
		max_val = abs(self.b).max()
		bounds = []
		for sign in self.orthant:
			if sign == -1: bounds.append((-max_val,0))
			if sign ==  1: bounds.append((0,max_val))
			if sign ==  0: bounds.append((-max_val,max_val))
		local_min = scipy.optimize.differential_evolution(self, bounds, **kwargs)
		if local_min.fun < self.min[0]:
			self.min = (local_min.fun, local_min.x)
		return (local_min.fun, local_min.x)

	def _local_min_random(self, max_iters = 10):
		"""Give an upper bound to the minimum by guessing starting values and computing the local minima.

		The method runs <max_iters> iterations:
			Generate a normally distributed start point, scaled by the maximal coefficient.
			Then run conjugate gradient algorithm to obtain a local minimum.
			The lowest value found is returned.

		Call:
			fmin, xmin = p._local_min_random()
		Input:
			max_iters [optional, default 50] - number of iterations
		Output:
			fmin - float, least value, that was found for polynomial p
			xmin - float-array, argument where fmin is attained
		"""
		#self.__call__ is clearer than just 'self'
		#xmin = scipy.optimize.fmin(self.__call__, np.zeros(self.A.shape[0] - 1), disp = False)
		xmin = scipy.optimize.fmin_cobyla(self, np.zeros(self.A.shape[0] - 1), lambda x: x*self.orthant)
		fmin = self.__call__(xmin)
		if max_iters is None:
			max_iters = np.prod(self.A.shape)
		#p_derivative = self.prime(variables = self.A.shape[0] - 1)
		#pprime = (lambda arg: np.array([p_i(arg) for p_i in p_derivative]))

		for _ in range(max_iters):
			x0 = self.b.max() * np.random.randn(self.A.shape[0] - 1)
			#TODO: one-liner
			for i in range(len(x0)):
				if np.sign(x0[i]) != self.orthant[i]:
					x0[i] *= -1
			#xmin_tmp = scipy.optimize.fmin_cg(self.__call__, x0, fprime = pprime, disp = False)
			xmin_tmp = scipy.optimize.fmin_cobyla(self, x0, lambda x: x*self.orthant)
			val = self.__call__(xmin_tmp)
			if val < fmin:
				fmin = val
				xmin = xmin_tmp
		if self.min[0] > fmin:
			self.min = (fmin, xmin)
		return fmin, xmin

	def _local_min_sonc(self):
		"""Give an upper bound to the minimum, based on the SONC decomposition.

		The method computes the SONC decomposition.
		Then it computes the minimum of each circuit polynomial by an explicit formula.
		The barycentre of these minima is then used as start point for a conjugate gradient algorithm on the polynomial.

		Call:
			fmin, xmin = p._sonc_min()
		Output:
			fmin - float, least value, that was found for polynomial p
			xmin - float-array, argument where fmin is attained
		"""
		fmin, xmin = np.inf, np.array([np.inf for _ in range(self.A.shape[0] - 1)])
		if self.solution is None or 'sonc' not in [sol['strategy'] for sol in self.old_solutions.values()]:
			self.run_sonc()

		p_derivative = self.prime(variables = self.A.shape[0] - 1)
		pprime = (lambda arg: np.array([p_i(arg) for p_i in p_derivative]))

		#save current solution, to restore it afterwards
		solution_restore = self.solution

		#check all SONC-solutions with finite value
		for solution in [sol for sol in self.old_solutions.values() if sol['strategy'] == 'sonc' and sol['verify'] == 1 and sol['opt'] < np.inf]:
			self.solution = solution
			self.set_cover(solution['params']['cover'])
			l = self.get_decomposition()
			minima = np.array([q.minimiser() for q in l])
			barycentre = minima.sum(axis = 0) / len(l)
			#return scipy.optimize.fmin_cg(self, barycentre, fprime = pprime, disp = False)
			p_relax = self.relax()
			min_relax = scipy.optimize.fmin_cg(p_relax, barycentre, fprime = pprime, disp = False)
			for i in range(len(min_relax)):
				if self.orthant[i] == 1: min_relax[i] = abs(min_relax[i])
				if self.orthant[i] == -1: min_relax[i] = -abs(min_relax[i])
			#xmin = scipy.optimize.fmin_cg(self, min_relax, fprime = pprime, disp = False)
			xmin = scipy.optimize.fmin_cobyla(self, min_relax, lambda x: x*self.orthant)
			fmin = self(xmin)

			if self.min[0] > fmin:
				self.min = (fmin, xmin)

		self.solution = solution_restore

		return fmin, xmin

	def gap(self):
		"""Return the gap between lower bound and found minimum."""
		return self.min[0] - self.lower_bound

	#=== Z3 ===

	def z3_nonnegative(self):
		"""Check nonnegativity via Z3.

		Call:
			nonneg = p.z3_nonnegative()
		Output:
			nonneg [bool] - whether the polynomial is nonnegative
		"""
		if not z3_found:
			raise ModuleNotFoundError('Z3 not installed')
		def prod(l):
			res = 1
			for e in l:
				res *= e
			return res
		variables = [z3.Real('x%d' % i) for i in range(self._variables)]
		solver = z3.Solver()
		solver.append(z3.ForAll(variables, sum([self.b[j] * prod([variables[i] ** self.A[i+1,j] for i in range(self._variables) ]) for j in range(self.A.shape[1])]) >= 0))
		return str(solver.check()) == 'sat'

if __name__ == "__main__":
	pass
	#p = Polynomial('standard_simplex',30, 60, 100, seed = 0)
	##collecting further nice examples
	#example4_2 = Polynomial(str(8*x(0)**6 + 6*x(1)**6 + 4*x(2)**6+2*x(3)**6 -3*x(0)**3*x(1)**2 + 8*x(0)**2*x(1)*x(2)*x(3) - 9*x(1)*x(3)**4 + 2*x(0)**2*x(1)*x(3) - 3*x(1)*x(3)**2 + 1))
	#example_small = Polynomial('general',4,8,8,3,seed = 0)
	#ex1 = Polynomial('general',10,20,100,80,seed = 1)
