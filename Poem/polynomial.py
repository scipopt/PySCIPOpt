#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Class for multivariate polynomials in sparse notation, focus on optimisation."""

import numpy as np
from datetime import datetime
import itertools
import warnings

import sparse
import pymp

import Poem.aux as aux
import Poem.polynomial_opt as polynomial_opt

np.set_printoptions(linewidth = 200)

__version__ = '0.2.0.1'

class Polynomial(polynomial_opt.Polynomial):
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
		if 'parent' in kwargs.keys():
			self.parent = kwargs['parent']
		else:
			self.parent = None

		if 'ancestor' in kwargs.keys():
			self.ancestor = kwargs['ancestor']
		else:
			self.ancestor = self
		if self.ancestor is self:
			self.active_children = [self]
			self.depth = None

		# -- set further defaults --
		self.child0 = None
		self.child1 = None
		self.sibling = None

		self.init_time += aux.dt2sec(datetime.now() - t0)

	def _store_solution(self, data):
		"""Store new solution, but keep the previous ones for different methods.

		In addition to the inherited method, an improvement of the lower bound is propagated upwards.
		"""
		super()._store_solution(data)
		#update lower bound
		if data['verify'] == 1 and self.lower_bound < -data['opt']:
			self.lower_bound = -data['opt']
			if self.parent is not None:
				self.parent._propagate_bound_up()

	# === Branching over the Orthants ===
	## ==== Running only minimal Orthants ====

	def _fork(self):
		"""Get list of polynomials, corresponding to the minimal orthants."""
		self._minimal_signs()
		return [Polynomial(self.A, self.b, orthant = orthant) for orthant, _ in self.minimal_signs]

	def _minimal_signs(self, sort = True):
		"""Compute a list of (sign, coefficients), such that the coefficient vectors are minimal.
		
		Call:
			p._minimal_signs([sort])
		Input:
			sort [optional, boolean, default True] - flag, whether to sort the list of sign vectors (by their number of negative terms);
				may affect the running time
		Output:
			None, the result is stored in p.minimal_signs
		"""
		self.clean()
		#determine the minimal orthants
		base_sign = np.array(self.b[self.non_squares] < 0, dtype = np.int)
		sign_list = [((-1)**np.array(comb), (np.dot(np.array(comb), self.A[1:,self.non_squares]) + base_sign) % 2) for comb in itertools.product([0,1], repeat=self.A.shape[0] - 1)]
		if sort:
			self.minimal_signs = aux.maximal_elements(sign_list, lambda u,v: (u[1] <= v[1]).all(), sort_key = lambda v: v[1].sum())
		else:
			self.minimal_signs = aux.maximal_elements(sign_list, lambda u,v: (u[1] <= v[1]).all(), sort_key = None)

	def forked_bound(self, strategy = 'sonc', parallel = True):
		"""Give an improved bound by computing a bound for each minimal orthant.

		Call:
			bound = p.forked_bound([strategy, parallel])
		Input:
			strategy[optional, string, default 'sonc']: which method to use for the lower bounds
				- 'sonc': see self.sonc_opt_python(), fastest
				- 'sage': see self.sage_opt_python()
				- 'all': see self.run_all(), best result
			parallel[optional, boolean, default True]: flag, whether to run the computations in parallel
		Output: 
			bound: float, the smallest bound found over the minimal orthants
		"""
		self.clean()
		t0 = datetime.now()
		polys = self._fork()
		if parallel:
			result_list = pymp.shared.list()
			with pymp.Parallel() as env:
				for index in env.range(len(polys)):
					q = polys[index]
					if strategy == 'sonc': q.run_sonc()
					if strategy == 'sage': q.sage_opt_python()
					if strategy == 'all' : q.run_all(call_sos = False)
					result_list.append((q.lower_bound, max([sol['verify'] for sol in q.old_solutions.values()])))
		else:
			result_list = []
			for q in polys:
				if strategy == 'sonc': q.run_sonc()
				if strategy == 'sage': q.sage_opt_python()
				if strategy == 'all' : q.run_all(call_sos = False)
				result_list.append((q.lower_bound, max([sol['verify'] for sol in q.old_solutions.values()])))

		fork_bound = min([entry[0] for entry in result_list])
		self.lower_bound = max(self.lower_bound, fork_bound)
		self.solution_time = aux.dt2sec(datetime.now() - t0)
		self._store_solution({'strategy' : 'fork', 'opt' : -fork_bound, 'time' : self.solution_time, 'params' : { 'strategy' : strategy, 'parallel': parallel }, 'language' : 'python', 'verify' : min([entry[1] for entry in result_list]), 'modeler': 'cvxpy', 'solver' : 'ECOS'})
		return self.lower_bound

	## ==== Branch and Bound ====

	def _create_search_tree(self, sort = True):
		"""Create a search tree, such that we only compute nodes leading to minimal orthants.
		
		For each node, it creates children accordingly, such that the leaves correspond to the minimal orthants.
		This avoids computing unnecessary branches.

		Warning: This deletes an existing search tree.
		"""
		#delete existing tree
		if self.child0 is not None or self.child1 is not None:
			warnings.warn('Overwriting search tree')
		self.child0 = None
		self.child1 = None
		self._minimal_signs(sort = sort)
		n = self.A.shape[0] - 1

		#insert leaves into search tree, creating inner nodes as required
		for orth, _ in self.minimal_signs:
			curr = self
			for i in range(n):
				curr_orthant = np.concatenate((orth[:i+1], np.zeros(n-i-1, dtype = np.int)))
				if orth[i] == -1:
					if curr.child0 is None:
						curr.child0 = Polynomial(self.A, self.b, orthant = curr_orthant, ancestor = self.ancestor, parent = curr)
						if curr.child1 is not None:
							curr.child0.sibling = curr.child1
					curr = curr.child0
				else:
					if curr.child1 is None:
						curr.child1 = Polynomial(self.A, self.b, orthant = curr_orthant, ancestor = self.ancestor, parent = curr)
						if curr.child0 is not None:
							curr.child1.sibling = curr.child0
					curr = curr.child1

	def _branch(self, reltol = aux.EPSILON, abstol = aux.EPSILON, verbose = aux.VERBOSE):
		"""Compute a lower bound via branch-and-bound.

		This function may only be called if both children do NOT exist,
			then they are created, or both exist.

		Call:
			p._branch([reltol, abstol, verbose])
		Input:
			abstol [optional, default: aux.EPSILON]: absolute tolerance, 
				branching stops, when duality gap is below this value
			reltol [optional, default: aux.EPSILON]: relative tolerance
				branching stops, when duality gap is below this value
			verbose [optional, boolean, default: aux.VERBOSE]: set verbosity
		"""
		gap = self.ancestor.min[0] - self.lower_bound
		depth = self.ancestor.depth - (self.orthant != self.ancestor.orthant).sum()
		#compare gap to min instead of lower_bound, since inf/inf = nan does not compare
		if gap > abstol and abs(gap / self.ancestor.min[0]) > reltol and depth > 0:
			#if we have not pre-created the search-tree, create both children
			if self.child0 is None and self.child1 is None:
				##choose index of variable
				#choose unknown variable with highest number of odd powers
				index = ((self.A[1:,self.non_squares] % 2).sum(axis = 1) * (1 - abs(self.orthant))).argmax()
				#If all even, then no further branching possible.
				if not (self.A[index + 1,:] % 2).any() or self.orthant[index] != 0:
					return
				##pick first variable of unknown sign
				#index = np.where(self.orthant == 0)[0][0]

				##fork
				orthant = self.orthant.copy()
				self.ancestor.max_size -= 2

				orthant[index] = -1
				self.child0 = Polynomial(self.A, self.b, orthant = orthant, ancestor = self.ancestor, parent = self)

				orthant[index] = 1
				self.child1 = Polynomial(self.A, self.b, orthant = orthant, ancestor = self.ancestor, parent = self)

				self.child1.sibling = self.child0
				self.child0.sibling = self.child1

			#descends further down, if single children, returns the freshly computed nodes
			c0 = self.child0._run_child(verbose = verbose)
			c1 = self.child1._run_child(verbose = verbose)
			self.ancestor.active_children.extend([c0, c1])
			c0.parent._propagate_bound_up()
			c1.parent._propagate_bound_up()
		elif aux.VERBOSE and depth > 0:
			print('Cut at depth %d' % depth)

	def _run_child(self, verbose = aux.VERBOSE):
		"""Call SONC with two covers and compute minima.

		This method is supposed to be called from the parent for both children.

		If we do not get additional monomial squares, we cannot improve the bound.
		In this case, we copy the values of the parent.
		Otherwise, we process the inherited solution and compute SONC solutions from scratch.

		Call:
			child = p.child0._run_child([verbose])
		Input:
			verbose [boolean, optional, default aux-VERBOSE]: verbosity flag
		Output:
			child [Polynomial]: the descendant, that was actually processed
				(procedure descends, as long as the current node has a single child)
		"""
		self.clean()
		if verbose:
			print('running orthant %s' % str(self.orthant))
		self.lower_bound = self.parent.lower_bound

		#if only one child, then descend further
		if self.child0 is not None and self.child1 is None:
			return self.child0._run_child()
		if self.child1 is not None and self.child0 is None:
			return self.child1._run_child()

		if set(self.monomial_squares) > set(self.parent.monomial_squares) or self.sibling is None:
			if self.trivial_check():
				return self
			self._inherit_solution()
			self.run_sonc()
			self.local_min(method = 'all')
			self.clear()
			#update minimum of ancestor
			if self.min[0] < self.ancestor.min[0]:
				self.ancestor.min = self.min
		else:
			self.old_solutions = self.parent.old_solutions
		return self
	
	def traverse(self, strategy = 'min', depth = None, max_size = None, reltol = aux.EPSILON, abstol = aux.EPSILON, call_sos = True, verbose = aux.VERBOSE, sparse = False):
		"""Improve the lower bound of the polynomial by branching over the signs of the variables.

		Call:
			p.traverse([strategy, depth, max_size, reltol, abstol])
		Input:
			strategy [string, default 'min']: strategy in which order to traverse the tree
				- 'min': choose node with minimal lower bound
				- 'dfs': depth-first search
				- 'bfs': breadth-first search
			depth [optional, default no limit]: maximal depth of the search tree
				mainly useful for 'bfs'/'dfs'
			max_size [optional, default no limit]: maximal size (number of nodes) of the search tree
				mainly useful for 'min'
			abstol [optional, default: aux.EPSILON]: absolute tolerance, 
				branching stops, when duality gap is below this value
			reltol [optional, default: aux.EPSILON]: relative tolerance
				branching stops, when duality gap is below this value
			call_sos [optional, default: True]: flag, whether to call SOS on the root
		"""
		if self.ancestor is not self:
			raise Exception('Traverse should only be called on ancestor.')
		if depth is None or depth > self.A.shape[0] - 1:
			depth = (self.orthant == 0).sum()
		self.depth = depth
		if max_size == None or max_size >= 2**self.A.shape[0]:
			max_size = 2**self.A.shape[0] - 1
		self.max_size = max_size - 1 #subtract 1 for root as initial node

		t0 = datetime.now()

		if sparse:
			self._create_search_tree()

		#decend into tree, if root has single child
		curr = self
		change = True
		while change:
			change = False
			if curr.child0 is not None and curr.child1 is None:
				curr = curr.child0
				change = True
			if curr.child1 is not None and curr.child0 is None:
				curr = curr.child1
				change = True
		self.active_children = [curr]
		#call all methods on this first node with two/no children
		curr.run_all(call_sos = call_sos)
		if curr.parent is not None:
			curr.parent._propagate_bound_up()
		curr.local_min(method = 'all')
		if curr.min[0] < curr.ancestor.min[0]:
			curr.ancestor.min = curr.min

		self.old_solutions = curr.old_solutions.copy()

		#_traverse inherits current bound, so we do not have to initialise with the current bound
		while self.active_children != []:
			self._traverse(strategy, reltol = reltol, abstol = abstol, verbose = verbose)

		self.solution_time = aux.dt2sec(datetime.now() - t0)
		self._store_solution({'strategy' : 'traverse', 'opt' : -self.lower_bound, 'time' : self.solution_time, 'params' : { 'strategy' : strategy, 'max_size' : max_size, 'reltol' : reltol, 'abstol' : abstol, 'depth' : depth, 'sparse': sparse}, 'language' : 'python', 'verify' : 1, 'modeler': 'cvxpy', 'solver' : 'all'})
		return self.lower_bound

	def _traverse(self, strategy, reltol = aux.EPSILON, abstol = aux.EPSILON, verbose = aux.VERBOSE):
		#pick next node to improve, according to the chosen strategy
		if strategy == 'dfs':
			#DFS = last entry of list
			curr_index = -1
		elif strategy == 'bfs':
			#BFS = first entry of list
			curr_index = 0
		elif strategy == 'min':
			#pick the polynomial which has the worst lower bound
			curr_index = 0
			for i in range(1,len(self.active_children)):
				if self.active_children[i].lower_bound < self.active_children[curr_index].lower_bound:
					curr_index = i
		else:
			raise Exception('Unknown strategy.')

		#Note: in the first iteration, curr will be root/ancestor
		curr = self.active_children[curr_index]

		improvable = True
		iterator = curr
		while iterator.parent is not None:
			if iterator.sibling is not None and iterator.lower_bound > iterator.sibling.lower_bound:
				improvable = False
				break
			iterator = iterator.parent

		#first try sage, otherwise branch
		if not improvable:
			#TODO: check whether we can return in this case; paper says we can
			self.active_children.pop(curr_index)
		elif 'sage' not in [key[1] for key in self.active_children[curr_index].old_solutions.keys()]:
			self.active_children[curr_index].sage_opt_python()
			self.active_children[curr_index].clear()
			#ancestor had SAGE already called, so it is safe to call self.parent
			self.active_children[curr_index].parent._propagate_bound_up()
		else:
			self.active_children.pop(curr_index)
			if self.max_size >= 2:
				curr._branch(reltol = reltol, abstol = abstol, verbose = verbose)

	def _propagate_bound_up(self):
		"""Check the lower bounds of the children an update self; then recurse on parent."""
		if self.child0 is None:
			if self.child1 is None:
				return
			else:
				low = self.child1.lower_bound
		else:
			if self.child1 is None:
				low = self.child0.lower_bound
			else:
				low = min(self.child0.lower_bound, self.child1.lower_bound)
		if low > self.lower_bound:
			self.lower_bound = low
			if self.parent is not None:
				self.parent._propagate_bound_up()

	def _inherit_solution(self):
		"""Inherit cover from nearest predecessor and run SONC."""
		#find parent-cover with best solution
		t0 = datetime.now()
		curr = self.parent
		while 'sonc' not in [sol['strategy'] for sol in curr.old_solutions.values() if sol['verify'] == 1]:
			curr = curr.parent
			if curr is None:
				return
		sol = min([sol for sol in curr.old_solutions.values() if sol['strategy'] == 'sonc' and sol['verify'] == 1], key = lambda sol: sol['opt'])
		self.set_cover(sol['params']['cover'])
		##TODO: inherit full solution, not just cover
		##TODO: avoid full intermediate array
		#C = []
		#for c in self.cover:
		#	for i in range(len(sol['params']['cover'])):
		#		if c == sol['params']['cover'][i]:
		#			C.append(sol['C'][i,:])
		#C = sparse.COO.from_numpy(C)
		#data = {'C': C, 'opt': sum(C[:,0]) - self.b[0], 'status': 1, 'verify': 1, 'solver_time': 0, 'solver': 'trivial', 'strategy': 'trivial', 'language': 'python'}
		#data['time'] = aux.dt2sec(datetime.now() - t0)
		#self._store_solution(data)

		#solve SONC with the above cover
		self.sonc_opt_python()

	#=== Auxiliary functions for the search tree ===

	def _tex_tree(self):
		"""Return LaTeX-code for the search tree of branch()."""
		res = ''
		stats = ''
		if self.ancestor is self: 
			res += '\\Tree'
			stats += '\\quad\\substack{t = \\timing\\\\s = %d}' % self._tree_size()
		if self.lower_bound - self.ancestor.min[0] > np.sqrt(aux.EPSILON):
			col = 'red'
		else:
			col = 'black'
		try:
			own_bound = '%.5g' % max([-sol['opt'] for sol in self.old_solutions.values() if sol['verify'] == 1 and sol['strategy'] != 'traverse'])
		except (AttributeError, KeyError):
			own_bound = '-\\infty'
		own_min = str('%.5g' % self.min[0]).replace('inf','\\infty')
		res += '[.{$\\substack{%s\\\\\\textcolor{%s}{%.5g}\\\\\\textcolor{blue}{%s}}%s$} \n' % (own_min, col, self.lower_bound, own_bound, stats)
		if self.child0 is not None:
			res += self.child0._tex_tree()
		if self.child1 is not None:
			res += self.child1._tex_tree()
		res += ']\n'
		return res

	def _tree_size(self):
		"""Return the size of the search tree of branch()."""
		size = 1
		if self.child0 is not None:
			size += self.child0._tree_size()
		if self.child1 is not None:
			size += self.child1._tree_size()
		return size

if __name__ == "__main__":
	pass
	#p = Polynomial('standard_simplex',30, 60, 100, seed = 0)
	##collecting further nice examples
	#example4_2 = Polynomial(str(8*x(0)**6 + 6*x(1)**6 + 4*x(2)**6+2*x(3)**6 -3*x(0)**3*x(1)**2 + 8*x(0)**2*x(1)*x(2)*x(3) - 9*x(1)*x(3)**4 + 2*x(0)**2*x(1)*x(3) - 3*x(1)*x(3)**2 + 1))
	#example_small = Polynomial('general',4,8,8,3,seed = 0)
	#ex1 = Polynomial('general',10,20,100,80,seed = 1)
