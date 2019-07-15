#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Functions for covering the interior points of a polytope with simplices."""

import random
from datetime import datetime
from scipy.optimize import linprog
import numpy as np
from networkx import Graph, traversal

import Poem.generate_poly as gen
import Poem.polytope as polytope
import Poem.aux as aux
from Poem.aux import dt2sec

class CoverGraph(Graph):
	"""Graph-class for coverings of a Newton-polytope.

	This class is intended to be used inside the Polynomial-class.
	The nodes are lists, which represent the simplices of a covering.
	Two of these are connected, if they share a common entry.

	We have one artificial node for the list [0].
	"""

	def __init__(self, T = []):
		"""Initialise the graph with the given covering."""
		Graph.__init__(self)
		self.add_entry([0])
		for l in T:
			self.add_entry(l)
	
	def add_entry(self,l):
		"""Add a node to the graph and create its edges."""
		index = self.number_of_nodes() - 1
		self.add_node(index)
		self.node[index]['list'] = set(l)
		for v in self.nodes():
			if v != index and self.node[v]['list'] & self.node[index]['list'] != set():
				self.add_edge(v,index)

	def get_order(self):
		"""Get nodes of G by BFS with start -1."""
		succ = traversal.bfs_successors(self,-1)
		order = [-1]
		for v in order:
			if v in succ.keys():
				order.extend(succ[v])
		return order

def compute_random_cover(A, m, runs = 0):
	#create short names
	n,t = A.shape
	#init
	V_index = range(m)
	U_index = range(m,t)
	c = -np.ones(m)
	T = []
	if runs == 0: runs = t

	for _ in range(runs):
		r = [0] + random.sample(range(1,m),n - 1)
		try:
			T_index = polytope._get_inner_points(A, U_index, r)
			T_index.sort()
		except:
			continue
		#U_index -= set(T_index)
		#V_index -= set(T_index)
		T.append(T_index)
	return [l for l in T if any([el > m for el in l])]

def compute_full_cover(A):
	T = compute_special_cover(A)
	#create short names
	n,t = A.shape
	m = np.count_nonzero(b > 0)
	#init
	V_index = {i for i in range(len(b)) if b[i] > 0}
	U_index = {i for i in range(len(b)) if b[i] < 0}
	Tflat = set([x for l in T for x in l])

	V_index -= Tflat
	if V_index == set(): return T

	TT = [l for l in compute_cover(A) if 0 in l]
	for l in TT:
		if set(l) - Tflat != set():
			T.append(l)
			Tflat |= set(l)
	V_index -= Tflat

	while V_index != set():
		r = [0] + random.sample(range(1,m),n - 1)
		try:
			l = polytope._get_inner_points(A, U_index, r)
		except:
			continue
		if set(l) - Tflat != set():
			T.append(l)
			V_index -= set(l)
			Tflat |= set(l)
	return T

if __name__ == "__main__":

	n = 3
	degree = 20
	terms = 20
	seed = 0

	t0 = datetime.now()
	A,b = gen.create_poly(n,degree,terms,seed = seed)

	#T = compute_cover(A)

	print("time: %.2f seconds" % dt2sec(datetime.now() - t0))

	#create short names
	n,t = A.shape
	m = np.count_nonzero(b > 0)
	#init
	U_index = {i for i in range(len(b)) if b[i] < 0}
	c = [-1] + [0 for _ in range(m-1)]
	T = []
	T_degenerated = []
	#setting up the LP
	A_eq = A[:,:m]
	A_ub = np.concatenate((np.eye(m,dtype = np.int), -np.eye(m,dtype = np.int)), axis = 0)
	b_ub = np.concatenate((np.ones(m), np.zeros(m)))
	
	#cover U
	while U_index != set():
		ui = U_index.pop()
		u = A[:,ui]
		#find vertices covering u
		res = linprog(c, A_ub, b_ub, A_eq, u)
		T_index = [i for i in range(m) if res['x'][i] > aux.EPSILON]
		#update target vector
		#get all points covered by T_index
		T_index = [ui] + polytope._get_inner_points(A, U_index, T_index)
		T_index.sort()
		#mark covered points
		U_index -= set(T_index)
		if -res['fun'] < aux.EPSILON:
			T_degenerated.append(T_index)
		else:
			T.append(T_index)
