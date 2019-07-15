#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Test suite for aux.py."""

import Poem.aux as aux
import random
import numpy as np
import types

def test_smaller_vectors_empty():
	assert(aux._smaller_vectors([]) == [[]])
	
def test_smaller_vectors_example():
	arg = [2,1,0]
	target = [[2,1,0], [2,0,0], [1,1,0], [1,0,0], [0,1,0], [0,0,0]]
	assert(set([tuple(item) for item in aux._smaller_vectors(arg)]) == set([tuple(item) for item in target]))

def test_smaller_vectors():
	for seed in range(10):
		np.random.seed(seed)
		arg = np.random.randint(7, size = 5)
		res = aux._smaller_vectors(arg)
		#test number of results
		assert(len(res) == np.prod([a + 1 for a in arg]))
		assert([0,0,0,0,0] in res)
		assert(arg.tolist() in res)
		for entry in res:
			assert((entry <= arg).all())

def test_vector_to_index_inverse():
	n = 5
	d = 10
	for seed in range(20):
		random.seed(seed)
		index = random.randint(0, aux.binomial(n + d, d) - 1)
		assert(aux._vector_to_index(aux._index_to_vector(index, n, d), d) == index)
	
def test_index_to_vector_inverse():
	n = 5
	d = 10
	for seed in range(20):
		np.random.seed(seed)
		vector = np.zeros(n, dtype = np.int)
		for i in range(n):
			vector[i] = random.randint(0, d - vector[:i].sum())
		assert(np.equal(aux._index_to_vector(aux._vector_to_index(vector, d), n, d), vector).all())

def test_is_psd():
	assert(aux.is_psd(np.zeros((7,7))))
	assert(not aux.is_psd(-np.eye(7)))

	for seed in range(20):
		np.random.seed(seed)
		C = np.random.rand(5,5)
		C = np.dot(C, C.T)
		assert(aux.is_psd(C))

def test_unify_status():
	assert(aux.unify_status(['Solved','optimal', 'optimal', 1]) == 1)
	assert(aux.unify_status(['Solved','optimal', 'no solution', 1]) == -1)
	assert(aux.unify_status(['Solved','Inaccurate', 'no solution', 0]) == -1)
	assert(aux.unify_status(['Solved','Inaccurate', 0]) == 0)

def test_maximal_elements():
	for seed in range(10):
		np.random.seed(seed)
		arg = []
		for _ in range(300):
			arg.append(np.random.randint(2, size = 20))
		arg_tuple = [tuple(arg_entry) for arg_entry in arg]
		res = aux.maximal_elements(arg)
		for entry in res:
			assert(tuple(entry) in arg_tuple)
			for arg_entry in arg:
				assert(not ((entry < arg_entry).any() and (entry <= arg_entry).all()))

def test_flatten():
	l = [1,2,[3,4],[[7,8],6]]
	l_gen = aux.flatten(l)
	l_flat = aux.flatten2(l)
	assert(type(l_gen) == types.GeneratorType)
	assert(type(l_flat) == list)
	l_gen_flat = list(l_gen)
	assert(all([type(elem) == int for elem in l_gen_flat]))
	assert(all([type(elem) == int for elem in l_flat]))
	assert(l_gen_flat == l_flat)
