#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Test suite for polytope.py."""

import Poem.polytope as polytope
import Poem.generate_poly as gen
import numpy as np

def test_is_in_convex_hull_agree():
	n = 3
	for seed in range(20):
		np.random.seed(seed)
		A = gen.make_affine(np.random.randint(10, size = (n,12)))
		v = np.random.randint(10, size = (n + 1))
		v[0] = 1
		assert(polytope.is_in_convex_hull((A,v)) == polytope.is_in_convex_hull_cvxpy((A,v)))

def test_convex_hull_agree():
	n = 4
	for seed in range(10):
		np.random.seed(seed)
		A = gen.make_affine(np.random.randint(10, size = (n,12)))
		assert(polytope.convex_hull_LP(A) == polytope.convex_hull_LP_serial(A))

def test_interior_points():
	n = 3
	for seed in range(5):
		np.random.seed(seed)
		A = gen.make_affine(np.random.randint(10, size = (n,12)))
		points = polytope.interior_points(A)
		number = polytope.number_interior_points(A)
		assert(len(points) == number)
		for point in points:
			assert(polytope.is_in_convex_hull((A,point)))

def test_get_inner_points():
	A = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [ 0, 10, 0, 6, 8, 1, 6, 8, 2], [ 0, 0, 10, 8, 6, 1, 6, 1, 1]])
	U_index = [5,6,7,8]
	assert(polytope._get_inner_points(A, U_index, [0,1,4]) == [0,1,4,7,8])
	assert(polytope._get_inner_points(A, U_index, [0,1,2]) == [0,1,2,5,7,8])
	res = polytope._get_inner_points(A, set(U_index), [0,1,3])
	res.sort()
	assert(res == [0,1,3,5,6,7,8])
