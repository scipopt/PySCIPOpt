#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Define constants and basic functions."""

import numpy as np
import sympy
import os
import math
import collections
from scipy.special import binom
from sympy.ntheory.continued_fraction import continued_fraction_iterator, continued_fraction_reduce
from Poem.LP_exact import LP_solve_exact, get_box

VERBOSE = False
#SAVE_PATH = os.path.expanduser('~') +'/local/'
SAVE_PATH = '../instances/'
DB_NAME = 'runs.db'
DIGITS = 23
EPSILON = 2**-DIGITS
FAULT_DATA = {'C':np.array([[]]), 'opt': np.inf, 'solver_time': 0, 'time': 0, 'verify': -1, 'status':'no solution'}

symlog = np.vectorize(sympy.log)

def parse(number):
	"""Parse a number from string to the correct datatype.

	Call:
		res = parse(number)
	Input:
		number: string, representing a number
	Output:
		res: number as int/float/sympy.Rational
	"""
	try:
		res = int(number)
	except ValueError:
		try:
			res = float(number)
		except ValueError:
			res = sympy.sympify(number)
	return res

def get_type(array):
	"""Determine common data type for given arrray.
	
	Possible results: int, np.float, sympy.Rational, object
	"""
	for type_attempt in [int, np.float, sympy.Rational]:
		if all([isinstance(entry, type_attempt) for entry in array]):
			return type_attempt
	return object

def bitsize(arg):
	"""Compute the bit size of a number or a collection of numbers.

	Call:
		size = bitsize(arg)
	Input:
		arg: number (int/np.int/float) or collection (np.array/list/...) in arbitrary nesting
	Output:
		size: (summed up) bit size of arg
	"""
	if isinstance(arg, collections.Iterable):
		return sum([bitsize(e) for e in arg])
	if type(arg) == int:
		return arg.bit_length()
	if type(arg) == np.int64:
		return int(arg).bit_length()
	if type(arg) == sympy.Rational:
		return arg.p.bit_length() + arg.q.bit_length()
	return 64 #default size for float and bounded integer

def binomial(n,k):
	"""Compute binomial coefficient as integer."""
	return int(binom(n,k))

def dt2sec(dt):
	"""Convert a datetime.timedelta object into seconds.

	Call:
		res = dt2sec(dt)
	Parameters:
		dt: datetime.timedelta object
	Output:
		res: a float, representing the seconds of dt
	"""
	return dt.microseconds / 1000000.0 + dt.seconds
	
def _smaller_vectors(alpha, mindegree = None, maxdegree = None):
	"""For given vector alpha list all other vectors, which are elementwise smaller.

	Optionally these can be restricted to some maximal degree.

	Call:
		vector_list = _smaller_vectors_bound(alpha, mindegree, maxdegree)
	Input:
		alpha: vector of non-negative integers
		mindegree: non-negative integer
		maxdegree: non-negative integer
	Output:
		vector_list: list of lists, containing all vectors of degree between (including) `mindegree` and `maxdegree`, which are elementwise at most `alpha`
	"""
	if mindegree is None: mindegree = 0
	if maxdegree is None: maxdegree = sum(alpha)

	if type(alpha) != list:
		alpha = list(alpha)

	#base case
	if sum(alpha) < mindegree: return []
	if maxdegree == 0: return [[0 for _ in alpha]]
	if alpha == []: return [[]]
	
	return [[i] + l for i in range(0, min(alpha[0], maxdegree) + 1) for l in _smaller_vectors(alpha[1:], mindegree - i, maxdegree - i)]

def _vector_to_index(vector,d):
	"""For some given exponent vector, compute the corresponding index.
	
	This function is the inverse function to _index_to_vector.
	Call:
		index = _vector_to_index(vector, d)
	Input:
		vector: array of integers with sum at most d
		d: maximal degree
	Output:
		index: index of the given vector in the lexicographic ordering of `n`-dimensional vector with sum at most `d`
	"""
	#check for valid input
	if (vector < 0).any():
		raise Exception('Negative entry.')
	if vector.sum() > d:
		raise Exception('Entry too large.')
	#base case
	if not vector.any():
		return 0
	#compute first entry and recurse
	n = len(vector)
	offset = sum([binomial(n-1+d-i, n-1) for i in range(vector[0])])
	return offset + _vector_to_index(vector[1:], d - vector[0])

def _index_to_vector(index,n,d):
	"""For some given index, compute the corresponding exponent vector.

	Call:
		vector = _index_to_vector(index,n,d)
	Input:
		index: non-negative integer
		n: number of variables
		d: maximal degree
	Output:
		vector: array of length `n`, the `index`-th exponent vector in the lexicographic ordering of `n`-dimensional vectors with sum at most `d`
	"""
	#base case
	if n == 1:
		return np.array([index])
	vector = np.zeros(n, dtype=np.int)

	#subtract binomial coefficients as long as possible, yields first entry, then recursion
	i = 0
	step = binomial(n-1+d-i,n-1)
	while index >= step:
		index -= step
		i += 1
		step = binomial(n-1+d-i,n-1)
	vector[0] = i
	vector[1:] = _index_to_vector(index, n-1, d-i)
	return vector

def is_psd(C):
	"""Check whether matrix C is positive semidefinite, by checking the eigenvalues."""
	try:
		return all(np.linalg.eigvalsh(C) >= -EPSILON)
	except LinAlgError:
		return False

def linsolve(A,b):
	"""Solve a linear equation system for a possible singular matrix."""
	if b.size == 0:
		return np.array([])
	return np.linalg.lstsq(A,b,rcond = None)[0]

def unify_status(status):
	"""Give a uniform representation for different status flags.

	The following are equivalent:
	1 = Solved = optimal
	0 = Inaccurate = optimal_inaccurate
	-1 = no solution
	"""
	if type(status) == list:
		for i in range(len(status)):
			if status[i] in ['Solved', 'optimal', 1]:
				status[i] = 1
			elif status[i] in ['no solution', -1]:
				status[i] = -1
			else: status[i] = 0
		return min(status)
	elif status in ['Solved', 'optimal']:
		return 1
	elif status == 'no solution':
		return -1
	else:
		return 0

def maximal_elements(input_list, comp = (lambda u,v: (u <= v).all()), sort_key = np.sum):
	"""Given a list of np.array, compute the maximal elements.

	Call:
		maximal = maximal_elements(input_list)
	Input:
		input_list: list of np.array, all need to have the same length.
	Output:
		maximal: list of np.array, which contains the maximal elements of input_list,
			the order used is elementwise "<="
	"""
	l = input_list.copy()
	if sort_key is not None:
		l.sort(reverse=True, key = sort_key)
	maximal = [l[0]]
	for e in l[1:]:
		i = 0
		replaced = False
		while i < len(maximal):
			if comp(maximal[i], e):
			#if (maximal[i] <= e).all():
				if replaced:
					maximal.pop(i)
					i -= 1
				else:
					maximal[i] = e
					replaced = True
			#if (e <= maximal[i]).all():
			if comp(e, maximal[i]):
				break
			i += 1
		if i == len(maximal) and not replaced:
			maximal.append(e)
	return maximal	

def flatten(l):
	"""Flatten a list of irregular depth to a generator."""
	#from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
	for el in l:
		if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
			yield from flatten(el)
		else:
			yield el

#alternative, check which one is faster
def flatten2(x):
	"""Flatten a list of irregular depth to a list."""
	#from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
	if isinstance(x, collections.Iterable):
		return [a for i in x for a in flatten2(i)]
	else:
		return [x]

def to_fraction(number, eps = EPSILON, bound = 0):
	"""Round a given number to a fraction with given accuracy.

	Call:
		frac = to_fraction(number[, eps][, bound])
	Input:
		number: float/agebraic number (any arithmetic expression that can be handled by sympy), number to be converted into a fraction
		eps [optional, default: aux.EPSILON]: desired absolute accuracy
		bound [optional, default: 0]: number, in which direction to round
			0 : closest
			-1: rounded down
			1 : rounded up
	Output:
		frac: symbolic fraction, such that |frac - number| < eps
	"""
	return get_box(number, 16)[bound]
	if type(number) in [float, np.float]:
		return get_box(number, 16)[0]
	gen = continued_fraction_iterator(number)
	cf = [next(gen)]
	frac = continued_fraction_reduce(cf)
	while abs(frac - number) >= eps:
		cf.append(next(gen))
		frac = continued_fraction_reduce(cf)
	#if necessary, perform one more step to have upper/lower bound (works, continued fractions change side each step)
	if bound > 0 and frac < number:
		cf.append(next(gen))
		frac = continued_fraction_reduce(cf)
	if bound < 0 and frac > number:
		cf.append(next(gen))
		frac = continued_fraction_reduce(cf)
	return frac
