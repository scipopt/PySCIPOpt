#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Define constants and basic functions."""

import numpy as np
import signal
import math
import os
from scipy.special import binom

VERBOSE = False
SAVE_PATH = os.path.expanduser('~') +'/local/'
DB_NAME = 'runs.db'
DIGITS = 6
EPSILON = 10**-DIGITS
FAULT_DATA = {'C':np.array([[]]), 'opt': np.inf, 'solver_time': 0, 'time': 0, 'verify': -1, 'status':'no solution'}
COMPUTATION_TIME = 3600
CHECK_TIME = 10

def binomial(n,k):
	"""Compute binomial coefficient as integer."""
	return int(binom(n,k))

def _vector_to_symmetric(C_vector):
	"""Transform a vector, corresponding to the upper triangle, into a symmetric matrix."""
	if type(C_vector) != np.ndarray: C_vector = np.array(C_vector)
	if len(C_vector.shape) > 1: C_vector = C_vector.T[0]
	size = int(np.sqrt(2*len(C_vector) + 0.25) - 0.5)
	if size * (size + 1) // 2 != len(C_vector):
		raise Exception('Vector length is no triangular number.')
	tri = np.zeros((size,size))
	tri[np.triu_indices(size, 0)] = C_vector
	tri[np.diag_indices_from(tri)] /= 2
	return tri + tri.T

def _symmetric_to_vector(C):
	"""Transform a symmetric matrix to a vector.

	This function is the inverse of _vector_to_symmetric().

	Call:
		C_vector = _symmetric_to_vector(C)
	Input:
		C: symmetric matrix
	Output:
		C_vector: vector, containing rowise the upper triangle of C
	"""
	if not np.equal(C, C.T).all():
		raise Exception('Matrix is not symmetric.')
	if type(C) == list: C = np.array(C)
	return np.array([C[i,j] for i in range(C.shape[0]) for j in range(i,C.shape[1])])

def dt2sec(dt):
	"""Convert a datetime.timedelta object into seconds.

	Call:
		res = dt2sec(dt)
	Parameters:
		dt: datetime.timedelte object
	Output:
		res: a float, representing the seconds of dt
	"""
	return dt.microseconds / 1000000.0 + dt.seconds
	
class TimeoutError(Exception):
	"""Basic Error class for timeouts."""

	pass

class timeout:
	"""Small environment to impose a maximum time for commands."""

	def __init__(self, seconds=1, error_message='Timeout'):
		"""Setup the timer with the given amount of seconds."""
		self.seconds = seconds
		self.error_message = error_message
	def handle_timeout(self, signum, frame):
		"""Event that happens, if the maximal time is reached."""
		raise TimeoutError(self.error_message)
	def __enter__(self):
		"""Start the timer, when entering the environment."""
		signal.signal(signal.SIGALRM, self.handle_timeout)
		signal.alarm(self.seconds)
	def __exit__(self, type, value, traceback):
		"""Stop the timer, when leaving the environment."""
		signal.alarm(0)
	
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

	try:
		if type(alpha) != list:
			alpha = list(alpha)
	except Exception as err:
		raise Exception('Cannot convert argument into list.')

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
	except Exception:
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
