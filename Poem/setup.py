#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Installer for Polynomial Optimisation."""

import subprocess
import argparse
import shutil

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Install all requirements for POEM.")
	parser.add_argument("-v", "--verbose", action = "store_true", dest = "verbose", help = "enables verbose mode")
	parser.add_argument("-u", "--user", action = "store_true", dest = "user", help = "install modules as user")
	parser.add_argument("-d", "--data-base", action = "store_true", dest = "data_base", help = "initialise the data base of test cases")
	#parser.add_argument("-i", "--input", dest = "inputFile", default = "input.csv", help = "the input filename")
	#parser.add_argument("-l", "--list", dest = "list", default = [1,2,3], help = "the input list")
	#parser.add_argument("-o", "--output", dest = "outputFile", default = "output.csv", help = "the output file")
	args = parser.parse_args()
	ModuleNotFoundError = ImportError
	#Checking imports
	call_list = ['pip3', 'install']
	if args.user:
		call_list += ['--user']

	try:
		import numpy
	except ModuleNotFoundError:
		subprocess.call(call_list + ['numpy'])
	try:
		import sympy
	except ModuleNotFoundError:
		subprocess.call(call_list + ['sympy'])
	try:
		import scipy
	except ModuleNotFoundError:
		subprocess.call(call_list + ['scipy'])
	try:
		import cvxpy
	except ModuleNotFoundError:
		subprocess.call(call_list + ['cvxpy'])
	try:
		import json_tricks
	except ModuleNotFoundError:
		subprocess.call(call_list + ['json_tricks'])
	try:
		import tabulate
	except ModuleNotFoundError:
		subprocess.call(call_list + ['tabulate'])
	try:
		import cvxopt
	except ModuleNotFoundError:
		subprocess.call(call_list + ['cvxopt'])
	try:
		import scs
	except ModuleNotFoundError:
		subprocess.call(call_list + ['scs'])
	try:
		import sqlite3
	except ModuleNotFoundError:
		subprocess.call(call_list + ['sqlite3'])
	try:
		import sparse
	except ModuleNotFoundError:
		subprocess.call(call_list + ['sparse'])
	try:
		import pymp
	except ModuleNotFoundError:
		subprocess.call(call_list + ['pymp-pypi'])
	try:
		import cpuinfo
	except ModuleNotFoundError:
		subprocess.call(call_list + ['py-cpuinfo'])

	#=== optional modules ===
	try:
		import cdd
	except ModuleNotFoundError:
		if shutil.which('cdd_both_reps') is not None:
			subprocess.call(call_list + ['cdd'])
	try:
		import z3
	except ModuleNotFoundError:
		if shutils.which('z3') is not None:
			subprocess.call(call_list + ['z3'])

	try:
		import matlab
		import matlab.engine
	except ModuleNotFoundError:
		if args.verbose:
			print('Warning: Matlab engine not found.')

	#Create pydoc
	doc_list = ['generate_poly.py', 'polytope.py', 'runner.py', 'polynomial.py', 'polynomial_base.py', 'aux.py', 'exceptions.py', 'LP_exact.py', 'AGE_polynomial.py', 'circuit_polynomial.py']
	for entry in doc_list:
		subprocess.call(['pydoc3', '-w', './' + entry])

	if args.data_base:
		subprocess.call('sqlite3 ../instances/runs.db \'.read ../sql/poly_db_init.sql\'')
		subprocess.call('sqlite3 ../instances/runs.db \'.read ../instances/polynomials.sql\'')

