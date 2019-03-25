#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""Module to run the oprtimisations programmes on all instances."""

import json_tricks as json
import sqlite3
import numpy as np
import os
import subprocess
from datetime import datetime
import argparse

try:
	import matlab
	import matlab.engine
except:
	print('Warning: Matlab engine not found. Main loop will fail.') 
	pass

import aux
from aux import binomial
from polynomial import Polynomial

def fill_DB():
	"""Create a series of input examples of sparse polynomials and add them to a database."""
	#initialise database if it does not exist
	if not os.path.isfile(aux.SAVE_PATH + aux.DB_NAME):
		subprocess.call(['sqlite3','-init','../sql/poly_db_init.sql',aux.SAVE_PATH + aux.DB_NAME, ''])
		
	#setup database
	conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
	cursor = conn.cursor()

	t0 = datetime.now()
	for number_of_variables in [2,3,4,8,10,20,30,40]:
		for degree in [6,8,10,20,30,40,50,60]:
			for terms in [6,9,12,20,24,30,50,100,200,300,500]:
				#if more terms than standard simplex, no polynomial exists
				if terms > binomial(number_of_variables + degree, degree):
					continue
				for seed in range(10):
					if aux.VERBOSE > 1:
						print('creating: %d, %d, %d, %d' % (number_of_variables, degree, terms, seed))
					if degree > number_of_variables:
						cursor.execute("insert into polynomial (shape, variables, degree, terms, inner_terms, seed) values (?,?,?,?,?,?);", ('standard_simplex', number_of_variables, degree, terms, terms - (number_of_variables + 1), seed))
						cursor.execute("insert into polynomial (shape, variables, degree, terms, inner_terms, seed) values (?,?,?,?,?,?);", ('simplex', number_of_variables, degree, terms, terms - (number_of_variables + 1), seed))
					steps = 5
					inner_set = set([ratio * (terms - (number_of_variables + 1)) // steps for ratio in range(1,steps)])
					for i in inner_set:
						cursor.execute("insert into polynomial (shape, variables, degree, terms, inner_terms, seed) values (?,?,?,?,?,?);", ('general', number_of_variables, degree, terms, i, seed))
	
	conn.commit()
	conn.close()
	if aux.VERBOSE:
		print("DB filled, time: %.2f seconds" % aux.dt2sec(datetime.now() - t0))

def check_all():
	"""Solve all instances from the given data base, that have not been computed yet.

	It opens the data base at <SAVE_PATH>/<DB_NAME> and asks for all polynomials that have not been computed.
	Then it runs each solver on each instance, checks the validity of the solution and inserts the solution into the data base.

	The computation can be softly interrupted by altering the file "status" to any string, which is not "run".
	"""
	#check run-flag
	if open('status','r').read() != 'run\n': return

	#establish DB-connection
	if not os.path.isfile(aux.SAVE_PATH + aux.DB_NAME):
		fill_DB()
	if aux.VERBOSE:	print("Connecting to data base")
	conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
	cursor = conn.cursor()

	#get the ID of all solvers
	cursor.execute('select programme.rowid, language, strategy, callname from programme, solver where programme.solver_id = solver.rowid;')
	prog = cursor.fetchall()
	solvers = {pr[1:] : pr[0] for pr in prog}

	#find all combinations that still have to be computed
	if aux.VERBOSE: print("Getting jobs to be done")
	cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where json is null;')
	todo_list = cursor.fetchall()
	if todo_list == []: return

	#establish Matlab-conection
	if aux.VERBOSE: print("Connecting to Matlab")
	t0 = datetime.now()
	matlab_engine = matlab.engine.start_matlab('-useStartupFolderPref -nosplash -nodesktop')
	matlab_start_time = aux.dt2sec(datetime.now() - t0)

	#running the main loop
	for todo in todo_list:
		if open('status','r').read() != 'run\n': break
		print('running poly %d' % todo[0])

		try:
			p = Polynomial(*todo[1:6], seed = todo[6], matlab_instance = matlab_engine)
			p.matlab_start_time = matlab_start_time
		except:
			cursor.execute('update polynomial set json = "fail" where shape = ? and variables = ? and degree = ? and terms = ? and inner_terms = ? and seed = ?;', todo[1:])
			conn.commit()
			print('failed')
			continue

		try:
			p.run_all(keep_alive = True)
			#p.run_all()
			for key in p.old_solutions.keys():
				data = p.old_solutions[key].copy()
				params = key[-1]
				code = solvers[key[:3]]
				cursor.execute('insert into run (status, verify, time, opt, json, poly_id, programme_id, params, timestamp, system_id) values (?,?,?,?,?,?,?,?,?,?);', 
						(data['status'], data['verify'], data['time'], data['opt'], json.dumps(data, allow_nan = True), todo[0], code, params, int(datetime.timestamp(datetime.now())), 1, ))
				cursor.execute('update polynomial set json = ? where shape = ? and variables = ? and degree = ? and terms = ? and inner_terms = ? and seed = ?;', [str(p)] + list(todo[1:]))
			conn.commit()
		except Exception as err:
			print('Exception %s' % format(err))
			print(todo)
			pass

	#commit and close connections
	conn.commit()
	conn.close()
	matlab_engine.quit()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "optimizes the problem for a given list of numbers of suppliers")
	parser.add_argument("-v", "--verbose", action = "store_true", dest = "verbose", help = "enables verbose mode")
	parser.add_argument("-r", "--run", action = "store_true", dest = "run", help = "run the test loop")
	#parser.add_argument("-i", "--input", dest = "inputFile", default = "input.csv", help = "the input filename")
	#parser.add_argument("-l", "--list", dest = "list", default = [1,2,3], help = "the input list")
	#parser.add_argument("-o", "--output", dest = "outputFile", default = "output.csv", help = "the output file")
	args = parser.parse_args()

	aux.VERBOSE = args.verbose
	if args.run:
		check_all()
