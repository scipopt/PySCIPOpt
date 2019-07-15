#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to run the oprtimisations programmes on all instances."""

import json_tricks as json
import sqlite3
import os
import subprocess
from datetime import datetime
import argparse
import numpy as np

import psutil
import cpuinfo
import platform

try:
	import matlab
	import matlab.engine
except:
	pass

import Poem.aux as aux
from Poem.aux import binomial
from Poem.polynomial import Polynomial
from Poem.exceptions import *

class DBrunner(object):
	"""Class for running tests and feed them into a DB."""

	def __init__(self):
		"""Create runner for the main testing loop.

		Also gets all version number of the solvers and modelers.
		"""
		self.connect()

		self._get_system_id()

		self.solver_rowid = {}
		self.modeler_rowid = {}
		self.programme_rowid = {}

		solver_packages = ['cvxopt', 'ecos', 'scs', 'Mosek']
		modeler_packages = ['cvxpy']
		##have to get Matlab version numbers manually into DB
		matlab_solvers = ['sedumi','sdpt3']
		matlab_modelers = ['cvx','sostools','gloptipoly','yalmip']

		self.cursor.execute('select m.name, solver.rowid, m.version from solver, (select name, max(version) as version from solver group by name) as m where solver.name = m.name and solver.version = m.version;')
		solver_versions = { entry[0]: entry[1:] for entry in self.cursor.fetchall() }
		for solver in matlab_solvers:
			self.solver_rowid[solver] = solver_versions[solver][0]
		self.cursor.execute('select m.name, modeler.rowid, m.version from modeler, (select name, max(version) as version from modeler group by name) as m where modeler.name = m.name and modeler.version = m.version;')
		modeler_versions = { entry[0]: entry[1:] for entry in self.cursor.fetchall() }
		for modeler in matlab_modelers:
			self.modeler_rowid[modeler] = modeler_versions[modeler][0]

		pip_versions = [e.split('==') for e in subprocess.check_output(['pip3', 'freeze'], universal_newlines = True).splitlines()]
		self.pip_versions = {e[0].lower(): e[1] for e in pip_versions}

	def conditional_insert(self, table, entry):
		"""Return the rowid of the request and insert if necessary.

		Input:
			table: string, name of table in DB
			entry: list of pairs (column-name, value), requested tuple in DB
		Output:
			rowid: rowid of the requested tuple
		"""
		request = ('select rowid from %s where ' % table) + ' and '.join(['%s = ?' % e[0] for e in entry])
		self.cursor.execute(request, [e[1] for e in entry])
		res = self.cursor.fetchall()
		if len(res) > 0:
			return res[0][0]
		else:
			self.cursor.execute('insert into %s (%s) values (%s);' % (table, ', '.join([e[0] for e in entry]), ','.join(['?' for _ in entry])), [e[1] for e in entry])
			self.cursor.execute(request, [e[1] for e in entry])
			res = self.cursor.fetchall()
			return res[0][0]

	def _get_modeler_id(self, modeler):
		modeler = modeler.lower()
		if modeler not in self.modeler_rowid.keys():
			if modeler in self.pip_versions.keys():
				self.modeler_rowid[modeler] = self.conditional_insert('modeler', [('version', self.pip_versions[modeler]), ('name', modeler)])
			else:
				self.modeler_rowid[modeler] = self.conditional_insert('modeler', [('version', 0), ('name', modeler)])
		return self.modeler_rowid[modeler]

	def _get_solver_id(self, solver):
		solver = solver.lower()
		if solver not in self.solver_rowid.keys():
			if solver in self.pip_versions.keys():
				self.solver_rowid[solver] = self.conditional_insert('solver', [('version', self.pip_versions[solver]), ('name', solver)])
			else:
				self.solver_rowid[solver] = self.conditional_insert('solver', [('version', 0), ('name', solver)])
		return self.solver_rowid[solver]

	def _get_programme_id(self, language, strategy, modeler, solver):
		modeler = self._get_modeler_id(modeler)
		solver = self._get_solver_id(solver)
		entry = (language, strategy, modeler, solver)
		if entry not in self.programme_rowid.keys():
			self.programme_rowid[entry] = self.conditional_insert('programme', [('language',language), ('strategy', strategy), ('modeler_id', modeler), ('solver_id', solver)])
		return self.programme_rowid[entry]

	def _get_system_id(self):
		#getting system information
		info = cpuinfo.get_cpu_info()
		system = {'table': 'system', 'entries': {}}
		system['entries']['OS'] = platform.platform()
		system['entries']['kernel'] = platform.uname().release
		system['entries']['CPU'] = info['brand']
		system['entries']['cpu_count'] = psutil.cpu_count()
		#system['entries']['freq_GHz'] = psutil.cpu_freq().max / 1000
		system['entries']['freq_GHz'] = float(info['hz_advertised'].split()[0])
		system['entries']['RAM_kB'] = psutil.virtual_memory().total // 1024
		self.system_id = self.conditional_insert('system', [(k,v) for k,v in system['entries'].items()])

	def run(self):
		"""Solve all instances from the given data base, that have not been computed yet.

		It opens the data base at <SAVE_PATH>/<DB_NAME> and asks for all polynomials that have not been computed.
		Then it runs each solver on each instance, checks the validity of the solution and inserts the solution into the data base.

		The computation can be softly interrupted by altering the file "status" to any string, which is not "run".
		"""
		#check run-flag
		if open('status','r').read() != 'run\n': return

		#find all combinations that still have to be computed
		if aux.VERBOSE: print("Getting jobs to be done")
		#restrict number of terms and variables, to avoid too large sizes
		self.cursor.execute('select rowid, bounded, string from polynomial where rowid not in (select distinct poly_id from run) and string != \'fail\' and bounded >= 0 and terms <= 50 and variables < 8;')
		todo_list = self.cursor.fetchall()
		#compute in random order, to have example from all sizes
		import random
		random.shuffle(todo_list)
		if todo_list == []: return

		#running the main loop
		for rowid, bounded, poly_string in todo_list:
			if open('status','r').read() != 'run\n': break
			#if bounded == -1: continue
			print('running poly %d' % rowid)
			#TODO: include Matlab option
			p = Polynomial(poly_string)
			p.traverse(verbose = True)
			solutions = p.old_solutions.copy()
			p = Polynomial(poly_string)
			p.traverse(verbose = True, sparse = True)
			p.forked_bound()
			p.forked_bound(strategy = 'all')
			solutions.update(p.old_solutions)
			for key, data in solutions.items():
				##key = (language, strategy, modeler, solver, params)
				programme_id = self._get_programme_id(*key[:4])
				self.cursor.execute('insert into run (opt, time, verify, poly_id, programme_id, params, timestamp, system_id) values (?,?,?,?,?,?,?,?);', 
						(data['opt'], data['time'], data['verify'], rowid, programme_id, key[-1], int(datetime.timestamp(datetime.now())), self.system_id))
			#update boundedness-status
			if bounded == 0 and p.solution['opt'] < np.inf and p.solution['verify'] == 1:
				self.cursor.execute('update polynomial set bounded = 1 where rowid = ?;', (rowid))
			self.cursor.execute('insert into minimum (poly_id, value, args_json, time, method) values (?,?,?,?,?);', (rowid, p.min[0], json.dumps(p.min[1]), p.solution['time'], 'traverse'))
			self.conn.commit()

	def run_symbolic(self):
		"""Solve all instances symbolically from the given data base, that have not been computed yet.

		It opens the data base at <SAVE_PATH>/<DB_NAME> and asks for all polynomials that have not been computed.
		Then it runs SAGE on each instance, checks the validity of the solution, does the symbolic post-processing and inserts the solution into the data base.

		The computation can be softly interrupted by altering the file "status" to any string, which is not "run".
		"""
		#check run-flag
		if open('status','r').read() != 'run\n': return

		bad = [19340]

		#find all combinations that still have to be computed
		if aux.VERBOSE: print("Getting jobs to be done")
		#restrict number of terms and variables, to avoid too large sizes
		self.cursor.execute('select rowid, string from polynomial where rowid not in (select distinct poly_id from symbolic_sonc) and string != \'fail\' and terms <= 50 and variables <= 10 and bounded = 1 and degree < 30;') 
		todo_list = self.cursor.fetchall()
		#compute in random order, to have example from all sizes
		import random
		random.shuffle(todo_list)
		if todo_list == []: return

		#running the main loop
		for rowid, poly_string in todo_list:
			if rowid in bad: continue
			if open('status','r').read() != 'run\n': break
			print('running poly %d' % rowid)
			#TODO: include Matlab option
			try:
				p = Polynomial(poly_string)
				p.scaleround(100)
				p.sonc_opt_python()
				print('solved poly %d' % rowid)
				if 'error' in p.solution.keys(): 
					print('unsolved')
					continue
				t0 = datetime.now()
				with timeout(60):
					decomp = p.get_decomposition(symbolic = True)
				t1 = aux.dt2sec(datetime.now() - t0)
				shift = sum([q.b[0] for q in decomp]) - p.b[0]
				diff = float(shift - p.solution['opt'])
				bits = sum([q.__sizeof__() for q in decomp]) + shift.p.bit_length() + shift.q.bit_length()
				self.cursor.execute('insert into symbolic_sonc (poly_id, opt, opt_sym, diff, bitsize, time, total_time) values (?,?,?,?,?,?,?);', (rowid, p.solution['opt'], str(shift), diff, bits, t1, t1 + p.solution['time']))
				self.conn.commit()
			except (TypeError, ValueError, OverflowError, TimeoutError, InfeasibleError) as err:
				print(repr(err))

	def connect(self):
		"""Connect to the database given in aux.py."""
		self.conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
		self.cursor = self.conn.cursor()

	def close(self):
		"""Close database connection."""
		self.conn.commit()
		self.conn.close()
	
	def __del__(self):
		"""Close database connection."""
		self.close()

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
	#cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where string is null;')
	cursor.execute('select rowid, * from polynomial where rowid not in (select distinct poly_id from run) and string != \'fail\';')
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
			if todo[7] is None:
				p = Polynomial(*todo[1:6], seed = todo[6], matlab_instance = matlab_engine)
			else:
				p = Polynomial(todo[7], matlab_instance = matlab_engine)
			p.matlab_start_time = matlab_start_time
		except:
			raise Exception('This should not happen.')
			cursor.execute('update polynomial set string = "fail" where shape = ? and variables = ? and degree = ? and terms = ? and inner_terms = ? and seed = ?;', todo[1:7])
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
				cursor.execute('insert into run (status, verify, time, opt, string, poly_id, programme_id, params, timestamp, system_id) values (?,?,?,?,?,?,?,?,?,?);', 
						(data['status'], data['verify'], data['time'], data['opt'], json.dumps(data, allow_nan = True), todo[0], code, params, int(datetime.timestamp(datetime.now())), 1, ))
				cursor.execute('update polynomial set string = ? where shape = ? and variables = ? and degree = ? and terms = ? and inner_terms = ? and seed = ?;', [str(p)] + list(todo[1:7]))
			conn.commit()
		except Exception as err:
			print('Exception %s' % format(err))
			print(todo)
			pass

	#commit and close connections
	conn.commit()
	conn.close()
	matlab_engine.quit()

def check_infinity():
	"""For all instances, where we have no bound, try to check whether they are unbounded."""
	#check run-flag
	if open('status','r').read() != 'run\n': return

	#establish DB-connection
	if not os.path.isfile(aux.SAVE_PATH + aux.DB_NAME):
		raise Exception('Data base not found')
	if aux.VERBOSE:	print("Connecting to data base")
	conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
	cursor = conn.cursor()

	#find all combinations that still have to be computed
	if aux.VERBOSE: print("Getting jobs to be done")
	#cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where string is null;')
	cursor.execute('select rowid, * from polynomial where bounded is null and string != \'fail\';')
	todo_list = cursor.fetchall()
	if todo_list == []: return

	for todo in todo_list:
		if open('status','r').read() != 'run\n': break
		print('running poly %d' % todo[0])
		p = Polynomial(todo[7])
		if p.detect_infinity() is None:
			bounded = -1
		else:
			bounded = 0
		cursor.execute('update polynomial set bounded = ? where rowid = ?;', (bounded, todo[0]))
		conn.commit()

	conn.commit()
	conn.close()

def check_degeneracy():
	"""For all instances (polynomials + cover), compute the number of degenerate points."""
	#check run-flag
	if open('status','r').read() != 'run\n': return

	#establish DB-connection
	if not os.path.isfile(aux.SAVE_PATH + aux.DB_NAME):
		raise Exception('Data base not found')
	if aux.VERBOSE:	print("Connecting to data base")
	conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
	cursor = conn.cursor()

	#find all combinations that still have to be computed
	if aux.VERBOSE: print("Getting jobs to be done")
	cursor.execute('select rowid, string from polynomial where shape = \'general\' and string != \'fail\' and degenerate_points is null;')
	todo_list = cursor.fetchall()
	if todo_list == []: return

	for todo in todo_list:
		if open('status','r').read() != 'run\n': break
		print('running polynomial %d' % todo[0])
		try:
			p = Polynomial(todo[1])
			p._compute_degenerate_points()
			cursor.execute('update polynomial set degenerate_points = ? where rowid = ?;', (len(p.degenerate_points), todo[0]))
			conn.commit()
		except Exception as err:
			print(repr(err))

	conn.commit()
	conn.close()

def check_sos_limit():
	"""Run SOS with sparse-option on all instances, which are one step above the threshold."""
	#check run-flag
	if open('status','r').read() != 'run\n': return

	sizes = [(3,30), (4,20), (8,8)]
	size_query = ' or '.join(['(variables = %d and degree = %d)' % (n,d) for n,d in sizes])

	#establish DB-connection
	if not os.path.isfile(aux.SAVE_PATH + aux.DB_NAME):
		raise Exception('Data base not found')
	if aux.VERBOSE:	print("Connecting to data base")
	conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
	cursor = conn.cursor()

	#get ID of sostools
	cursor.execute('select programme.rowid, language, strategy, callname from programme, solver where programme.solver_id = solver.rowid and callname = \'sostools\';')
	code = cursor.fetchone()[0]

	#establish Matlab-conection
	if aux.VERBOSE: print("Connecting to Matlab")
	t0 = datetime.now()
	matlab_engine = matlab.engine.start_matlab('-useStartupFolderPref -nosplash -nodesktop')
	matlab_start_time = aux.dt2sec(datetime.now() - t0)

	#find all combinations that still have to be computed
	if aux.VERBOSE: print("Getting jobs to be done")
	#cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where string is null;')
	cursor.execute('select rowid, string from polynomial where string != \'fail\' and (%s);' % size_query)
	todo_list = cursor.fetchall()
	if todo_list == []: return

	for todo in todo_list:
		if open('status','r').read() != 'run\n': break
		print('running polynomial %d' % todo[0])
		try:
			p = Polynomial(todo[1], matlab_instance = matlab_engine)
			p.matlab_start_time = matlab_start_time
		except Exception as err:
			print(repr(err))
			
		try:
			p.sostools_opt(sparse = True)
			for key in p.old_solutions.keys():
				data = p.old_solutions[key].copy()
				params = key[-1]
				cursor.execute('insert into run (status, verify, time, opt, json, poly_id, programme_id, params, timestamp, system_id) values (?,?,?,?,?,?,?,?,?,?);', 
						(data['status'], data['verify'], data['time'], data['opt'], json.dumps(data, allow_nan = True), todo[0], code, params, int(datetime.timestamp(datetime.now())), 1, ))
			conn.commit()
		except Exception as err:
			print('Exception %s' % format(err))
			print(todo)

	conn.commit()
	conn.close()

def transfer_data():
	"""Transfer data from old DB schema to new one."""
	#get polynomials from old DB
	conn = sqlite3.connect('../instances/polynomials.db')
	cursor = conn.cursor()
	cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed, json, bounded, degenerate_points from polynomial;')
	polynomials = cursor.fetchall()
	conn.close()

	#insert polynomials into new DB, with new schema
	conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
	cursor = conn.cursor()
	cursor.execute('delete from generator;')
	cursor.execute('delete from polynomial;')
	for entry in polynomials:
		cursor.execute('insert into generator (rowid, shape, variables, degree, terms, inner_terms, seed) values (?,?,?,?,?,?,?);', entry[:-3])
		if entry[-3] != 'fail':
			if entry[-3] is None:
				p = Polynomial(*entry[1:6], seed = entry[6])
			else:
				p = Polynomial(entry[-3])
			cursor.execute('insert into polynomial (rowid, variables, degree, terms, generator_id, bounded, degenerate_points, string) values (?,?,?,?,?,?,?,?);', (entry[0], str(p._variables), str(p._degree), p.A.shape[1], entry[0], entry[-2], entry[-1], str(p)))
			conn.commit()
	conn.close()



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "optimizes the problem for a given list of numbers of suppliers")
	parser.add_argument("-v", "--verbose", action = "store_true", dest = "verbose", help = "enables verbose mode")
	parser.add_argument("-r", "--run", action = "store_true", dest = "run", help = "run the test loop")
	parser.add_argument("-i", "--infinity", action = "store_true", dest = "infinity", help = "check for infinity")
	parser.add_argument("-d", "--degenerate", action = "store_true", dest = "degenerate", help = "check for degenerate points")
	parser.add_argument("-s", "--sos", action = "store_true", dest = "sos", help = "call only sos-sparse on some instances")
	#parser.add_argument("-i", "--input", dest = "inputFile", default = "input.csv", help = "the input filename")
	#parser.add_argument("-l", "--list", dest = "list", default = [1,2,3], help = "the input list")
	#parser.add_argument("-o", "--output", dest = "outputFile", default = "output.csv", help = "the output file")
	args = parser.parse_args()

	aux.VERBOSE = args.verbose
	if args.run:
		check_all()
	if args.infinity:
		check_infinity()
	if args.degenerate:
		check_degeneracy()
	if args.sos:
		check_sos_limit()

	db = DBrunner()
