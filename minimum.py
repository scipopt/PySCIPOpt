#!/usr/bin/env ipython
from polynomial import *
from generate_poly import *
from scipy import optimize
from itertools import combinations
from aux import linsolve, dt2sec
from datetime import datetime

import os
import subprocess

# New Database, path should be in same folder as "polynomial.sql"
DB_Path = os.path.expanduser('~') +'/bachelorarbeit/SONC_Minimization/comparison/python/'
DB_Name = 'result.db'

def minimum(p, start):
#computes minimum of polynomial p with gradient method	
	p = Polynomial(str(p))
	#p as function for optimize.fmin
	def f(x):
		return p(*x)
	#find the minimum with scipy
	return optimize.fmin(f, start)

def start_point(p, pos = 0):
#choose random starting point for gradient method, uniform distribution in box around origin, if pos = True, then only on positive orthant
	p = Polynomial(str(p))
	#number of variables
	A = p.A[1:,:]
	n = A.shape[0]
	# size of box chosen random (+-10 around origin)
	if pos:
		return 20*np.random.random_sample(n,)
	return 40*np.random.random_sample(n,) -20


def minimum_of_sonc(p, start):
#uses SONC decomposition to compute minimum of SONC (= sum of the circuit polynomials from decomposition)
	dec = p.get_decomposition()
	summe = dec[0]
	for q in dec[1:]:
		summe += q
	return [minimum(summe, start),summe]

def minimum_of_circuits(p, start):
#compute minima of circuit polynomials of sonc decomposition, gives back list of minima of circuit polynomials computed via gradient method
	dec = p.get_decomposition()
	min_dec = []
	for q in dec:
		mini = minimum(q, start)
		min_dec.append(mini)
	return(min_dec)

def bary(minlist):
#computes barycentre of a given list of points, gives back one point
	n = len(minlist[0])
	l = len(minlist)
	centre = np.zeros(n)
	for m in minlist:
		for i in range(n):
			centre[i] += m[i]
	for i in range(n):
		centre[i] = centre[i]/l
	return centre

def symbolic_general(p):
#computes minimum of p symbolically, for len(y) =1 (otherwise dimensions do not fit) 
	lamb = np.transpose([p.lamb[0]])
	alph = p.A[:,:p.monomial_squares][1:]
	#alph = [alph[i][1:] for i in range(len(alph))]
	b = np.transpose([p.b[:p.monomial_squares]])	
	y = p.A[:,p.monomial_squares:][1:]
	c = p.b[p.monomial_squares:]
	if  b[0] != lamb[0]:
		b0 = b[0]
		b = [lamb[0]/b0*b[i] for i in range(len(b))]
		c = lamb[0]/b0*c
	lamb = lamb[1:]
	b = b[1:]
	right = np.log(abs(-lamb/b*c))
	root = linsolve(np.transpose(alph)[1:]-np.transpose(y), right)
	return np.exp(root)

def circuit_number(p):
# computes circuit number of p, gives back boolean whether circuit number == -c (up to eps = 10e-7)
	p.sonc_opt_python()
	lamb = np.transpose([p.lamb[0]])
	b = np.transpose([p.b[:p.monomial_squares]])
	if lamb[0] ==0:
		lamb = lamb[1:]
		b = b[1:]
	b0 = b[0]
	b = [lamb[0]/b0*b[i] for i in range(len(b))]
	circ_num = np.prod((b/lamb)**lamb)
	c = p.b[p.monomial_squares:]
	c = lamb[0]/b0*c
	return abs(circ_num + c[0]) < 10e-7
'''	
def orths(barylist):
#returns list of barylist with value on all different orthants,
	combs = []
	orthas = []
	for i in range(1, len(barylist)+1):
		comb = [list(x) for x in combinations(barylist, i)]
		combs.extend(comb)
	b = barylist[:]	
	for comb in combs:
		for i in comb:
			for els in b:
				if els == i:
					b[b.index(els)] = np.negative(els)
		orthas.append(list(b))
		b = barylist[:]
	return(orthas)
'''
def get_results(p):
#run all different methods to compute the minimum of a polynomial p, with time for each method
	if p.is_sum_of_monomial_squares():
		t0 = datetime.now()		
		results = {"start":'not necessary', "min_circ_symb":"0",  "update_symb":"0","sonc_min":"0", "f_sonc_min":"0", "p_sonc_min":"0", "func_update_symb":str(p.b[0]),"func_sonc_grad":str(p.b[0]), "t_symb":datetime.now() - t0, "t_sonc_grad":datetime.now() - t0, "p_min_grad":'0', "func_p_grad":str(p.b[0]), "end_func_grad":p.b[0], "min_func_grad":p.b[0],"iterations":0, "t_grad":datetime.now()-t0}
		return results
	data = p.sonc_opt_python()
	if data['verify'] == 1 and not p.monomial_squares == p.A.shape[1]:
		t0 = datetime.now()
		start = start_point(p)
		t_start = datetime.now()-t0

		#compute minima of circuit polynomials numerically, take barycentre of minima as start point for gradient on p		
		t0 = datetime.now()
		minima = minimum_of_circuits(p,start)
		update = minimum(p,bary(minima))
		func_update_grad = p(np.array(update).T[0])
		t_bary_grad = datetime.now() - t0 + t_start

		#compute minima of circuit polynomial via symbolic computation, take barycentre of minima as start point for gradient on p
		t0 = datetime.now()
		symb_list = []
		decomp = p.get_decomposition()
		for q in decomp:
			q._compute_zero_cover()
			symb_list.append(symbolic_general(q))
		update_symb = minimum(p,bary(symb_list))
		func_update_symb = p(np.array(update_symb).T[0])
		t_symb = datetime.now() -t0
		#compute minimum of SONC polynomial, take minimum as start point for gradient on p
		t0 = datetime.now()
		sonc_min = minimum_of_sonc(p,start)
		sonc = Polynomial(str(sonc_min[1]))
		f_sonc_min = sonc(np.array(sonc_min[0]).T[0])
		p_sonc_min = minimum(p,sonc_min[0])
		func_sonc_grad = p(np.array(p_sonc_min).T[0])
		t_sonc_grad = datetime.now() -t0 + t_start
		#take gradient on p with random start points, stop if solution at least as  good as with SONC
		t0 =datetime.now()
		p_min_grad = minimum(p,start)
		func_p_grad = p(np.array(p_min_grad).T[0])
		start_list = [start]
		func_list =[func_p_grad]
		count = 1
		while abs(func_p_grad - func_update_symb) > 10e-5 and count < 50:
			start = start_point(p)
			start_list.append(start)
			p_min_grad = minimum(p, start)
			func_p_grad = p(np.array(p_min_grad).T[0])
			func_list.append(func_p_grad)
			count += 1
		min_func_grad = min(func_list)
		end_func_grad = func_list[-1]
		#print(func_list)
		t_grad = datetime.now()-t0 + t_start
		results = {"start":start_list, "min_circ_symb": symb_list, "update_symb":update_symb, "sonc_min":sonc_min[0], "f_sonc_min":f_sonc_min, "p_sonc_min":p_sonc_min, "t_symb": t_symb, "t_sonc_grad":t_sonc_grad, "func_update_symb":func_update_symb, "func_sonc_grad":func_sonc_grad, "p_min_grad":p_min_grad, "func_p_grad":func_list, "end_func_grad": end_func_grad, "min_func_grad": min_func_grad, "iterations":count, "t_grad":t_grad}
		#results = {"sonc_min":sonc_min[0], "f_sonc_min":f_sonc_min}
	else: 
		t0 = datetime.now()
		results = {"start":'not solvable', "min_circ_symb":'not solvable', "update_symb":'not solvable',"sonc_min":'not solvable', "f_sonc_min":'not solvable', "p_sonc_min":'not solvable', "t_symb":t0 - t0, "t_sonc_grad":t0 - t0, "func_update_symb":'not solvable', "func_sonc_grad":'not solvable', "p_min_grad":'not solvable', "func_p_grad":'not solvable',"end_func_grad":'not solvable', "min_func_grad":'not solvable', "iterations":0, "t_grad":t0-t0}
	return results
	
def dbfile():
#Database with all solutions for minima computation (in table "run", maybe change to new table?)
	#initialise database if it does not exist
	if not os.path.isfile(DB_Path + DB_Name):
		subprocess.call(['sqlite3','-init','../python/polynomials.sql',DB_Path + DB_Name, ''])
		
	conn = sqlite3.connect(DB_Path + DB_Name)
	cursor = conn.cursor()

	cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where json != "fail" and variables < 13 and degree <= 10')
	todo_list = cursor.fetchall()
	if todo_list == []: return

	for todo in todo_list[2000:]:
		if open('status','r').read() != 'run\n': break
		print('running poly %d' % todo[0])

		p = Polynomial(*todo[1:6], seed = todo[6])
		#print('p =', p)
		p._normalise()
		p.run_all()
		result = get_results(p)
		#for key in p.old_solutions.keys():
		#	data = p.old_solutions[key].copy()
		#	params = key[-1]
		data = p.solution		
		cursor.execute('insert into run(status, verify, time, opt, json, poly_id, programme_id, params, timestamp, system_id, starting, min_circ_symb, min_p_symb, func_p_symb, t_symb, p_sonc_grad, func_sonc_grad, t_sonc_grad, p_min_grad, func_p_grad, end_func_grad, min_func_grad, iterations, t_grad) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);',(data['status'], data['verify'], data['time'], data['opt'], json.dumps(data, allow_nan = True), todo[0], None, None, int(datetime.timestamp(datetime.now())), 1,str(result['start']), str(result['min_circ_symb']), str(result['update_symb']), str(result['func_update_symb']), aux.dt2sec(result['t_symb']), str(result['p_sonc_min']), str(result['func_sonc_grad']), aux.dt2sec(result['t_sonc_grad']), str(result['p_min_grad']), str(result['func_p_grad']), result['end_func_grad'], result['min_func_grad'], result['iterations'], aux.dt2sec(result['t_grad'])))
		cursor.execute('update polynomial set json = ? where shape = ? and variables = ? and degree = ? and terms = ? and inner_terms = ? and seed = ?;', [str(p)] + list(todo[1:]))
		conn.commit()

	#commit and close connections
	conn.commit()
	conn.close()

def dbsymbolic():
	'''Database with solutions to symbolic computation for circuit polynomials only
	circuit polynomials are taken from sonc decomposition of polynomials in "polynomials.sql"

	Database (stored in table "symbolic" of DB_Name) contains:
		- circuit polynomials as string
		- boolean whether circuit number == -c
		- computed value of minimum plus associated function value
		- time needed for computation of symbolic computation
		- comparison to gradient method (computed minimum plus function value with gradient method and time needed)'''

	#initialise database if it does not exist
	if not os.path.isfile(DB_Path + 'symbolic.db'):
		subprocess.call(['sqlite3','-init','../python/polynomials.sql',DB_Path + 'symbolic.db', ''])
		
	conn = sqlite3.connect(DB_Path + 'symbolic.db')
	cursor = conn.cursor()

	cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where json != "fail" and variables < 13 and degree <= 10 and terms < 10 and inner_terms < 5')
	todo_list = cursor.fetchall()
	if todo_list == []: return
	for todo in todo_list[500:]:	
		if open('status','r').read() != 'run\n': break
		print('running poly %d' % todo[0])
		p = Polynomial(*todo[1:6], seed = todo[6])
		p._normalise()
		if p.monomial_squares == p.A.shape[1]:
			data = { 'time': 0, 'language': 'python', 'solver': 'trivial', 'strategy': 'trivial', 'status': 1, 'verify': 1, 'params': {}, 'C': np.array([]), 'opt': -p.b[0] }	
		else: data = p.sonc_opt_python()
		if data['verify']==1 and not p.monomial_squares == p.A.shape[1]:
			decomp = p.get_decomposition()
			for q in decomp:
				#q.sonc_opt_python()
				t0 = datetime.now()
				q._compute_zero_cover()
				symb = symbolic_general(q)
				func_value_symb = q(*np.array(symb).T[0])
				t_symb = datetime.now() - t0
				t0 = datetime.now()
				start = start_point(q, pos = 1)
				numerical = minimum(q, start)
				func_value_num = q(*np.array(numerical))
				count = 1
				while abs(func_value_num - func_value_symb) > 10e-5 and count < 100:
					start = start_point(q, pos = 1)
					numerical = minimum(q, start)
					func_value_num = q(*np.array(numerical))
					count += 1
				t_numeric = datetime.now() -t0
				if circuit_number(q) == True: circuit_numb = 1
				else: circuit_numb = 0
				cursor.execute('INSERT INTO symbolic (polynomial, circuit_number, symbolic, func_value_symb, t_symb, starting_point, iterations, numerical, func_value_num, t_numeric) values (?,?,?,?,?,?,?,?,?,?);',(str(q.symbolic()), circuit_numb, str(symb), str(func_value_symb), aux.dt2sec(t_symb), str(start), count, str(numerical), str(func_value_num), aux.dt2sec(t_numeric)))
				conn.commit() 
		
	conn.commit()
	conn.close()

#dbsymbolic()
#dbfile()

'''
conn = sqlite3.connect(DB_Path + 'result.db')
cursor = conn.cursor()
add = 'ALTER TABLE run ADD COLUMN sonc_min text'
#cursor.execute(add)
ad = 'ALTER TABLE run ADD COLUMN f_sonc_min float'
#cursor.execute(ad)
cursor.execute('select rowid, shape, variables, degree, terms, inner_terms, seed from polynomial where json != "fail" and variables < 13 and degree <= 10')
todo = cursor.fetchall()
for t in todo[827:]:
	#print(t)
	#f = t[0].split(',')
	#for el in f:
	#print(el)
	#	if el[-1] == ']': el = el[:-1]
	#	if el[0] == '[': el = el[1:]
	#	if el == 'not solvable': el = 'inf'
	#	el = float(el)
		#l.append(el)
	#print(l)
	#n = max([int(i) for i in re.findall(r'x\(?([0-9]+)\)?', str(t))]) + 1	
	#cursor.execute('UPDATE symbolic set variables = ? where polynomial = ?', (n,t[0]))
	p = Polynomial(*t[1:6], seed = t[6])
	#print('p =', p)
	p._normalise()
	p.run_all()
	result = get_results(p)
	cursor.execute('UPDATE run set sonc_min = ? where poly_id = ?',(str(result['sonc_min']),t[0]))
	cursor.execute('UPDATE run set f_sonc_min = ? where poly_id = ?', (result['f_sonc_min'],t[0])) 
	conn.commit()
conn.close
'''
			

