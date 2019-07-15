#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to generate LaTeX output from the sql database."""

import Poem.aux as aux
import sqlite3

conn = sqlite3.connect(aux.SAVE_PATH + aux.DB_NAME)
cursor = conn.cursor()

var_list = [2,3,4,8,10,20,30,40]
term_list = [6,9,12,20,24,30,50,100,200,300,500]
degree_list = [6,8,10,20,30,40,50,60]
solver_list = ['ECOS','SeDuMi','SDPT3']

res = '\\begin{tabular}{c|' + 'c'*len(var_list) + '}\n'
res += '\t' + ''.join(['&%d' % var for var in var_list]) + '\\\\\n\t\\hline\n'
for term in term_list:
	line = [str(term)]
	for var in var_list:
		#cursor.execute('select avg(time_sos) as time from best where variables = ? and degree = ?;', (var, term))
		cursor.execute('select time from (select time_even/time_var as time from ecos_split where variables = ? and terms = ? and time_even is not null and time_var is not null) order by time limit 1 offset (select count(*) from ecos_split where variables = ? and terms = ? and time_even is not null and time_var is not null) / 2;', (var, term, var, term))
		#cursor.execute('select avg(opt_even - opt_var) as time from ecos_split where variables = ? and terms = ?;', (var, term))
		#cursor.execute('select avg(time) as time from ecos_variable, polynomial, max_terms where ecos_variable.poly_id = polynomial.rowid and polynomial.variables = max_terms.variables and polynomial.degree = max_terms.degree and polynomial.terms = max_terms.max_terms and shape = \'general\' and polynomial.variables = ? and polynomial.degree = ?;', (var, term))
		try:
			entry = cursor.fetchone()
			line.append('%.3g' % entry)
		except:
			line.append('-')
	res += '\t' + '&'.join(line) + '\\\\\n'
res += '\\end{tabular}'
