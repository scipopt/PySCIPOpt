#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

import subprocess
from Poem.polynomial import *
from env_changer import *

TIME_BOUND = np.inf

#p = Polynomial('general',5,12,21,15,seed = 8)
#interesting instance for SAGE, verification fails
#p = Polynomial('general',5,12,21,15,seed = 8)
p = Polynomial(8150)
#p = Polynomial('general',5,12,21,15,seed = 8)

#p.run_all()
#t = sum([s['time'] for s in p.old_solutions.values()])
#try:
#	m = int(np.round(TIME_BOUND / t))
#except:
#	m = None

t0 = datetime.now()
p.traverse(reltol = 1e-3, max_size = None, sparse = True)
tree = p._tex_tree()
t1 = aux.dt2sec(datetime.now() - t0)

writer = open('/home/hennich/.tmp/BnB_tree/tree.tex', 'w')
writer.write('\\newcommand{\\timing}{%.2f}' % t1)
writer.write(tree)
writer.close()

with runTestEnv('/home/hennich/.tmp/BnB_tree'):
	subprocess.call(['pdflatex', 'LaTeX.tex'])

print(t1)
print(p._tree_size())

#bench = open('benchmarks','r').read().splitlines()
#benchmarks = {entry[0] : entry[1] for entry in [foo.split(' = ') for foo in bench if foo.find(' = ') > 0]}
