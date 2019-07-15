#!/usr/bin/env ipython
# -*- coding: utf-8 -*-

from Poem.polynomial import *

import pymp
result_list = pymp.shared.list()
with pymp.Parallel() as env:
	for seed in env.range(32):
		p = Polynomial('general',8,20,57,33, seed = seed)
		p.sonc_opt_python()
		result_list.append(p.solution['opt'])
