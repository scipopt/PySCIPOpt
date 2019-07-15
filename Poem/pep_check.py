#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pep257
for e in pep257.check(['generate_poly.py', 'polytope.py','runner.py','polynomial.py', 'polynomial_opt.py', 'polynomial_base.py', 'circuit_polynomial.py', 'AGE_polynomial.py', 'LP_exact.py', 'aux.py']): print(e)
