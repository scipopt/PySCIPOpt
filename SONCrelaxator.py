#! usr/bin/env python3
from pyscipopt       import SCIP_RESULT, Relax, Term, Expr
from constrained     import *
from convert         import *
from Poem.polynomial import *
from Poem.exceptions import InfeasibleError

import numpy as np
import re


class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    def relaxinit(self):
        #dictionary to store solved instances and corresponding solution
        self.solved_instance = dict()

    def relaxexec(self):
        """execution method of SONC relaxator"""

        nvars = len(self.model.getVars())
        conss = self.model.getConss()
        constraint_list = []

        #transform each constraint (type linear, quadratic or expr) into a Polynomial to get lower bound with all constraints used
        for cons in conss:
            constype = cons.getType()
            lhs = self.model.getLhs(cons)
            rhs = self.model.getRhs(cons)
            if  constype == 'expr':
                #get constraints as as polynomial (POEM)
                exprcons = self.model.getConsExprPolyCons(cons)
                polynomial = ExprToPoly(exprcons, nvars)
                polynomial.clean()

                #transform into problem with constraints >= 0
                if not self.model.isInfinity(-lhs):
                    polynomial.b[0] -= lhs
                    constraint_list.append(polynomial.copy())
                    polynomial.b[0] += lhs
                if not self.model.isInfinity(rhs):
                    polynomial.b[0] -= rhs
                    polynomial.b *= -1
                    constraint_list.append(polynomial)

            elif constype == 'linear':
                #get linear constraint as Polynomial (POEM)
                coeffdict = self.model.getValsLinear(cons)
                A = np.array([np.zeros(nvars+1)]*nvars)
                b = np.zeros(nvars+1)
                for i,(key,val) in enumerate(coeffdict.items()):
                    b[i] = val
                    j = re.findall(r'x\(?([0-9]+)\)?', str(key))
                    A[int(j[0])][i] = 1.0
                polynomial = Polynomial(A,b)
                polynomial.clean()

                #transform into problem with constraints >= 0
                if not self.model.isInfinity(-lhs):
                    polynomial.b[0] -= lhs
                    constraint_list.append(polynomial.copy())
                    polynomial.b[0] += lhs
                if not self.model.isInfinity(rhs):
                    polynomial.b[0] -= rhs
                    polynomial.b *= -1
                    constraint_list.append(polynomial)
                #print(lhs,rhs)
            elif constype == 'quadratic':
                #get quadratic constraint as Polynomial (POEM)
                bilin, quad, lin = self.model.getTermsQuadratic(cons)
                #number of terms in constraint +1 for constant term
                nterms = len(bilin)+len(quad)*2+len(lin)+1
                A = np.array([np.zeros(nterms)]*nvars)
                b = np.zeros(nterms)
                #index of term, 0 is for constant term
                i = 1
                for el in bilin:
                    b[i] = el[2]
                    j = int(str(el[0])[-1])
                    A[j][i] = 1.0
                    j = int(str(el[1])[-1])
                    A[j][i] = 1.0
                    i += 1
                for el in quad:
                    j = int(str(el[0])[-1])
                    b[i] = el[1]
                    A[j][i] = 2.0
                    i += 1
                    b[i] = el[2]
                    A[j][i] = 1.0
                    i += 1
                for el in lin:
                    b[i] = el[1]
                    j = int(str(el[0])[-1])
                    A[j][i] = 1.0
                    i += 1
                polynomial = Polynomial(A,b)
                polynomial.clean()

                #transform into problem with constraints >= 0
                if not self.model.isInfinity(-lhs):
                    polynomial.b[0] -= lhs
                    constraint_list.append(polynomial.copy())
                    polynomial.b[0] += lhs
                if not self.model.isInfinity(rhs):
                    polynomial.b[0] -= rhs
                    polynomial.b *= -1
                    constraint_list.append(polynomial)
            else:
                raise Warning("relaxator not available for constraints of type ", constype)

        #No  constraints of type expr, quadratic or linear
        if constraint_list == []:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        #get Objective as Polynomial (POEM)
        obj = ExprToPoly(self.model.getObjective(), nvars)

        #use the Variable bounds as well
        #TODO: improve usage of bounds, maybe delete Constraints if same as bounds (for Polynomial)
        #TODO: problem if bounds are tightened since it makes completely new polynomial (linear term is changed)
        #TODO: 'polynomial unbounded at point ..' appears if variables do not appear in polynomial (possibility to fix them to 0?)
        for y in self.model.getVars():
            if y.getUbLocal() == y.getLbLocal() and y.getUbLocal() == 0:
                constraint_list.append(ExprToPoly({Term(y,y):-1.0}, nvars))
            else:
                if y.getUbLocal() != 1e+20:
                    boundcons = ExprToPoly(Expr({Term(): y.getUbLocal(), Term(y):-1.0}), nvars)
                    #if not boundcons in constraint_list:
                    #print(boundcons)
                    constraint_list.append(boundcons)
                if y.getLbLocal != -1e+20: # and y.getLbLocal != 0.0:
                    boundcons = ExprToPoly(Expr({Term(): -y.getLbLocal(), Term(y):1.0}), nvars)
                    #if not boundcons in constraint_list:
                    #print(boundcons)
                    constraint_list.append(boundcons)

        #check if instance was already solved
        ncon = len(constraint_list)
        poly = build_lagrangian(np.ones(ncon), obj, constraint_list)
        if str(poly) in self.solved_instance.keys():
            data = self.solved_instance[str(poly)]
        else:
            #improve start point if possible using already solved instances
            start = np.ones(ncon)
            bound = np.inf
            for data in self.solved_instance.values():
                if data.success or data.status == 9:
                    if -data.fun > -bound and len(data.x) == ncon:
                        start = data.x
                        bound = -data.fun
            #find lower bound using SONC
            data = constrained_opt(obj, constraint_list, start=start)
            #store {polynomial: solution} as solved, so do not need to compute it twice
            self.solved_instance[str(poly)] = data

        #return if InfeasibleError occurs (unbounded point in SONC decomposition)
        #print([str(p) for p in constraint_list])
        if data.status == 10:
            print(data.message)
            return {'result': SCIP_RESULT.DIDNOTRUN}
        print(data.message, data.status)

        #optimization terminated successfully, lower bound found
        if data.success:
            #print(-data.fun, data.x)
            print('lower bound shifted: ', -data.fun+self.model.getObjoffset())
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}
        #this is still a lower bound, but probably not the best possible
        if data.status >= 4 and not -data.fun==np.inf:
            print('lower bound shifted iteration: ', -data.fun+self.model.getObjoffset())
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}
        return {'result': SCIP_RESULT.DIDNOTRUN}
