#! usr/bin/env python3
from pyscipopt  import SCIP_RESULT, Relax, Term, Expr
from POEM       import Polynomial, InfeasibleError, build_lagrangian, constrained_opt, solve_GP, OptimizationProblem, Constraint
from convert    import ExprToPoly, PolyToExpr

import numpy as np
import re


class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    def relaxinit(self):
        #dictionary to store solved instances and corresponding solution
        self.solved_instance = dict()

    def relaxexec(self):
        """execution method of SONC relaxator"""

        def _nonnegCons(polynomial,rhs,lhs):
            """adds constraints such that constraints >= 0 is satisfied
            :param: polynomial: polynomial expression of constraint
            :param: rhs: right hand side of constraint
            :param: lhs: left hand side of constraint"""
            if not self.model.isInfinity(-lhs):
                polynomial.b[0] -= lhs
                #constraint_list.append(polynomial.copy())
                optProblem.addCons(Constraint(polynomial, '>='))
                polynomial.b[0] += lhs
            if not self.model.isInfinity(rhs):
                polynomial.b[0] -= rhs
                polynomial.b *= -1
                #constraint_list.append(polynomial)
                optProblem.addCons(Constraint(polynomial, '>='))
            return

        def _nonnegCons2(polynomial,rhs, lhs):
            if not self.model.isInfinity(-lhs):
                con = Constraint(polynomial,'>=', lhs)
                optProblem.addCons(con)
            if not self.model.isInfinity(rhs):
                con = Constraint(polynomial, '<=', rhs)
                optProblem.addCons(con)

        optProblem = OptimizationProblem()
        nvars = len(self.model.getVars())
        conss = self.model.getConss()
        #constraint_list = []

        #transform each constraint (type linear, quadratic or expr) into a Polynomial to get lower bound with all constraints used (SCIP -> POEM)
        for cons in conss:
            constype = cons.getType()
            lhs = self.model.getLhs(cons)
            rhs = self.model.getRhs(cons)
            if  constype == 'expr':
                #transform expr of SCIP into Polynomial of POEM
                exprcons = self.model.getConsExprPolyCons(cons)
                polynomial = ExprToPoly(exprcons, nvars)
                polynomial.clean()

                #add constraints to constraint_list with constraints >= 0
                _nonnegCons2(polynomial, rhs, lhs)
 
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

                #add constraints to constraint_list with constraints >= 0
                _nonnegCons2(polynomial, rhs, lhs)
                
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

                #add constraints to constraint_list with constraints >= 0
                _nonnegCons2(polynomial, rhs, lhs)
                
            else:
                #TODO: what to do with non-polynomial constraints? (possibly linear relaxation)
                raise Warning("relaxator not available for constraints of type ", constype)

        #No  constraints of type expr, quadratic or linear, relaxator not applicable
        if optProblem.constraints == []: #constraint_list == []:
            return {'result': SCIP_RESULT.DIDNOTRUN}
        #get Objective as Polynomial
        optProblem.setObjective(ExprToPoly(self.model.getObjective(), nvars))

        #use the Variable bounds as linear constraints
        #TODO: improve usage of bounds, maybe delete Constraints if same as bounds (for Polynomial)
        #TODO: problem if bounds are tightened since it makes completely new polynomial (linear term is changed)
        #TODO: 'polynomial unbounded at point ..' appears if variables do not appear in polynomial (possibility to fix them to 0?)
        for i,y in enumerate(self.model.getVars()):
            if y.getUbLocal() == y.getLbLocal() and y.getUbLocal() == 0:
                equ = False
                #print(cons)
                p = np.zeros(len(self.model.getVars())+1)
                p[0]=1
                p[i+1]=2
                for j in optProblem.constraints: #constraint_list:
                    for k in range(1,len(j.A[0])):
                        #print(i,p,j)
                        if len(p) == len(j.A.T[k,:]) and np.equal(p,j.A.T[k,:]).all():
                            equ = True
                            break
                #TODO: need to also make sure, this constraint only appears if y**2 not in any other constraint
                if not equ:
                    #constraint_list.append(ExprToPoly({Term(y,y):-1.0}, nvars))
                    optProblem.addCons(Constraint(ExprToPoly({Term(y,y):-1.0}, nvars),'>='))
            else:
                if y.getUbLocal() != 1e+20:
                    boundcons = ExprToPoly(Expr({Term(): y.getUbLocal(), Term(y):-1.0}), nvars)
                    #constraint_list.append(boundcons)
                    optProblem.addCons(Constraint(boundcons, '>='))
                if y.getLbLocal() != -1e+20: #TODO: do we also need: and y.getLbLocal() != 0.0:
                    boundcons = ExprToPoly(Expr({Term(): -y.getLbLocal(), Term(y):1.0}), nvars)
                    #constraint_list.append(boundcons)
                    optProblem.addCons(Constraint(boundcons, '>='))
        #print('cons',len(constraint_list))
        #print([str(con) for con in optProblem.constraints])
        #constraint_list = [str(con) + " >= 0" for con in constraint_list]
        #print(constraint_list)

        """Here starts the real computation
            first try to use the GP, if that does not work use scipy"""
        #---try to solve it using GP, so do not need scipy---
        #TODO: sometimes get lower bound > solution, so maybe need to take constant term better into account?
        problem = solve_GP(optProblem)
        if problem.status=='optimal':
            print("lower bound: ", self.model.getObjoffset()-problem.value)
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': self.model.getObjoffset()-problem.value}

        """
        #TODO: this part has to be rewritten since constraint_list is no longer used
        #check if instance was already solved
        #TODO: quite inefficient since we compute the lagrangian again, find easier way to check this
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
            data = constrained_opt(obj, constraint_list, start=[])
            #store {polynomial: solution} as solved, so do not need to compute it twice
            self.solved_instance[str(poly)] = data

        #return if InfeasibleError (status=10) occurs (unbounded point in SONC decomposition)
        if data.status == 10:
            print(data.message)
            return {'result': SCIP_RESULT.DIDNOTRUN}

        #optimization terminated successfully, lower bound found
        if data.success:
            print('lower bound shifted: ', -data.fun+self.model.getObjoffset())
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}

        #this is still a lower bound, but probably not the best possible
        if data.status >= 4 and not -data.fun==np.inf:
            print('lower bound shifted iteration: ', -data.fun+self.model.getObjoffset())
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}

        #TODO: maybe we can use the other status cases given by scipy as well
        """
        #relaxator did not run successfully, did not find a lower bound
        return {'result': SCIP_RESULT.DIDNOTRUN}
