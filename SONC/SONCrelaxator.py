#! usr/bin/env python3
from pyscipopt  import SCIP_RESULT, Relax
from POEM       import Polynomial, solve_GP, OptimizationProblem, Constraint, dt2sec
from SONC.convert    import ExprToPoly
from SONC.SONCpreprocess import preprocess
#from scipyRelax import scipyRelax

import numpy as np
import re
import warnings
from datetime import datetime
#scipy=False

class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    def relaxinit(self):
        #list to store solutions of relaxator and LP relaxation
        self.solved_instance = []

        #list to store current variable bounds
        self.varBounds = [(y.getUbLocal(), y.getLbLocal()) for y in self.model.getVars()]
    
    def relaxexit(self):
        #output of all relaxator instance solved and dual bounds and LP relaxations at that time
        print('SONC relaxator                                   current dual bounds                     LP relaxation')
        for el in self.solved_instance:
            print(el)
            if len(el)==3:
                print((1-el[0][1]/el[2][1])*100,'%')

    def relaxexec(self):
        """execution method of SONC relaxator"""

        def _nonnegCons(polynomial,rhs, lhs):
            if not self.model.isInfinity(-lhs):
                con = Constraint(polynomial,'>=', lhs)
                optProblem.addCons(con)
            if not self.model.isInfinity(rhs):
                con = Constraint(polynomial, '<=', rhs)
                optProblem.addCons(con)
        
        t0 = datetime.now()
        #do not reuse the relaxator if the variable bounds have not changed
        if len(self.solved_instance) != 0 and all(self.varBounds[i] == (y.getUbLocal(), y.getLbLocal()) for i,y in enumerate(self.model.getVars())):
            if self.solved_instance[-1][0][0]=='optimal':
                return {'result':SCIP_RESULT.SUCCESS, 'lowerbound': self.solved_instance[-1][0][1]}
            else:
                return {'result':SCIP_RESULT.DIDNOTRUN}
        #store new variable bounds if they are changed
        self.varBounds = [(y.getUbLocal(), y.getLbLocal()) for y in self.model.getVars()]
        
        optProblem = OptimizationProblem()
        nvars = len(self.model.getVars())
        conss = self.model.getConss()
        
        #transform each constraint (type linear, quadratic or expr) into a Polynomial to get lower bound with all constraints used (SCIP -> POEM)
        for cons in conss:
            constype = cons.getType()
            if constype in ['expr', 'linear', 'quadratic']:
                lhs = self.model.getLhs(cons)
                rhs = self.model.getRhs(cons)
            if  constype == 'expr':
                #transform expr of SCIP into Polynomial of POEM
                exprcons = self.model.getConsExprPolyCons(cons)
                polynomial = ExprToPoly(exprcons, nvars, self.model.getVars())

            elif constype == 'linear':
                #get linear constraint as Polynomial (POEM)
                coeffdict = self.model.getValsLinear(cons)
                A = np.array([np.zeros(nvars+1)]*nvars)
                b = np.zeros(nvars+1)
                for i,(key,val) in enumerate(coeffdict.items()):
                    b[i] = val
                    #j = re.findall(r'x\(?([0-9]+)\)?', str(key))
                    for j,var in enumerate(self.model.getVars()):
                        if re.search(str(var), str(key)):
                            A[j][i] +=1
                            break
                    #A[int(j[0])][i] = 1.0
                polynomial = Polynomial(A,b)
                
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
                    for j, var in enumerate(self.model.getVars()):
                        if re.search(str(var), str(el[0])):
                            #j = int(str(el[0])[-1])
                            A[j][i] = 1.0
                        elif re.search(str(var), str(el[1])):
                            #j = int(str(el[1])[-1])
                            A[j][i] = 1.0
                    i += 1
                for el in quad:
                    #j = int(str(el[0])[-1])
                    b[i] = el[1]
                    for j, var in enumerate(self.model.getVars()):
                        if re.search(str(var), str(el[0])):
                            A[j][i] = 2.0
                            i += 1
                            b[i] = el[2]
                            A[j][i] = 1.0
                            i += 1
                            break
                for el in lin:
                    b[i] = el[1]
                    #j = int(str(el[0])[-1])
                    for j, var in enumerate(self.model.getVars()):
                        if re.search(str(var), str(el[0])):
                            A[j][i] = 1.0
                            i += 1
                polynomial = Polynomial(A,b)
                
            else:
                warnings.warn('relaxator not available for constraints of type %s' % constype)

            if constype in ['linear', 'quadratic', 'expr']:
                polynomial.clean()
                #add constraints to optProblem.constraints
                _nonnegCons(polynomial, rhs, lhs)
        
        #if at most 50% of the constraints is used, relaxator is not used
        if len(optProblem.constraints) <= self.model.getNConss()//2:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        #get Objective as Polynomial
        optProblem.setObjective(ExprToPoly(self.model.getObjective(), nvars, self.model.getVars()))

        #use preprocessing to get a structure that (hopefully) fits for POEM
        optProblem.infinity = self.model.infinity() #make sure infinity means the same for both SCIP and POEM (i.e. if SCIP infinity is changed by user)
        optProblem = preprocess(optProblem, self.model.getVars())
        
        #try to solve problem using GP 
        optProblem = solve_GP(optProblem)
        
        #store solved instances for output in the end
        self.solved_instance.append([(optProblem.status, optProblem.lowerbound,dt2sec(datetime.now()-t0)),(self.model.getDualbound(), self.model.getDualboundRoot())])
        if self.model.getLPSolstat() != 0:
            self.solved_instance[-1].append((self.model.getLPSolstat(), self.model.getLPObjVal()))

        if optProblem.status == 'optimal':
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': optProblem.lowerbound}

        #if scipy:
        #    return scipyRelax(optProblem,self)

        #relaxator did not run successfully, did not find a lower bound
        return {'result': SCIP_RESULT.DIDNOTFIND}
