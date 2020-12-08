#! usr/bin/env python3
from pyscipopt  import SCIP_RESULT, Relax
from POEM       import Polynomial, solve_GP, OptimizationProblem, Constraint
from convert    import ExprToPoly, PolyToExpr
from SONCpreprocess import preprocess
#from scipyRelax import scipyRelax

import numpy as np
import re

#scipy=False

class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    def relaxinit(self):
        #dictionary to store solved instances and corresponding solution
        self.solved_instance = dict()

    def relaxexec(self):
        """execution method of SONC relaxator"""

        def _nonnegCons(polynomial,rhs, lhs):
            if not self.model.isInfinity(-lhs):
                con = Constraint(polynomial,'>=', lhs)
                optProblem.addCons(con)
            if not self.model.isInfinity(rhs):
                con = Constraint(polynomial, '<=', rhs)
                optProblem.addCons(con)

        optProblem = OptimizationProblem()
        nvars = len(self.model.getVars())
        conss = self.model.getConss()

        #transform each constraint (type linear, quadratic or expr) into a Polynomial to get lower bound with all constraints used (SCIP -> POEM)
        for cons in conss:
            constype = cons.getType()
            lhs = self.model.getLhs(cons)
            rhs = self.model.getRhs(cons)
            #print('lhs',lhs,'rhs',rhs)
            if  constype == 'expr':
                #transform expr of SCIP into Polynomial of POEM
                exprcons = self.model.getConsExprPolyCons(cons)
                polynomial = ExprToPoly(exprcons, nvars)

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
                
            else:
                warnings.warn("relaxator not available for constraints of type ", constype)

            if constype in ['linear', 'quadratic', 'expr']:
                polynomial.clean()
                #add constraints to optProblem.constraints
                _nonnegCons(polynomial, rhs, lhs)

        #if at most 50% of the constraints is used, relaxator is not used
        if len(optProblem.constraints) <= self.model.getNConss()//2:
            return {'result': SCIP_RESULT.DIDNOTRUN}

        #get Objective as Polynomial
        optProblem.setObjective(ExprToPoly(self.model.getObjective(), nvars))

        #use preprocessing to get a structure that (hopefully) fits for POEM
        optProblem.infinity = self.model.infinity() #make sure infinity means the same for both SCIP and POEM (i.e. if SCIP infinity is changed by user)
        optProblem = preprocess(optProblem, self.model.getVars())
        #try to solve problem using GP 
        optProblem = solve_GP(optProblem)

        if optProblem.solve_time == 'optimal':
            #print("lower bound: ", self.model.getObjoffset()-problem.value)
            #print("sol: ", optProblem.lowerbound)
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': optProblem.lowerbound}

        #if scipy:
        #    return scipyRelax(optProblem,self)

        #relaxator did not run successfully, did not find a lower bound
        return {'result': SCIP_RESULT.DIDNOTRUN}
