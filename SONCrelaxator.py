#! usr/bin/env python3
from pyscipopt      import SCIP_RESULT, Relax, Term
from Poem.polynomial     import *
from constrained    import *
from Poem.exceptions     import InfeasibleError


import numpy as np
import re


class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    #dictionary to store solved instances with solution
    solved_instance = dict()
    def relaxexec(self):
        """execution method of SONC relaxator"""
        def ExprToPoly(Exp, obj = False):
            """turn pyscipopt.scip.Expr into a Polynomial (POEM)
            :param: Exp: expression of type pyscipopt.scip.Expr
            :param: obj: If Exp is the Objective, need to get the constant term (default = False) of the Objective
            """
            nvar = len(self.model.getVars())
            nterms = len([key for key in Exp])

            #get constant Term 
            if Term() in Exp:
                const = Exp[Term()]
            elif obj:
                #objective does not have constant term explicitly, find separately
                const = self.model.getObjoffset()
                nterms += 1
            else:
                const = 0.0
                nterms += 1

            #turn Expr into Polynomial (POEM)
            A = np.array([np.zeros(nterms)]*nvar)
            b = np.zeros(nterms)
            b[0] = const
            #j = 0 is for constant term
            j = 1
            for key in Exp:
               if not key == Term():
                    for el in tuple(key):
                        i = re.findall(r'x\(?([0-9]+)\)?', str(el))
                        A[int(i[0])][j] += 1
                    b[j] = Exp[key]
                    j += 1
            return Polynomial(A,b) 

        conss = self.model.getConss()
        constraint_list = []

        #flag to indicate whether SONC is necessary, i.e. whether we have a polynomial constraint (not only linear and quadratic)
        polycons = False
        
        #transform each constraint (type linear, quadratic or expr) into a Polynomial to get lower bound with all constraints used
        #TODO: why does SCIP always call relaxator on same constraints (even though  they are the transformed ones)?
        for cons in conss:
            constype = cons.getType()
            if  constype == 'expr':
                #get constraints as as polynomial (POEM)
                exprcons = self.model.getConsExprPolyCons(cons)
                polynomial = ExprToPoly(exprcons)
                polynomial.clean()
                
                #transform into problem with constraints >= 0
                if not self.model.isInfinity(-self.model.getLhs(cons)):
                    polynomial.b[0] -= self.model.getLhs(cons)
                #TODO: if Lhs and Rhs are != inf, need to transform into two constraints with constraints >= 0
                if not self.model.isInfinity(self.model.getRhs(cons)):
                    polynomial.b[0] -= self.model.getRhs(cons)
                    polynomial.b *= -1
                
                constraint_list.append(polynomial)
                polycons = True

            elif constype == 'linear':
                #get linear constraint as Polynomial (POEM)
                coeffdict = self.model.getValsLinear(cons)
                nvar = len(self.model.getVars())
                A = np.array([np.zeros(nvar+1)]*nvar)
                b = np.zeros(nvar+1)
                for i,(key,val) in enumerate(coeffdict.items()):
                    b[i] = val
                    j = re.findall(r'x\(?([0-9]+)\)?', str(key))
                    A[int(j[0])][i] = 1.0
                polynomial = Polynomial(A,b)
                polynomial.clean()

                #transform into problem with constraints >= 0
                #TODO: possible to have Lhs and Rhs != inf
                if not self.model.isInfinity(-self.model.getLhs(cons)):
                    polynomial.b[0] -= self.model.getLhs(cons)
                elif not self.model.isInfinity(self.model.getRhs(cons)):
                    polynomial.b[0] -= self.model.getRhs(cons)
                    polynomial.b *= -1

                constraint_list.append(polynomial)

            elif constype == 'quadratic':
                #get quadratic constraint as Polynomial (POEM)
                bilin, quad, lin = self.model.getTermsQuadratic(cons)
                nvar = len(self.model.getVars())
                #number of terms in constraint +1 for constant term
                nterms = len(bilin)+len(quad)*2+len(lin)+1
                A = np.array([np.zeros(nterms)]*nvar)
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
                #TODO: possible to have Lhs and Rhs != inf
                if not self.model.isInfinity(-self.model.getLhs(cons)):
                    polynomial.b[0] -= self.model.getLhs(cons)
                elif not self.model.isInfinity(self.model.getRhs(cons)):
                    polynomial.b[0] -= self.model.getRhs(cons)
                    polynomial.b *= -1

                constraint_list.append(polynomial)
            else:
                raise Warning("relaxator not available for constraints of type ", constype)

        #get Objective as Polynomial (POEM)
        obj = ExprToPoly(self.model.getObjective(), True)

        #find lower bound using SONC
        #TODO: maybe can use different (faster) solver if only linear and quadratic constraints? (polycons = False)
        data, poly = constrained_opt(obj, constraint_list, SoncRelax.solved_instance)

        #store {polynomial: solution} as solved, so do not need to compute it twice 
        #TODO: uses transformed problem, but this gives always the same constraints/objective, maybe only call in root node? (or examples too easy)
        SoncRelax.solved_instance[str(poly)] = data

        #return if InfeasibleError occurs (unbounded point in SONC decomposition)
        if type(data) == InfeasibleError:
            return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -self.model.infinity()}
        
        #optimization terminated successfully, lower bound found
        if data.success:
            #TODO: do we need to mark the relaxator Solutions? How?
            #self.model.clearRelaxSolVals()
            #for i,v in enumerate(self.model.getVars()):
            #    self.model.setRelaxSolVal(v,data.x[i]) #data.x corresponds to lamb and NOT x
            #print(self.model.isRelaxSolValid()) #gives always False (not exact enough?)
            #self.model.markRelaxSolValid(False)
            print(-data.fun, data.x)
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}
        return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -self.model.infinity()}
