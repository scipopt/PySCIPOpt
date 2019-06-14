#! usr/bin/env python3
from pyscipopt      import SCIP_RESULT, Relax, Term
from polynomial     import *
from constrained    import *

import numpy as np
import re





class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    def relaxexec(self):
        """execution method of SONC relaxator"""
        def ExprToPoly(Exp, obj = False):
            """turn pyscipopt.scip.Expr into a Polynomial (POEM)
            :param: Exp: expression of type pyscipopt.scip.Expr
            :param: obj: If Exp is the Objective, need to get the constant term (default = False)
            """
            nvar = len(self.model.getVars())
            nterms = len([key for key in Exp])

            #get constant Term 
            if Term() in Exp:
                const = Exp[Term()]
            elif obj:
                const = self.model.getObjoffset()
                nterms += 1
            else:
                const = 0.0
                nterms += 1

            #turn Expr into Polynomial (POEM)
            A = np.array([np.zeros(nterms)]*nvar)
            b = np.zeros(nterms)
            b[0] = const
            #number of terms, 0 is for constant term
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
                #TODO: need to make sure that constraint >= 0, so need to switch signs everywhere and then add rhs (subtract -rhs)
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

        if not polycons:
            #have only linear and quadratic constraints, do not need SONC
            #TODO: maybe can run easy solver? 
            return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -self.model.infinity()}

        #find lower bound using SONC
        data = constrained_opt(obj, constraint_list)
        print(data)
        if data.success:
            #optimization terminated successfully, lower bound found
            print(-data.fun)
            #TODO: does lowerbound need to take into account the constant terms of objective or constraints??
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}
        return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -self.model.infinity()}
