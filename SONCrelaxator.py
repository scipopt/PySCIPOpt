#! usr/bin/env python3
from pyscipopt       import SCIP_RESULT, Relax, Term, Expr
from constrained     import *
from Poem.polynomial import *
from Poem.exceptions import InfeasibleError


import numpy as np
import re


class SoncRelax(Relax):
    """Relaxator class using SONCs to find a lower bound"""
    def relaxinit(self):
        #dictionary to store solved instances with solution
        self.solved_instance = dict()

    def relaxexec(self):
        """execution method of SONC relaxator"""
        def ExprToPoly(Exp):
            """turn pyscipopt.scip.Expr into a Polynomial (POEM)
            :param: Exp: expression of type pyscipopt.scip.Expr
            """
            nvar = len(self.model.getVars())
            nterms = len([key for key in Exp])

            #get constant Term
            if Term() in Exp:
                const = Exp[Term()]
            else:
                const = 0.0
                nterms += 1

            #turn Expr into Polynomial (POEM)
            A = np.array([np.zeros(nterms)]*nvar)
            b = np.zeros(nterms)
            b[0] = const
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
                    constraint_list.append(polynomial.copy())
                    polynomial.b[0] += self.model.getLhs(cons)
                if not self.model.isInfinity(self.model.getRhs(cons)):
                    polynomial.b[0] -= self.model.getRhs(cons)
                    polynomial.b *= -1
                    constraint_list.append(polynomial)

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
                    constraint_list.append(polynomial.copy())
                    polynomial.b[0] += self.model.getLhs(cons)
                if not self.model.isInfinity(self.model.getRhs(cons)):
                    polynomial.b[0] -= self.model.getRhs(cons)
                    polynomial.b *= -1
                    constraint_list.append(polynomial)
                #print(self.model.getLhs(cons),self.model.getRhs(cons))
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
                    constraint_list.append(polynomial.copy())
                    polynomial.b[0] += self.model.getLhs(cons)
                if not self.model.isInfinity(self.model.getRhs(cons)):
                    polynomial.b[0] -= self.model.getRhs(cons)
                    polynomial.b *= -1
                    constraint_list.append(polynomial)
            else:
                raise Warning("relaxator not available for constraints of type ", constype)

        #No  constraints of type expr, quadratic or linear
        if constraint_list == []:
            return {'result': SCIP_RESULT.DIDNOTRUN}
        #get Objective as Polynomial (POEM)
        obj = ExprToPoly(self.model.getObjective())

        #use the Variable bounds as well
        #TODO: improve usage of bounds, maybe delete Constraints if same as bounds (for Polynomial)
        #TODO: problem if bounds are tightened since it makes completely new polynomial (linear term is changed)
        for y in self.model.getVars():
            if y.getUbLocal() != 1e+20:
                boundcons = ExprToPoly(Expr({Term(): y.getUbLocal(), Term(y):-1.0})) #Polynomial('-'.join((str(y),str(y.getLbLocal()))))
                #if not boundcons in constraint_list:
                #print(boundcons)
                constraint_list.append(boundcons)
            if y.getLbLocal != -1e+20: # and y.getLbLocal != 0.0:
                boundcons = ExprToPoly(Expr({Term(): -y.getLbLocal(), Term(y):1.0}))#Polynomial('+'.join((str(y),str(y.getUbLocal()))))
                #if not boundcons in constraint_list:
                #print(boundcons)
                constraint_list.append(boundcons)
            #print('original: ', y.getLbOriginal(), y.getUbOriginal())
            #print('global: ', y.getLbGlobal(), y.getUbGlobal())
            #print('local: ', y.getLbLocal(), y.getUbLocal())
        #print([str(con) for con in constraint_list])
        #find lower bound using SONC
        #TODO: maybe can use different (faster) solver if only linear and quadratic constraints?
        data, poly = constrained_opt(obj, constraint_list, self.solved_instance)
        #store {polynomial: solution} as solved, so do not need to compute it twice
        self.solved_instance[str(poly)] = data
        #return if InfeasibleError occurs (unbounded point in SONC decomposition)
        #print([str(p) for p in constraint_list])
        if type(data) == InfeasibleError:
            print(data)
            return {'result': SCIP_RESULT.DIDNOTRUN}

        #optimization terminated successfully, lower bound found
        if data.success:
            #print(-data.fun, data.x)
            print('lower bound shifted: ', -data.fun+self.model.getObjoffset())
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}
        return {'result': SCIP_RESULT.DIDNOTRUN}
