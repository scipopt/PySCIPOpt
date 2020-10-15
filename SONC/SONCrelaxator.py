#! usr/bin/env python3
from pyscipopt  import SCIP_RESULT, Relax, Term, Expr
from POEM       import Polynomial, InfeasibleError, build_lagrangian, constrained_scipy, solve_GP, OptimizationProblem, Constraint
from convert    import ExprToPoly, PolyToExpr
from SONCpreprocess import preprocess

import numpy as np
import re

scipy=False

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

        #use the Variable bounds as linear constraints
        #TODO: improve usage of bounds, maybe delete Constraints if same as bounds (for Polynomial)
        #TODO: problem if bounds are tightened since it makes completely new polynomial (linear term is changed)
        #TODO: 'polynomial unbounded at point ..' appears if variables do not appear in polynomial (possibility to fix them to 0?)
        #TODO: use transformed problem
        #print('transformed', self.model.getVars(transformed=True))
        """for i,y in enumerate(self.model.getVars(transformed=False)):
            if y.getUbLocal() == y.getLbLocal() and y.getUbLocal() == 0:
                #TODO: if variables are fixed, delete them from the objective and constraints, by evaluating in fixed value (need to make sure the variables order is still correct)
                equ = False
                #print(cons)
                p = np.zeros(len(self.model.getVars())+1)
                p[0]=1
                p[i+1]=2
                for j in optProblem.constraints: 
                    for k in range(1,len(j.A[0])):
                        #print(i,p,j)
                        if len(p) == len(j.A.T[k,:]) and np.equal(p,j.A.T[k,:]).all():
                            equ = True
                            break
                #TODO: need to also make sure, this constraint only appears if y**2 not in any other constraint
                if not equ:
                    optProblem.addCons(Constraint(ExprToPoly({Term(y,y):1.0}, nvars),'<='))
            else:
                if y.getUbLocal() != self.model.infinity():
                    boundcons = ExprToPoly(Expr({Term(): -y.getUbLocal(), Term(y):1.0}), nvars)
                    optProblem.addCons(Constraint(boundcons, '<='))
                if y.getLbLocal() != -self.model.infinity() and y.getLbLocal() != 0:
                    boundcons = ExprToPoly(Expr({Term(): y.getLbLocal(), Term(y):-1.0}), nvars)
                    optProblem.addCons(Constraint(boundcons, '<='))"""
        #print('before pre',optProblem)
        #use preprocessing to get a structure that (hopefully) fits for POEM
        optProblem = preprocess(optProblem, self.model)
        #print('after pre', optProblem)
        """Here starts the real computation
            first try to use the GP, if that does not work use scipy (optional)"""
        #try to solve problem using GP 
        problem = solve_GP(optProblem)

        if optProblem.constraints!=[] and problem.status=='optimal' or type(problem)==dict() and problem['verify']:
            #print("lower bound: ", self.model.getObjoffset()-problem.value)
            #print("sol: ", optProblem.lowerbound)
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': optProblem.lowerbound}

        if scipy:
            #TODO: where can optional param scipy be changed during call?!, do we really still need that?
            #check if instance was already solved
            #TODO: quite inefficient since we compute the lagrangian again, find easier way to check this
            ncon = len(optProblem.constraints)
            poly = build_lagrangian(np.ones(ncon), optProblem.objective, optProblem.constraints)
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
                data = constrained_scipy(optProblem.objective, optProblem.constraints, start=[])
                #store {polynomial: solution} as solved, so do not need to compute it twice
                self.solved_instance[str(poly)] = data

            """List of possible outputs for data.status and meaning (from solver SLSQP):
                -1 : Gradient evaluation required (g & a)
                 0 : Optimization terminated successfully
                 1 : Function evaluation required (f & c)
                 2 : More equality constraints than independent variables
                 3 : More than 3*n iterations in LSQ subproblem
                 4 : Inequality constraints incompatible
                 5 : Singular matrix E in LSQ subproblem
                 6 : Singular matrix C in LSQ subproblem
                 7 : Rank-deficient equality constraint subproblem HFTI
                 8 : Positive directional derivative for linesearch
                 9 : Iteration limit exceeded

            customly created status:
                10 : InfeasibleError occured in POEM
            """
            #TODO: maybe we can use the other status cases given by scipy as well

            #return if InfeasibleError (status=10) occurs (unbounded point in SONC decomposition)
            if data.status == 10:
                print(data.message)
                return {'result': SCIP_RESULT.DIDNOTRUN}

            #optimization terminated successfully, lower bound found
            if data.success: #data.status=0
                print('lower bound shifted: ', -data.fun+self.model.getObjoffset())
                return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}

            #this is still a lower bound, but probably not the best possible
            if data.status >= 4 and not -data.fun==np.inf: #TODO: maybe independent of data.status if data.fun always exists?
                print('lower bound shifted iteration: ', -data.fun+self.model.getObjoffset())
                return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}

        #relaxator did not run successfully, did not find a lower bound
        return {'result': SCIP_RESULT.DIDNOTRUN}
