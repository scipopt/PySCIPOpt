#! usr/bin/env python3
from pyscipopt import Model, SCIP_PARAMSETTING, SCIP_RESULT, Expr, Relax, Term, ExprCons
from polynomial import *
from generate_poly import *
import re
import numpy as np
from exceptions import InfeasibleError
from constrained import *
from SONCrelaxator import *

def example():
    m = Model()
    #m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.disablePropagation()
    #m.setIntParam('lp/solvefreq', -1)
    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")
    #x2 = m.addVar(vtype = "C", name = "x2")
    #x3 = m.addVar(vtype = "C", name = "x3")
    #p = 7.93823555685 + 1.80070743765*x0**6 + 4.40432092848*x1**6 + 0.950088417526*x0**2*x1**2 + 2.2408931992*x0**1*x1**2 + 1.86755799015*x0**1*x1**4 - 0.977277879876*x0**2*x1**1 - 0.151357208298*x0**3*x1**1  - 0.103218851794*x0**4*x1**1
    #p = x0**3+ x0**3*x1**2 +2*x1-x1*x0
    #m.addCons(p <= x0+x1)
    #m.addCons(x1-x0>=0)
    #m.addCons(x1**3>=5)
    #m.addCons(x0**3+x1>= 3)
    #m.addCons(x0**2*x1**2+x0**4+x1**4+2*x0*x1>=-15)
    #m.addCons(2*x0<=7)
    #m.addCons(1+x0**2*x1**4+x0**4*x1**2-3*x0**2*x1**2<= 10) #no segfault
    #m.addCons(x0**2+3*x1**2+x0**4*x1**2+2*x0**4*x1**4+x0**2*x1**6+3*x0*x1**2-x0**2*x1**3 <= 100) #segfault, not guarantee finite termination
    #m.addCons(1.5*x0**2*x1**6+x0**6*x1**2+2*x0**2*x1**2-3*x0**3*x1**3 <= 100) #nan
    #m.addCons(1.0 + 3*x0**2*x1**6 +2*x0**6*x1**2 + 6*x0**2*x1**2 - x0*x1**2 - 2*x0**2*x1 - 3*x0**3*x1**3 <=100) #segfault, not guarantee finite termination
    #m.addCons(0.0 + 3.0*x0 + 5.0*x0**2 - 1.0*x1 + 2.0*x1**2 + 4.0*x0*x1 >= 0) #feasible solution found even though, scip says infeasible
    #m.addCons(3*x0**2+4*x0*x1+2*x1+5*x0-x1**2 >= 0) #nan
    """m.addCons(x0**4+x1**4 - 42 >= 0) #works
    m.addCons(x0**4-3*x0+2*x1**2 -1>= 0) #works
    m.addCons(x0**2*x1 + 3*x1 >=x0)
    m.setObjective(x0)    
    """
    
    #m.addCons(x0**2*x1**2+x0*x1+x0**4+x1**4-x0<=10) #does work, no reasonable solutions
    #m.setObjective(-3*x1-x0)
    #m.setObjective(-2*x0+x1)
    
    """
    #unbounded x1 since occurs only with exponent 1
    m.addCons(x0 <= 3)
    m.addCons(x1 <= 4)
    m.addCons(8*x0**3 - 2*x0**4 - 8*x0**2 + x1 >= 2)
    m.addCons(32*x0**3 - 4*x0**4 - 88*x0**2 + 96*x0 + x1 >= 36)
    m.setObjective(-x0-x1)
    """
    """
    #Problem only containing linear and quadratic constraints, using sonc anyway takes long time
    m.addCons(0.25*x1 - 0.0625*x1**2 - 0.0625*x0**2 + 0.5*x0 <= 1)
    m.addCons(0.0714285714285714*x1**2 + 0.0714285714285714*x0**2 - 0.428571428571429*x1 - 0.428571428571429*x0 <= -1)
    m.addCons((1<=x0)<=5.5)
    m.addCons((1<=x1)<=5.5)
    m.setObjective(x1)
    """
    """
    #TODO:problem if unboundedness occurs after some iterations, do not get -1e+20, but some value, hopefully solved by using flag success of scipy
    #terminates, gives reasonable lower bound
    m.addCons(0<=0.5+x0**2*x1**1-x0**6*x1**4-x0**3*x1**3)
    m.addCons(-x0<=1-x0**4-x0**2*x1**4)
    m.setObjective(3+x0) 
    """
    
    #Henning E-Mail Test Polynomials modified, terminates but needs some time(less than 1 min), good lower bound (exact)!!!
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    m.addCons(x0**2*x1 + 3*x1 >=0)
    m.setObjective(-x0)
    
    """
    #Henning E-Mail Test Polynomials modified, reasonable lower bound
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    m.addCons(-x0**2*x1 - 3*x1 +0.5*x0 >=0)
    m.setObjective(-0.5*x0)
    """
    """
    #Henning E-Mail Test Polynomials modified, terminates but needs some time, lowerbound:-6.6, solution: -1.5
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    #m.addCons(x0**2*x1 >=0)
    m.setObjective(3*x1-x0)
    """
    """
    #unbounded 
    m.addCons(x0+x0*x1+x0**4*x1**2>=0)
    m.addCons(0.5+x0**2*x1**4-x0**2*x1**6>=0)
    m.setObjective(1-x0)
    """
    """
    #Example 4.8 (6)i. in Lower Bounds for a polynomial (Ghasemi, Marshall), lower bound: -1.6, sol: 0.0
    m.addCons(1-2*x1-6*x0**2-x0**4>=0)
    m.addCons(-x0**3-x1**4>=0)
    m.setObjective(x0+x1)
    """
    """
    #Example 4.8 (8) very slow, but gives good bound (-7.01e-05, sol: -9.06e-07)
    m.addCons(2*x0**2-x1>=0)
    m.addCons(x1-x0**4*x1+x1**5-x0**6-x1**6>=0)
    m.addCons(x1-5*x0**2+x0**4*x1-x0**6-x1**6>=0)
    m.setObjective(-x1)
    """
    """
    #terminates, reasonable lower bound
    m.addCons(1-x0**4-x1**4>=0)
    m.addCons(x0**3+2>=0)
    m.setObjective(x0-2)
    """
    """
    #ex14_1_1, unbounded point x3
    m.addCons(-x3+x0<=0)
    m.addCons(-x3+x0>=0)
    m.addCons(2*x2**2 + 4*x1*x2 - 42*x1 + 4*x1**3 - x3 <= 14)
    m.addCons(-2*x2**2 - 4*x1*x2 + 42*x1 - 4*x1**3 - x3 <= -14)
    m.addCons(2*x1**2 + 4*x1*x2 - 26*x2 + 4*x2**3 - x3 <= 22)
    m.addCons(-2*x1**2 - 4*x1*x2 + 26*x2 - 4*x2**3 - x3 <= -22)
    m.addCons((-5<=x1)<=5)
    m.addCons((-5<=x2)<=5)
    m.setObjective(x0)
    """
    """
    #ex4_1_1, unobounded point x0
    m.addCons(-(x1**6 - 2.08*x1**5 + 0.4875*x1**4 + 7.1*x1**3 - 3.95*x1**2 - x1) + x0 <= 0.1)
    m.addCons(-(x1**6 - 2.08*x1**5 + 0.4875*x1**4 + 7.1*x1**3 - 3.95*x1**2 - x1) + x0 >= 0.1)
    m.addCons((-2<=x1)<=11)
    m.setObjective(x0)
    """
    """
    #st_e19, unbounded point x0
    m.addCons(-x1+x2 <= 8)
    m.addCons(x1**2-2*x1+x2<=-2)
    m.addCons(-(x1**4-14*x1*2+24*x1-x2**2)+x0 == 0)
    m.addCons((-8<=x1)<=10)
    m.addCons(x2<=10)
    m.setObjective(-x0)
    """
    return m

"""def Poly_to_Expr(p,m):
#rewrite polynomial p in as dict in the way it is used in Expr (uses __dict__ of class polynomial, but this gives the wrong dict for const Term), input: Polynomials object p
    dictP = p.__dict__()
    d = dict()
    print(dictP)
    for key in dictP.keys():
        i = 0
        t =[]
        var = m.getVars()
        varstr = [None]*len(var)
        for j in range(len(var)):
            varstr[j] = str(var[j])
        for el in key[1:]:
            f = 'x' + str(i) 
            for _ in range(el):
                t.append(var[varstr.index(f)])
            i+=1
        print(t)
        term = Term()
        for el in t:
            term += Term(el)
        #print(term)
        d[term] = dictP[key]
        #print(t)
    return Expr(d)

def tuple_to_list(t):
#convert a tuple of form (x0,..,x0,...xn,...xn) into list with entry[i] = number of times xi can be found in tuple, input: tuple t
    nmax = max([int(i) for i in re.findall(r'x\(?([0-9]+)\)?', str(t))]) + 1
    power = [0 for _ in range(nmax)]
    for el in t:
        i = re.findall(r'x\(?([0-9]+)\)?', str(el))
        power[int(i[0])] += 1
    return power

def Expr_to_Poly(Exprs):
#convert Expr object into Polynomial object to be able to use Polynomial class, input: Expr object Exprs
#TODO: reimplement function using matrices A,b instead of strings, and make sure that all constraints and objective have the same size (with len(A)=number of variables existing in problem)
    try:
        const = Exprs[Term()]
    except:
        const = 0.0
    P = str(const)
    for key in Exprs:
       if not key == Term():
            power = tuple_to_list(tuple(key))
            term = str(Exprs[key])+'*'+'*'.join('x'+str(i)+'**'+str(power[i]) for i in range(len(power)) if power[i] != 0) 
            P += '+' + term
    P = P.replace('+-','-')
    if P =='0.0': return Polynomial('0.0*x0')
    else: return Polynomial(P)
"""
"""def ExprToPoly(Exp, obj = False):
    #print(Exp)
    nvar = len(m.getVars())
    #print(nvar)
    nterms = len([key for key in Exp])
    #print(nterms)
    if Term() in Exp:
        const = Exp[Term()]
        #print('const=',const)
    elif obj:
        const = m.getObjoffset()
        nterms += 1
    else:
        const = 0.0
        nterms += 1
    #print(nterms)
    A = np.array([np.zeros(nterms)]*nvar)
    b = np.zeros(nterms)
    #print(A)
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
       """
"""
class SoncRelax(Relax):
    def relaxexec(self):
        conss = m.getConss()
        count = 0
        constrained_list = []
        #flag to indicate whether SONC is necessary, i.e. whether we have a polynomial constraint (not only linear and quadratic)
        polycons = False
        for con in conss:
            contype = con.getType()
            feasible = False
            if  contype == 'expr':
                polycons = True
                #get constraints as as polynomial (POEM)
                exprcon = m.getConsExprPolyCons(con)
                polynomial = ExprToPoly(exprcon)
                polynomial.clean()
                #transform into problem with constraints >= 0
                if not m.isInfinity(-m.getLhs(con)):
                    polynomial.b[0] -= m.getLhs(con)
                    print('lhs=',m.getLhs(con))
                #TODO: need to make sure that constraint >= 0, so need to switch signs everywhere and then add rhs (subtract -rhs)
                if not m.isInfinity(m.getRhs(con)):
                    print('rhs=',m.getRhs(con))
                    polynomial.b[0] -= m.getRhs(con)
                    polynomial.b *= -1
                    print('b=',polynomial.b)
                print('A=',polynomial.A,'b=', polynomial.b)
                constrained_list.append(polynomial)
                #if polynomial.monomial_squares == []:
                #    print("Polynomial cannot be decomposed via SONC.")
                #    continue
                #TODO: run_sonc() does not always give solutions, need to make sure this works (treat the errors)
                #try:
                #    polynomial.run_sonc()
                #    feasible = True
                #except InfeasibleError:
                #    feasible = False
            elif contype == 'linear':
                exprcon = m.getValsLinear(con)
                nvar = len(m.getVars())
                A = np.array([np.zeros(nvar+1)]*nvar)
                b = np.zeros(nvar+1)
                i = 0
                for (key,val) in exprcon.items():
                    b[i] = val
                    j = re.findall(r'x\(?([0-9]+)\)?', str(key))
                    A[int(j[0])][i] = 1.0
                    i += 1
                    #polynomial += '+' + str(val) + '*' + str(key) #polynomial.replace('+-', '-')
                polynomial = Polynomial(A,b)
                polynomial.clean()

                if not m.isInfinity(-m.getLhs(con)):
                    polynomial.b[0] -= m.getLhs(con)
                elif not m.isInfinity(m.getRhs(con)):
                    polynomial.b[0] -= m.getRhs(con)
                    polynomial.b *= -1
                constrained_list.append(polynomial)
                #print('linear=',polynomial.A,polynomial.b)
            elif contype == 'quadratic':
                bilin, quad, lin = m.getTermsQuadratic(con)
                #print(bilin, quad, lin)
                #print(len(bilin)+len(quad)*2+len(lin))
                nvar = len(m.getVars())
                nterms = len(bilin)+len(quad)*2+len(lin)+1
                A = np.array([np.zeros(nterms)]*nvar)
                b = np.zeros(nterms)
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
                    i += 1"""
"""polynomial = '0.0'
                for el in lin:
                    polynomial += '+' + str(el[1]) + '*' + str(el[0])
                for el in quad:
                    if el[-1] != 0.0:
                        polynomial += '+' + str(el[-1]) + '*' + str(el[0])
                    if el[1] != 0.0:
                        polynomial += '+' + str(el[1]) + '*' + str(el[0]) + '**' + '2.0'
                
                for el in bilin:
                    if el[-1] != 0.0:
                        polynomial += '+' + str(el[-1]) + '*' + str(el[0]) + '*' + str(el[1])
                polynomial = Polynomial(polynomial.replace('+-', '-'))"""
"""polynomial = Polynomial(A,b)
                polynomial.clean()
                #print(polynomial)
                
                if not m.isInfinity(-m.getLhs(con)):
                    polynomial.b[0] -= m.getLhs(con)
                elif not m.isInfinity(m.getRhs(con)):
                    polynomial.b[0] -= m.getRhs(con)
                    polynomial.b *= -1
                #print(polynomial)
                constrained_list.append(polynomial)
        obj = ExprToPoly(m.getObjective(), True)
        #print(obj.A,obj.b)
        #print("obj =", obj)
        if not polycons:
            #TODO: maybe can run SOS/trivial_solution instead if we know, only linear and quadratic constraints??
            return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -m.infinity()}
        data = constrained_opt(obj, constrained_list)
        print(data)
        #print('fmin = ', fmin)
        if data.success: #np.isnan(xmin).all():
            print(data)
            print(-data.fun)
            #TODO: does lowerbound need to take into account the constant terms of objective or constraints??
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': -data.fun}
        return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -m.infinity()}

      """          
"""        #TODO: make sure to only use constraints the are consexpr (so only nonlinear constraints that are polynomials) -> cons.getType() == 'expr'
        #2: relax constraints via Sonc relaxations
        #3: give relaxed constraints back into SCIP, let SCIP work with these constraints
"""    
if __name__=="__main__":
    m = example()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    relaxator = SoncRelax()
    m.includeRelax(relaxator,"SONCRelaxator", "Relaxator using SONC decompositions", freq=1)
    m.optimize()
    var = m.getVars()
    for v in var:
        print(v,"=", m.getVal(v))
    print(m.getObjVal())
    
    n = example()
    n.setPresolve(SCIP_PARAMSETTING.OFF)
    n.disablePropagation()
    n.optimize()
    va = n.getVars()
    for v in va:
        print(v, "=", n.getVal(v))
    print(n.getObjVal())
    
    """#need to make sure, we have verify == 1
                #need a submodel to work on with the new constraints, but original variables
                #print("sols = ",polynomial.solution)
                if polynomial.solution['verify'] == 1:
                    count += 1
                    #submodel.hideOutput()
                    #submodel.setPresolve(SCIP_PARAMSETTING.OFF)
                    #get_decomposition() does not know "trivial" as solver strategy (ie. SOS), maybe make an extra case(since we don't need to do much in that case anyway)
                    
                    decomps = polynomial.get_decomposition()
                    polynomial.print_solutions()
                    print(len(decomps))
                    for poly in decomps:
                        #print("poly = ",poly.A, poly.b)
                        lhs = m.getLhs(con)
                        rhs = m.getRhs(con)
                        exprpoly = Poly_to_Expr(poly,m)
                        #TODO: we always get same submodel over and over again (neverending loop) or maybe not, since relaxator does not get included again (problem only if directly in m, not if in submodel?)
                        print(lhs)
                        constraint = ExprCons(exprpoly, lhs, rhs)
                        print(constraint)
                        submodel.addCons(constraint, name = 't' + str(len(submodel.getConss())+1))
                    #print(exprpoly, rhs)
            #else:
            #    submodel.addPyCons(con) #segfault since not allowed to use constraints from original problem!!
            print(count)
        #m.optimize()
        if count > 0:
            submodel.setObjective(m.getObjective())
            print(submodel.getConss())
            submodel.optimize()
            print("optimized")
            print(submodel.getVal(var[0]))
            print(submodel.getVal(var[1]))
            print(submodel.getObjVal())
    
            sol = submodel.getBestSol()
            lowerbound = submodel.getSolObjVal(sol)
            print("lowerbound = ", lowerbound-polynomial.solution['opt'])
            return {'result': SCIP_RESULT.SUCCESS, 'lowerbound':lowerbound-polynomial.solution['opt']}
        else: 
            return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': m.infinity()}"""
