from pyscipopt import Model, SCIP_PARAMSETTING, SCIP_RESULT
from pyscipopt.scip import Relax, Term, Expr, ExprCons
from polynomial import *
from generate_poly import *
import re
import numpy as np
from exceptions import InfeasibleError
from constrained import *


def example():
    m = Model()
    #m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.disablePropagation()
    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")
    x2 = m.addVar(vtype = "C", name = "x2")
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
    m.addCons(x0**4+x1**4 - 42 >= 0) #works
    m.addCons(x0**4-3*x0+2*x1**2 -1>= 0) #works
    m.addCons(x0**2*x1 + 3*x1-x0 >=x2)
    #m.addCons(x0**2*x1**2+x0*x1+x0**4+x1**4-x0<=10) #does work, no reasonable solutions
    #m.setObjective(-3*x1-x0)
    #m.setObjective(-2*x0+x1)
    m.setObjective(x2)
    #TODO: too often unbounded, something is wrong in the way the polynomials are presented
    """
    m.addCons(x0 <= 3)
    m.addCons(x1 <= 4)
    m.addCons(8*x0**3 - 2*x0**4 - 8*x0**2 + x1 <= 2)
    m.addCons(32*x0**3 - 4*x0**4 - 88*x0**2 + 96*x0 + x1 <= 36)
    m.setObjective(-x0-x1)
    """
    """
    #Problem only containing linear and quadratic constraints
    m.addCons(0.25*x1 - 0.0625*x1**2 - 0.0625*x0**2 + 0.5*x0 <= 1)
    m.addCons(0.0714285714285714*x1**2 + 0.0714285714285714*x0**2 - 0.428571428571429*x1 - 0.428571428571429*x0 <= -1)
    m.addCons((1<=x0)<=5.5)
    m.addCons((1<=x1)<=5.5)
    m.setObjective(x1)
    """
    """
    #lower bound at 0.25, even though minimum at 0.0??
    m.addCons(0<=0.5+x0**2*x1**1-x0**6*x1**4-x0**3*x1**3)
    m.addCons(x2<=1+x0**4+x0**2*x1**4)
    m.setObjective(x2) 
    """
    """
    #Henning E-Mail Test Polynomials
    m.addCons(x0**4+x1**4-42>=0)
    m.addCons(x0**4-3*x0+2*x1**2-1>=0)
    m.addCons(x0**2*x1 + 3*x1 - x0>=x2)
    m.setObjective(x2)
    """
    """
    m.addCons(-x0+1>=0)
    m.addCons(x0+1>=0)
    m.addCons(-x1+1>=0)
    m.addCons(x1+1>=0)
    m.addCons(x0**3+x1**3-x0*x1+4>=x2)
    m.setObjective(x2)
    """
    return m

def Poly_to_Expr(p,m):
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

def ExprToPoly(Exp):
    #print(Exp)
    nvar = len(m.getVars())
    #print(nvar)
    nterms = len([key for key in Exp])
    #print(nterms)
    
    if Term() in Exp:
        const = Exp[Term()]
        #print('const=',const)
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
            
class SoncRelax(Relax):
#TODO: problems in Polynomial class whenever Polynomial has no constant term, so maybe just add/sub the lhs/rhs to get constant term? otherwise need to change polynomial class
    def relaxexec(self):
        #1: get the constraints using SCIP functions in cons_expr
        conss = m.getConss()
        #print("relaxator used")
        count = 0
        submodel = Model()                    
        #var = m.getVars()
        constrained_list = []
        #for v in var:
        #    submodel.addVar(v.name)
        #flag to indicate whether SONC is necessary, i.e. whether we have a polynomial constraint (not only linear and quadratic)
        polycons = False
        for con in conss:
            contype = con.getType()
            feasible = False
            if  contype == 'expr':
                #print("type expr")
                polycons = True
                exprcon = m.getConsExprPolyCons(con)
                #print(exprcon)
                #print("exprcon works")
                #if exprcon[Term()] == 0.0:
                #    exprcon += 1
                polynomial = ExprToPoly(exprcon)
                #print(ExprToPoly(exprcon))
                #print(polynomial.A, polynomial.b)
                polynomial.clean()
                #transform into problem with constraints >= 0
                if not m.isInfinity(-m.getLhs(con)):
                    polynomial.b[0] -= m.getLhs(con)
                    print('lhs=',m.getLhs(con))
                #TODO: need to make sure that constraint >= 0, so need to switch signs everywhere and then add rhs (subtract -rhs)
                if not m.isInfinity(m.getRhs(con)):
                    print('rhs=',m.getRhs(con))
                    polynomial.b[0] -= m.getRhs(con)
                    #polynomial.b *= -1
                    print('b=',polynomial.b)
                #print(polynomial)
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
                    #polynomial.b *= -1
                constrained_list.append(polynomial)
                print('linear=',polynomial.A,polynomial.b)
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
                    i += 1
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
                polynomial = Polynomial(A,b)
                polynomial.clean()
                #print(polynomial)
                
                if not m.isInfinity(-m.getLhs(con)):
                    polynomial.b[0] -= m.getLhs(con)
                elif not m.isInfinity(m.getRhs(con)):
                    polynomial.b[0] -= m.getRhs(con)
                    #polynomial.b *= -1
                #print(polynomial)
                constrained_list.append(polynomial)

        obj = ExprToPoly(m.getObjective())
        #print(obj.A,obj.b)
        #print("obj =", obj)
        if not polycons:
            return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -m.infinity()}
        fmin, xmin = constrained_opt(obj, constrained_list)
        print('fmin = ', fmin)
        if np.isnan(xmin).all():
            return {'result': SCIP_RESULT.DIDNOTRUN, 'lowerbound': -m.infinity()}
        return {'result': SCIP_RESULT.SUCCESS, 'lowerbound': fmin}
                
        #TODO: make sure to only use constraints the are consexpr (so only nonlinear constraints that are polynomials) -> cons.getType() == 'expr'
        #2: relax constraints via Sonc relaxations
        #3: give relaxed constraints back into SCIP, let SCIP work with these constraints
    
if __name__=="__main__":
    m = example()
    #m = Model()
    #x0 = m.addVar(vtype = "C", name = "x0")
    #x1 = m.addVar(vtype = "C", name = "x1")
    #m.addCons(0.8*x1**2+0.9*x0**4*x1**2-1.5*x0*x1**2 <= 100.0)
    #m.setObjective(x1 - 2*x0)
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.includeRelax(SoncRelax(),"SONCRelaxator", "Relaxator using SONC decompositions", freq=1)
    """obj = m.getObjective()
    xmin = [0,1]
    print(obj)
    lowerbound = 0
    for term in obj:
        print(re.findall(r'x\(?([0-9]+)\)?', str(term)))
        lowerbound += obj[term]*xmin[int(re.findall(r'x\(?([0-9]+)\)?', str(term))[0])]
    print(lowerbound)"""
    m.optimize()
    var = m.getVars()
    print(m.getVal(var[0]))
    print(m.getVal(var[1]))
    print(m.getObjVal())
    
    n = example()
    n.setPresolve(SCIP_PARAMSETTING.OFF)
    n.disablePropagation()
    """conss = n.getConss()
    for con in conss:
        if con.getType() == 'expr':
                print("type expr")
                exprcon = n.getConsExprPolyCons(con)
                polynomial = Expr_to_Poly(exprcon)
                print(polynomial)
                #polynomial.run_sonc()
                #decomps = polynomial.get_decomposition()
                #for poly in decomps:
                #    print(poly)
                #print(exprcon)"""
    n.optimize()
    va = n.getVars()
    print(n.getVal(va[0]))
    print(n.getVal(va[1]))
    print(n.getObjVal())
    """r = Polynomial([[2,6,2,3], [2,2,6,3]], [1.5,1,2,-3])
    print(r)
    p = Polynomial('1.5*x0**2*x1**6+x0**6*x1**2+2*x0**2*x1**2-3*x0**3*x1**3')
    p.clean()
    #var = m.getVars()
    print(p)
    print(r.A)
    print(r.b)
    #print(Term(var[0], var[0], var[1]))
    #print(Poly_to_Expr(p,m))
    p.sonc_opt_python()
    for el in p.get_decomposition():
        #el.dirty = True
        #el.clean()
        print('el = ', el)
        print(el.A, el.b)
    #q = Polynomial('1.0 + 3*x0**2*x1**6 +2*x0**6*x1**2 + 6*x0**2*x1**2 - x0*x1**2 - 2*x0**2*x1 - 3*x0**3*x1**3')
    A,b = create_poly(2,6,4,np.array([[1,1,1,1],[0,0,6,1],[2,6,0,3]]))
    #print(A,b)
    #q = Polynomial('x0**2+3*x1**2+x0**4*x1**2+2*x0**4*x1**4+x0**2*x1**6+3*x0*x1**2-x0**2*x1**3')
    q = Polynomial('0.8*x1**2+0.9*x0**4*x1**2-1.5*x0*x1**2+1 ')
    q.clean()
    print(q)
    q.run_sonc()
    print(q.get_solution(0))
    q.print_solutions()
    #print(q.get_solution(1))
    #print(q.solution)
    for poly in q.get_decomposition():
        #poly.dirty = False
        #poly.clean()
        print('poly = ',poly)
        print(poly.A, poly.b)"""
    
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
