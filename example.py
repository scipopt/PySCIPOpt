from pyscipopt import Model, quicksum, Conshdlr, SCIP_RESULT
from pyscipopt.scip import Relax, Expr, Term, ExprCons, expr_to_nodes
import numpy as np
from polynomial import *
from minimum import *
from types import SimpleNamespace
 

#TODO: which of the functions are really necessary when working with consexpr?? maybe possible to get constraints as strings or something, s.t. it can be forwarded into class Polynomial directly? The other way would be to get a list of the terms of the constraint (polynomial)
#instead of the def "example" maybe add a function that tests all at once (as in test_relax.py)
def example():
#create model example with polynomial constraint
    m = Model()
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")
    x2 = m.addVar(vtype = "C", name = "x2")
    #p = 5.2921570379 + 1.2004716251*x1**1 + 2.93621395232*x0**2 + 2.2408931992*x1**2 + 1.86755799015*x0**2*x1**2 - 0.977277879876*x0**1*x1**2
    p = 7.93823555685 + 1.80070743765*x0**6 + 4.40432092848*x1**6 + 0.950088417526*x0**2*x1**2 + 2.2408931992*x0**1*x1**2 + 1.86755799015*x0**1*x1**4 - 0.977277879876*x0**2*x1**1 - 0.151357208298*x0**3*x1**1 - 0.103218851794*x0**4*x1**1+3*x0+x1
    c = m.addCons(p<=x2)
    d = m.addCons(x0<=5)
    e = m.addCons(x1<=5)
    m.setObjective(x2)
    #c.data = p,x2
    m.data = x0, x1,x2, p, c, d, e
    return m

def tuple_to_list(t):
#convert a tuple of form (x0,..,x0,...xn,...xn) into list with entry[i] = number of times xi can be found in tuple, input: tuple t
    if len(t) == 0:
        return []
    nmax = max([int(i) for i in re.findall(r'x\(?([0-9]+)\)?', str(t))]) + 1
    power = [0 for _ in range(nmax)]
    for el in t:
        i = re.findall(r'x\(?([0-9]+)\)?', str(el))
        power[int(i[0])] += 1
    return power
    
def Poly_to_Expr(p):
#rewrite polynomial p in as dict in the way it is used in Expr (uses __dict__ of class polynomial, dict of form (1, exponent):coefficient), input: Polynomials object p
    dictP = p.__dict__()
    d = dict()
    for key in dictP.keys():
        i = 0
        t =[]
        for el in key[1:]:
            f = 'x'+str(i)
            for _ in range(el):
               t.append(f)
            i+=1
        d[Term(tuple(eval(el) for el in t))] = dictP[key]
    return Expr(d)
        
def Expr_to_Poly(Exprs):
#convert Expr object into Polynomial object to be able to use Polynomial class, input: Expr object Exprs
    P = ''
    for key in Exprs:
        if key == Term():
            const = Exprs[key]
            P += str(const)
        else:
            power = tuple_to_list(tuple(key))
            term = str(Exprs[key])+'*'+'*'.join('x'+str(i)+'**'+str(power[i]) for i in range(len(power)) if power[i] != 0) 
            P += '+' + term
    P = P.replace('+-','-')
    return Polynomial(P)

def add_Poly_Cons(model, p, rhs = 0.0):
#include a new Constraint into model model, use Polynomial object as Constraint, input: SCIP model model, Polynomial object p, optional right hand side rhs 
#TODO: right now Constraint gets included in original model, change to "subproblem where relaxation is used"
    e = Poly_to_Expr(p)
    model.addCons(e <= rhs)
    return model

class SoncRelax(Relax):
    def relaxexec(self):
        Cons = m.getConss()
        subprob = Model()
        varlist = m.getVars()
        for var in varlist:
            subprob.addVar(vtype = var.vtype(),name = str(var))
        for con in Cons:
            print(con.getType())
            if con.getType() == 'nonlinear': #instead if con.getType == Polynomial, since don't care about the rest?
                liste = m.getExprCons(con)
                #print(liste)
                #liste = [Polynomial(liste[i][j]) for i in range(len(liste)) for j in range(len(liste[i]))]
                rhs = m.getRhs(con)
                lhs = m.getLhs(con)
                for i in range(len(liste)):
                    for p in liste[i]:
                        print(p)
                        if type(p) == type(''):
                            p = Polynomial(p)
                            p.sonc_opt_python()
                            rel = p.relax()
                            newCon = Poly_to_Expr(rel)
                            newCon = ExprCons(newCon,lhs,rhs)
                        else: newCon = p
                        subprob.addCons(newCon)
                        print(newCon)
            else: 
                subprob.addPyCons(con)
        print(subprob.getConss())
        subprob.optimize()
        sol = subprob.getBestSol()
        var = subprob.getVars()
        for i in range(len(var)):
            val = subprob.getSolVal(sol,var[i])
            subprob.setRelaxSolVal(var[i], val) 
                #implement here the relaxation, TODO: find way to explore the constraints data in python
            #TODO:transform con into form s.t. it can be read from class Polynomial
        return {"result": SCIP_RESULT.SUCCESS, 'lowerbound': m.getSolObjVal(sol)}
    
    #1. Constraint into form to put into Polynomial class
    #2. Build relaxation 


if __name__ == "__main__":
#TODO: 1.get the data of the nonlinear constraint( already have lhs, rhs) alternatively with consdata?, 2. implement relaxator: get constraints data; put data into Polynomial class; relax; rebuild into constraints, 3. checkout how to work on subproblems
    limit = 100.0
    #p = '5.2921570379 + 1.2004716251*x1**1 + 2.93621395232*x0**2 + 2.2408931992*x1**2 + 1.86755799015*x0**2*x1**2 - 0.977277879876*x0**1*x1**2'
    p = '7.93823555685 + 1.80070743765*x0^6 + 4.40432092848*x1^6 + 0.950088417526*x0^2*x1^2 + 2.2408931992*x0^1*x1^2 + 1.86755799015*x0^1*x1^4 - 0.977277879876*x0^2*x1^1 - 0.151357208298*x0^3*x1^1 - 0.103218851794*x0^4*x1^1+3*x0+x1'
    P = Polynomial(p)
    P.sonc_opt_python()
    Rel = P.relax()
    """
    start = start_point(P)
    #print(minimum_of_circuits(P,start))
    #symb = symbolic_general(P)
    P._normalise()
    P.run_all()
    result = get_results(P)
    minima = minimum_of_circuits(P,start)
    update = minimum(P,bary(minima))
    P.print_solutions()
    m = example()
    m.setRealParam("limits/time", limit)
    m.optimize()
    print(P(*update))
    
    print(P(*minimum(P,start)))
    print(m.getObjVal())
    """
    m = example()
    cons = m.getConss()
    x0, x1, x2, p, c, d, e = m.data
    #c = m.getConss()[0]
    #print(m.getRhs(c), m.getLhs(c))
    liste = m.getExprCons(cons[0])
    #print(liste)
    #for i in range(len(liste)):
     #   for j in range(len(liste[i])):
      #      print(type(liste[i][j]))
       #     print(type(liste[i][j])==type(''))
    #print(p)
    #print(Expr_to_Poly(p))
    #print(m.getTermsQuadratic(d))
    """
    con = cons[0]
    print(con[Term(x0)])
		  
    print(p)
    n = max([int(i) for i in re.findall(r'x\(?([0-9]+)\)?', str(Rel))]) + 1
    dictt = {tuple('x'.join(str(k)) *j) :Rel.b[i] for i in range(Rel.A.shape[1]) for k in range(n) for j in Rel.A[:,i]}
    #print(p.expr)
    #print(m.getParam(cons[0]))"""
    #m.printCons(cons[0])
    m.includeRelax(SoncRelax(),"test", "test of relaxation inclusion")
    #print(m.getConss())
    m.optimize()
    #print(c.data)
    q = Expr_to_Poly(p)
    #print(q)
    r = Poly_to_Expr(P)
    #print(r)
    #n = add_Poly_Cons(m,q)
    #print(n.getConss())
    #q.sonc_opt_python()
    
    #maybe able to modify addCons s.t. the consdata hold the python expr dict, then can use scipconsgetdata to get the data back?
