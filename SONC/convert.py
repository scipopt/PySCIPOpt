#! usr/bin/env python3
from pyscipopt  import Term, Expr
from POEM       import Polynomial

import numpy as np
import re


def ExprToPoly(Exp, nvar, Vars):
    """turn pyscipopt.scip.Expr into a Polynomial (SCIP -> POEM)
     - param Exp: expression of type pyscipopt.scip.Expr
     - param nvar: number of variables of given problem
     - param Vars: list of variables of given problem
    """
    nterms = len([key for key in Exp])

    #get constant Term
    if Term() in Exp:
        const = Exp[Term()]
    else:
        const = 0.0
        nterms += 1
    
    #turn Expr into Polynomial 
    A = np.array([np.zeros(nterms)]*nvar)
    b = np.zeros(nterms)
    b[0] = const
    j = 1
    for key in Exp:
        if str(key) == 'Term(objconstant)':
            b[0] = Exp[key]
        elif not key == Term():
            for el in tuple(key):
                for i,var in enumerate(Vars):
                    #if re.search(str(var), str(el)):
                    if str(var) == str(el) or 't_'+str(var) == str(el):
                        A[i][j] +=1
                        break
            b[j] = Exp[key]
            j += 1
    return Polynomial(A,b)

def PolyToExpr(p,var):
    """turn Polynomial into a pyscipopt.scip.Expr (POEM -> SCIP)
    :param: p: Polynomial object
    :param: var: list of variables
    """
    #TODO: currently working with strings, change that to arrays
    dictP = p.__dict__()
    d = dict()
    for key in dictP.keys():
        i = 0
        t =[]
        varstr = [None]*len(var)
        for j in range(len(var)):
            varstr[j] = str(var[j])
        for el in key[1:]:
            f = 'x' + str(i)
            for _ in range(el):
                t.append(var[varstr.index(f)])
            i+=1
        term = Term()
        for el in t:
            term += Term(el)
        d[term] = dictP[key]
    return Expr(d)
