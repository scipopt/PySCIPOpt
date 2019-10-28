#! usr/bin/env python3
from pyscipopt       import Term, Expr
from POEM.python.polynomial import *

import numpy as np
import re


def ExprToPoly(Exp, nvar):
    """turn pyscipopt.scip.Expr into a Polynomial (POEM)
    :param: Exp: expression of type pyscipopt.scip.Expr
    :param: nvar: number of variables of given problem
    """
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

def PolyToExpr(p,var):
    """turn Polynomial (POEM) into a pyscipopt.scip.Expr (uses __dict__ of class polynomial, but this gives the wrong dict for const Term)
    :param: p: Polynomial object
    :param: var: list of variables
    """
    dictP = p.__dict__()
    d = dict()
    #print(dictP)
    for key in dictP.keys():
        i = 0
        t =[]
        #var = model.getVars()
        varstr = [None]*len(var)
        for j in range(len(var)):
            varstr[j] = str(var[j])
        for el in key[1:]:
            f = 'x' + str(i)
            for _ in range(el):
                t.append(var[varstr.index(f)])
            i+=1
        #print(t)
        term = Term()
        for el in t:
            term += Term(el)
        #print(term)
        d[term] = dictP[key]
        #print(t)
    return Expr(d)
