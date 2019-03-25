from pyscipopt import Model, SCIP_PARAMSETTING
from pyscipopt.scip import Relax

def test():
    m = Model()
    m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    
    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")

    #addCons
    m.addCons(x1-x0>=0)
    m.addCons(x1+x0>=4)
    c2 = m.addCons(x1**3<=8)
    
    m.setObjective(x1 + 2*x0)
    print(c2.getType())
    #print(m.getConsExprExprNPolyTerms(c2))
    
if __name__ == "__main__":
    test()
