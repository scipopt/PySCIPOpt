from pyscipopt import Model
from pyscipopt.scip import Relax

calls = []

class SoncRelax(Relax):
    def relaxexec(self):
        calls.append('relaxexec')
        

def test_relax():
    m = Model()
    m.hideOutput()
    #include relaxator
    m.includeRelax(SoncRelax(),'testrelaxator','Test that relaxator gets included')
    
    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")
    x2 = m.addVar(vtype = "C", name = "x2")
    
    #addCons
    m.addCons(x0 >= 2)
    m.addCons(x0**2 <= x1)
    m.addCons(x1 * x2 >= x0)
    
    m.setObjective(x1 + x0)
    m.optimize()
    print(m.getVal(x0))
    assert 'relaxexec' in calls
    assert len(calls) == 1
    
    
if __name__ == "__main__":
    test_relax()
