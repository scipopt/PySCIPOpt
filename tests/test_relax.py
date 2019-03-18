from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING
from pyscipopt.scip import Relax

calls = []
class MyRelax(Relax):
    def relaxexec(self):
        #returns a fixed lowerbound value (higher than the best solution) 
        calls.append('relaxexec')
        lowerbound = 6.0
        return {'result': SCIP_RESULT.SUCCESS, 'lowerbound':lowerbound}
        
class BadHeur(Heur):
    #returns some bad (fixed) bound for the given problem
    def heurexec(self, heurtiming, nodeinfeasible):
        calls.append('heurexec')
        sol = self.model.createSol(self)
        vars = self.model.getVars()

        self.model.setSolVal(sol, vars[0], 2.0)
        self.model.setSolVal(sol, vars[1], 2.0)
        accepted = self.model.trySol(sol)
        sols = self.model.getSols()
        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

def myModel():
    m = Model()
    m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    
    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")

    #addCons
    m.addCons(x1-x0>=0)
    m.addCons(x1+x0>=4)
    m.addCons(2*x1<=7)
    
    m.setObjective(x1 + 2*x0)
    return m

def test_relax():
    #Get the real best solution, not using the heuristic or the relaxator, as a comparison
    m = myModel()
    vars = m.getVars()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.optimize()
    print("Best Solution without Relaxator: ")
    print(" x0 = ", m.getVal(vars[0]), "\n", "x1 = ", m.getVal(vars[1]), "\n", "Bound = ", m.getObjVal(), '\n')
    assert 'relaxexec' not in calls
    assert 'heurexec' not in calls
    
    #get best solution, if relaxator and heuristic are included
    n = myModel()
    n.setIntParam('lp/solvefreq', -1)
    #include relaxator
    relaxator = MyRelax()
    n.includeRelax(relaxator,'badrelaxator', 'test that relaxator gets included by taking a bad relaxator', 1)
    #include heuristics
    heuristic = BadHeur()
    n.includeHeur(heuristic, "BadHeur", "Heuristic returning fixed values", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
    n.optimize()
    print("Solution using a bad heuristic and a bad relaxator:")
    print(" x0 = ", n.getVal(vars[0]), '\n', "x1 = ", n.getVal(vars[1]), '\n', "Bound = ", n.getObjVal())
    assert 'relaxexec' in calls
    assert 'heurexec' in calls
    
if __name__ == "__main__":
    test_relax()
