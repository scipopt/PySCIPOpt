from pyscipopt import Model, Conshdlr, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from types import SimpleNamespace

ids = []

class MyConshdlr(Conshdlr):

    def createData(self, constraint, nvars, othername):
        print("Creating data for my constraint: %s"%constraint.name)
        constraint.data = SimpleNamespace()
        constraint.data._nvars = nvars
        constraint.data._myothername = othername

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        print("[consenfolp]")
        for constraint in constraints:
            assert id(constraint) in ids
        return {"result": SCIP_RESULT.FEASIBLE}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason):
        print("[conscheck]")
        for constraint in constraints:
            assert id(constraint) in ids
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, nlockspos, nlocksneg):
        print("[conslock]")
        assert id(constraint) in ids

    def constrans(self, sourceconstraint):
        print("[constrans]")
        assert id(sourceconstraint) in ids
        return {}

    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        print("[consprop]")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def conssepalp(self, constraints, nusefulconss):
        print("[conssepalp]")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def conssepasol(self, constraints, nusefulconss, solution):
        print("[conssepasol]")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def conspresol(self, constraints, nrounds, presoltiming,
                   nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                   nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict):
        print("[conspresol]")
        return result_dict

    def consdelete(self, constraint):
        print("[consdelete]")
        assert id(constraint) in ids

    def consinit(self, constraints):
        print("[consinit]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consexit(self, constraints):
        print("[consexit]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consinitpre(self, constraints):
        print("[consinitpre]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consexitpre(self, constraints):
        print("[consexitpre]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consinitsol(self, constraints):
        print("[consinitsol]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consexitsol(self, constraints, restart):
        print("[consexitsol]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consinitlp(self, constraints):
        print("[consinitlp]")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def consactive(self, constraint):
        print("[consactive]")
        assert id(constraint) in ids

    def consdeactive(self, constraint):
        print("[consdeactive]")
        assert id(constraint) in ids

    def consenable(self, constraint):
        print("[consenable]")
        assert id(constraint) in ids

    def consdisable(self, constraint):
        print("[consdisable]")
        assert id(constraint) in ids

    def consdelvars(self, constraints):
        print("[consdelvars]")
        for constraint in constraints:
            assert id(constraint) in ids

    def consprint(self, constraint):
        print("[consprint]")
        assert id(constraint) in ids

    def consgetnvars(self, constraint):
        print("[consgetnvars]")
        assert id(constraint) in ids
        return {}


def test_conshdlr():
    def create_model():
        # create solver instance
        s = Model()

        # add some variables
        x = s.addVar("x", obj = -1.0, vtype = "I", lb=-10)
        y = s.addVar("y", obj = 1.0, vtype = "I", lb=-1000)
        z = s.addVar("z", obj = 1.0, vtype = "I", lb=-1000)

        # add some constraint
        s.addCons(314*x + 867*y + 860*z == 363)
        s.addCons(87*x + 875*y - 695*z == 423)

        # create conshdlr and include it to SCIP
        conshdlr = MyConshdlr()
        s.includeConshdlr(conshdlr, "PyCons", "custom constraint handler implemented in python",
                          sepapriority = 1, enfopriority = -1, chckpriority = 1, sepafreq = 10, propfreq = 50,
                          eagerfreq = 1, maxprerounds = -1, delaysepa = False, delayprop = False, needscons = True,
                          presoltiming = SCIP_PRESOLTIMING.FAST, proptiming = SCIP_PROPTIMING.BEFORELP)

        cons1 = s.createCons(conshdlr, "cons1name")
        ids.append(id(cons1))
        cons2 = s.createCons(conshdlr, "cons2name")
        ids.append(id(cons2))
        conshdlr.createData(cons1, 10, "cons1_anothername")
        conshdlr.createData(cons2, 12, "cons2_anothername")

        # add these constraints
        s.addPyCons(cons1)
        s.addPyCons(cons2)
        return s

    s = create_model()

    # solve problem
    s.optimize()

if __name__ == "__main__":
    test_conshdlr()
