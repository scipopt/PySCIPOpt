from pyscipopt import Model, Conshdlr, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from sys import version_info

if version_info >= (3, 3):
    from types import SimpleNamespace
else:
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))

        def __eq__(self, other):
            return self.__dict__ == other.__dict__

## callbacks which are not implemented yet:
# PyConsGetdivebdchgs
# PyConsGetvars
# PyConsCopy

## callbacks which are not tested here are:
# consenfops
# consresprop
# conscopy
# consparse
# consgetvars
# consgetdivebdchgs

## callbacks which are not called are:
# conssepasol
# consdelvars
# consprint

ids = []
calls = set([])

class MyConshdlr(Conshdlr):

    def __init__(self, shouldtrans, shouldcopy):
        self.shouldtrans = shouldtrans
        self.shouldcopy = shouldcopy

    def createData(self, constraint, nvars, othername):
        print("Creating data for my constraint: %s"%constraint.name)
        constraint.data = SimpleNamespace()
        constraint.data.nvars = nvars
        constraint.data.myothername = othername

    ## fundamental callbacks ##
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        calls.add("consenfolp")
        for constraint in constraints:
            assert id(constraint) in ids
        return {"result": SCIP_RESULT.FEASIBLE}

    # consenfops

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        calls.add("conscheck")
        for constraint in constraints:
            assert id(constraint) in ids
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        calls.add("conslock")
        assert id(constraint) in ids

    ## callbacks ##
    def consfree(self):
        calls.add("consfree")

    def consinit(self, constraints):
        calls.add("consinit")
        for constraint in constraints:
            assert id(constraint) in ids

    def consexit(self, constraints):
        calls.add("consexit")
        for constraint in constraints:
            assert id(constraint) in ids

    def consinitpre(self, constraints):
        calls.add("consinitpre")
        for constraint in constraints:
            assert id(constraint) in ids

    def consexitpre(self, constraints):
        calls.add("consexitpre")
        for constraint in constraints:
            assert id(constraint) in ids

    def consinitsol(self, constraints):
        calls.add("consinitsol")
        for constraint in constraints:
            assert id(constraint) in ids

    def consexitsol(self, constraints, restart):
        calls.add("consexitsol")
        for constraint in constraints:
            assert id(constraint) in ids

    def consdelete(self, constraint):
        calls.add("consdelete")
        assert id(constraint) in ids

    def constrans(self, sourceconstraint):
        calls.add("constrans")
        assert id(sourceconstraint) in ids
        if self.shouldtrans:
            transcons = self.model.createCons(self, "transformed_" + sourceconstraint.name)
            ids.append(id(transcons))
            return {"targetcons" : transcons}
        return {}

    def consinitlp(self, constraints):
        calls.add("consinitlp")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def conssepalp(self, constraints, nusefulconss):
        calls.add("conssepalp")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def conssepasol(self, constraints, nusefulconss, solution):
        calls.add("conssepasol")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        calls.add("consprop")
        for constraint in constraints:
            assert id(constraint) in ids
        return {}

    def conspresol(self, constraints, nrounds, presoltiming,
                   nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                   nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict):
        calls.add("conspresol")
        return result_dict

    # consresprop

    def consactive(self, constraint):
        calls.add("consactive")
        assert id(constraint) in ids

    def consdeactive(self, constraint):
        calls.add("consdeactive")
        assert id(constraint) in ids

    def consenable(self, constraint):
        calls.add("consenable")
        assert id(constraint) in ids

    def consdisable(self, constraint):
        calls.add("consdisable")
        assert id(constraint) in ids

    def consdelvars(self, constraints):
        calls.add("consdelvars")
        for constraint in constraints:
            assert id(constraint) in ids

    def consprint(self, constraint):
        calls.add("consprint")
        assert id(constraint) in ids

    # conscopy
    # consparse
    # consgetvars

    def consgetnvars(self, constraint):
        calls.add("consgetnvars")
        assert id(constraint) in ids
        return {"nvars": 1, "success": True}

    # consgetdivebdchgs


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
        conshdlr = MyConshdlr(shouldtrans=True, shouldcopy=False)
        s.includeConshdlr(conshdlr, "PyCons", "custom constraint handler implemented in python",
                          sepapriority = 1, enfopriority = 1, chckpriority = 1, sepafreq = 10, propfreq = 50,
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

    # so that consfree gets called
    del s

    # check callbacks got called
    assert "consenfolp" in calls
    assert "conscheck" in calls
    assert "conslock" in calls
    assert "consfree" in calls
    assert "consinit" in calls
    assert "consexit" in calls
    assert "consinitpre" in calls
    assert "consexitpre" in calls
    assert "consinitsol" in calls
    assert "consexitsol" in calls
    assert "consdelete" in calls
    assert "constrans" in calls
    assert "consinitlp" in calls
    assert "conssepalp" in calls
    #assert "conssepasol" in calls
    assert "consprop" in calls
    assert "conspresol" in calls
    assert "consactive" in calls
    assert "consdeactive" in calls
    assert "consenable" in calls
    assert "consdisable" in calls
    #assert "consdelvars" in calls
    #assert "consprint" in calls
    assert "consgetnvars" in calls