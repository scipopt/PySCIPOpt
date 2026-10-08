from pyscipopt import Model, Symhdlr, SCIP_RESULT


class RecordingSymhdlr(Symhdlr):
    def __init__(self):
        self.tryadd_calls = []
        self.exit_data = None
        self.nprop = 0

    def symtryadd(self, symtype, perms, permvars, permvardomcenter, id, allowbdchgs):
        self.tryadd_calls.append((len(perms), len(permvars)))
        assert all(len(p) == len(permvars) for p in perms)
        return {"success": True, "data": {"id": id, "nperms": len(perms)}}

    def symexit(self, symcomps):
        self.exit_data = [(c.name, c.data) for c in symcomps]

    def symprop(self, symcomps, proptiming):
        self.nprop += 1
        return {"result": SCIP_RESULT.DIDNOTFIND}


def test_symhdlr():
    m = Model()
    m.hideOutput()
    n, k = 6, 4
    x = {(i, j): m.addVar(vtype="B", obj=1 + (j == 0)) for i in range(n) for j in range(k)}
    for i in range(n):
        m.addCons(sum(x[i, j] for j in range(k)) == 1)
    for j in range(k):
        m.addCons(sum((i % 3 + 2) * x[i, j] for i in range(n)) <= 7)

    symhdlr = RecordingSymhdlr()
    m.includeSymhdlr(symhdlr, "recording", "records symmetry components", priority=10**6, propfreq=1)
    m.optimize()

    assert m.getStatus() == "optimal"
    assert len(symhdlr.tryadd_calls) == 1
    nperms, npermvars = symhdlr.tryadd_calls[0]
    assert nperms > 0 and npermvars == n * k
    assert symhdlr.nprop > 0

    m.freeTransform()
    assert symhdlr.exit_data == [("symcomp0", {"id": 0, "nperms": nperms})]
