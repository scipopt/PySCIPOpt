cimport pyscipopt.scip as scip

class py_sepadata(object):
    pass

cdef class Separator:
    cdef scip.SCIP_SEPA* _sepa

    cdef scip.SCIP_SEPADATA* _sepadata

    def getSepaData(self):
        return py_sepadata

def py_scip_execlp(Model model, Separator sepa):
    pass

def py_scip_execsol(Model model, Separator sepa):
    pass

cdef SCIP_RETCODE scipSepaExecLP(SCIP* _scip,
                                 SCIP_SEPA* _sepa,
                                 SCIP_RESULT* _result):
    s = Model()
    s._scip = _scip
    sepa = Separator()
    sepa._sepa = _sepa
    _result[0] = py_scip_execlp(s, sepa)
    del s
    del sepa
    return scip.SCIP_OKAY

cdef SCIP_RETCODE scipSepaExecSol(SCIP* _scip,
                                  SCIP_SEPA* _sepa,
                                  SCIP_SOL* _sol,
                                  SCIP_RESULT* _result):
    s = Model()
    s._scip = _scip
    sepa = Separator()
    sepa._sepa = _sepa
    _result[0] = py_scip_execsol(s, sepa)
    del s
    del sepa
    return scip.SCIP_OKAY
