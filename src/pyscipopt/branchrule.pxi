cdef class Branchrule:
    cdef public Model model

    def branchfree(self):
        pass

    def branchinit(self):
        pass

    def branchexit(self):
        pass

    def branchinitsol(self):
        pass

    def branchexitsol(self):
        pass

    def branchexeclp(self, allowaddcons):
        # this method needs to be implemented by the user
        return {}

    def branchexecext(self, allowaddcons):
        # this method needs to be implemented by the user
        return {}

    def branchexecps(self, allowaddcons):
        # this method needs to be implemented by the user
        return {}



cdef SCIP_RETCODE PyBranchruleCopy (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleFree (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.branchfree()
    Py_DECREF(PyBranchrule)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleInit (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.branchinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExit (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.branchexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleInitsol (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.branchinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExitsol (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.branchexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExeclp (SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    result_dict = PyBranchrule.branchexeclp(allowaddcons)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExecext(SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    result_dict = PyBranchrule.branchexecext(allowaddcons)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExecps(SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    result_dict = PyBranchrule.branchexecps(allowaddcons)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY
