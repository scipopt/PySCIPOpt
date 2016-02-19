cdef SCIP_RETCODE PyBranchruleCopy (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleFree (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.free()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleInit (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.init()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExit (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.exit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleInitsol (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.initsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExitsol (SCIP* scip, SCIP_BRANCHRULE* branchrule):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    PyBranchrule.exitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExeclp (SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    result[0] = PyBranchrule.execlp()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExecext(SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    result[0] = PyBranchrule.execext()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBranchruleExecps(SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result):
    cdef SCIP_BRANCHRULEDATA* branchruledata
    branchruledata = SCIPbranchruleGetData(branchrule)
    PyBranchrule = <Branchrule>branchruledata
    result[0] = PyBranchrule.execps()
    return SCIP_OKAY

cdef class Branchrule:
    cdef public object data     # storage for the python user
    cdef public Model model

    def free(self):
        pass

    def init(self):
        pass

    def exit(self):
        pass

    def initsol(self):
        pass

    def exitsol(self):
        pass

    def execlp(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def execext(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def execps(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

