cdef SCIP_RETCODE PyHeurCopy (SCIP* scip, SCIP_HEUR* heur):
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurFree (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.free()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurInit (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.init()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurExit (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.exit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurInitsol (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.initsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurExitsol (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.exitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurExec (SCIP* scip, SCIP_HEUR* heur, SCIP_HEURTIMING heurtiming, SCIP_Bool nodeinfeasible, SCIP_RESULT* result):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    returnvalues = PyHeur.heurexec()
    result_dict = returnvalues
    result[0] = result_dict.get("result", SCIP_DIDNOTFIND)
    return SCIP_OKAY

cdef class Heur:
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

    def heurexec(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}
