cdef class Heur:
    cdef public Model model
    cdef public str name

    def heurfree(self):
        pass

    def heurinit(self):
        pass

    def heurexit(self):
        pass

    def heurinitsol(self):
        pass

    def heurexitsol(self):
        pass

    def heurexec(self, heurtiming, nodeinfeasible):
        print("python error in heurexec: this method needs to be implemented")
        return {}



cdef SCIP_RETCODE PyHeurCopy (SCIP* scip, SCIP_HEUR* heur):
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurFree (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.heurfree()
    Py_DECREF(PyHeur)
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurInit (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.heurinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurExit (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.heurexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurInitsol (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.heurinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurExitsol (SCIP* scip, SCIP_HEUR* heur):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    PyHeur.heurexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyHeurExec (SCIP* scip, SCIP_HEUR* heur, SCIP_HEURTIMING heurtiming, SCIP_Bool nodeinfeasible, SCIP_RESULT* result):
    cdef SCIP_HEURDATA* heurdata
    heurdata = SCIPheurGetData(heur)
    PyHeur = <Heur>heurdata
    result_dict = PyHeur.heurexec(heurtiming, nodeinfeasible)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY
