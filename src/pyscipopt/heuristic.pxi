##@file heuristic.pxi
#@brief Base class of the Heuristics Plugin
cdef class Heur:
    cdef public Model model
    cdef public str name

    def heurfree(self):
        '''calls destructor and frees memory of primal heuristic'''
        pass

    def heurinit(self):
        '''initializes primal heuristic'''
        pass

    def heurexit(self):
        '''calls exit method of primal heuristic'''
        pass

    def heurinitsol(self):
        '''informs primal heuristic that the branch and bound process is being started'''
        pass

    def heurexitsol(self):
        '''informs primal heuristic that the branch and bound process data is being freed'''
        pass

    def heurexec(self, heurtiming, nodeinfeasible):
        '''should the heuristic the executed at the given depth, frequency, timing,...'''
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
