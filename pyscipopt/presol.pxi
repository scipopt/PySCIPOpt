cdef SCIP_RETCODE PyPresolCopy (SCIP* scip, SCIP_PRESOL* presol):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPresolFree (SCIP* scip, SCIP_PRESOL* presol):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    PyPresol.presolfree()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPresolInit (SCIP* scip, SCIP_PRESOL* presol):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    PyPresol.presolinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPresolExit (SCIP* scip, SCIP_PRESOL* presol):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    PyPresol.presolexit()
    return SCIP_OKAY


cdef SCIP_RETCODE PyPresolInitpre (SCIP* scip, SCIP_PRESOL* presol):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    PyPresol.presolinitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPresolExitpre (SCIP* scip, SCIP_PRESOL* presol):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    PyPresol.presolexitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPresolExec (SCIP* scip, SCIP_PRESOL* presol, int nrounds, SCIP_PRESOLTIMING presoltiming, int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes, int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides, int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes, int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    returnvalues = PyPresol.presolexec()
    result[0] = returnvalues.get("result", result[0])
    return SCIP_OKAY

cdef class Presol:
    cdef public object data     # storage for the python user
    cdef public Model model

    def presolfree(self):
        pass

    def presolinit(self):
        pass

    def presolexit(self):
        pass

    def presolinitpre(self):
        pass

    def presolexitpre(self):
        pass

    def presolexec(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

