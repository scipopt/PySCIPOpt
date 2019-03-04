##@file presol.pxi
#@brief Base class of the Presolver Plugin
cdef class Presol:
    cdef public Model model

    def presolfree(self):
        '''frees memory of presolver'''
        pass

    def presolinit(self):
        '''initializes presolver'''
        pass

    def presolexit(self):
        '''deinitializes presolver'''
        pass

    def presolinitpre(self):
        '''informs presolver that the presolving process is being started'''
        pass

    def presolexitpre(self):
        '''informs presolver that the presolving process is finished'''
        pass

    def presolexec(self, nrounds, presoltiming):
        '''executes presolver'''
        print("python error in presolexec: this method needs to be implemented")
        return {}



cdef SCIP_RETCODE PyPresolCopy (SCIP* scip, SCIP_PRESOL* presol):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPresolFree (SCIP* scip, SCIP_PRESOL* presol):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    PyPresol.presolfree()
    Py_DECREF(PyPresol)
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

cdef SCIP_RETCODE PyPresolExec (SCIP* scip, SCIP_PRESOL* presol, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_PRESOLDATA* presoldata
    presoldata = SCIPpresolGetData(presol)
    PyPresol = <Presol>presoldata
    result_dict = PyPresol.presolexec(nrounds, presoltiming)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    nfixedvars[0] += result_dict.get("nnewfixedvars", 0)
    naggrvars[0] += result_dict.get("nnewaggrvars", 0)
    nchgvartypes[0] += result_dict.get("nnewchgvartypes", 0)
    nchgbds[0] += result_dict.get("nnewchgbds", 0)
    naddholes[0] += result_dict.get("nnewaddholes", 0)
    ndelconss[0] += result_dict.get("nnewdelconss", 0)
    naddconss[0] += result_dict.get("nnewaddconss", 0)
    nupgdconss[0] += result_dict.get("nnewupgdconss", 0)
    nchgcoefs[0] += result_dict.get("nnewchgcoefs", 0)
    nchgsides[0] += result_dict.get("nnewchgsides", 0)
    return SCIP_OKAY
