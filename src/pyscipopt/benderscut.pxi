##@file benderscut.pxi
#@brief Base class of the Benderscut Plugin
cdef class Benderscut:
    cdef public Model model
    cdef public Benders benders
    cdef public str name

    def benderscutfree(self):
        pass

    def benderscutinit(self):
        pass

    def benderscutexit(self):
        pass

    def benderscutinitsol(self):
        pass

    def benderscutexitsol(self):
        pass

    def benderscutexec(self, solution, probnumber, enfotype):
        print("python error in benderscutexec: this method needs to be implemented")
        return {}

cdef SCIP_RETCODE PyBenderscutCopy (SCIP* scip, SCIP_BENDERS* benders, SCIP_BENDERSCUT* benderscut) with gil:
    return SCIP_OKAY

cdef SCIP_RETCODE PyBenderscutFree (SCIP* scip, SCIP_BENDERSCUT* benderscut) with gil:
    cdef SCIP_BENDERSCUTDATA* benderscutdata
    benderscutdata = SCIPbenderscutGetData(benderscut)
    PyBenderscut = <Benderscut>benderscutdata
    PyBenderscut.benderscutfree()
    Py_DECREF(PyBenderscut)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBenderscutInit (SCIP* scip, SCIP_BENDERSCUT* benderscut) with gil:
    cdef SCIP_BENDERSCUTDATA* benderscutdata
    benderscutdata = SCIPbenderscutGetData(benderscut)
    PyBenderscut = <Benderscut>benderscutdata
    PyBenderscut.benderscutinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBenderscutExit (SCIP* scip, SCIP_BENDERSCUT* benderscut) with gil:
    cdef SCIP_BENDERSCUTDATA* benderscutdata
    benderscutdata = SCIPbenderscutGetData(benderscut)
    PyBenderscut = <Benderscut>benderscutdata
    PyBenderscut.benderscutexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBenderscutInitsol (SCIP* scip, SCIP_BENDERSCUT* benderscut) with gil:
    cdef SCIP_BENDERSCUTDATA* benderscutdata
    benderscutdata = SCIPbenderscutGetData(benderscut)
    PyBenderscut = <Benderscut>benderscutdata
    PyBenderscut.benderscutinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBenderscutExitsol (SCIP* scip, SCIP_BENDERSCUT* benderscut) with gil:
    cdef SCIP_BENDERSCUTDATA* benderscutdata
    benderscutdata = SCIPbenderscutGetData(benderscut)
    PyBenderscut = <Benderscut>benderscutdata
    PyBenderscut.benderscutexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBenderscutExec (SCIP* scip, SCIP_BENDERS* benders, SCIP_BENDERSCUT* benderscut, SCIP_SOL* sol, int probnumber, SCIP_BENDERSENFOTYPE type, SCIP_RESULT* result) with gil:
    cdef SCIP_BENDERSCUTDATA* benderscutdata
    benderscutdata = SCIPbenderscutGetData(benderscut)
    PyBenderscut = <Benderscut>benderscutdata
    if sol == NULL:
        solution = None
    else:
        solution = Solution.create(scip, sol)
    enfotype = type
    result_dict = PyBenderscut.benderscutexec(solution, probnumber, enfotype)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY
