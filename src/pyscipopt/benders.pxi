cdef class Benders:
    cdef public Model model

    def bendersfree(self):
        pass

    def bendersinit(self):
        pass

    def bendersexit(self):
        pass

    def bendersinitpre(self):
        pass

    def bendersexitpre(self):
        pass

    def bendersinitsol(self):
        pass

    def bendersexitsol(self):
        pass

    def benderscreatesub(self, probnumber):
        print("python error in benderscreatesub: this method needs to be implemented")
        return {}

    def benderspresubsolve(self):
        pass

    def benderssolvesub(self, solution, probnumber):
        pass

    def benderspostsolve(self, solution, infeasible):
        pass

    def bendersfreesub(self, probnumber):
        pass

    def bendersgetvar(self, variable, probnumber):
        print("python error in bendersgetvar: this method needs to be implemented")
        return {}


cdef SCIP_RETCODE PyBendersCopy (SCIP* scip, SCIP_BENDERS* benders):
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersFree (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersfree()
    Py_DECREF(PyBenders)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersInit (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersExit (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersInitpre (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersinitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersExitpre (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersexitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersInitsol (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersExitsol (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersCreatesub (SCIP* scip, SCIP_BENDERS* benders, int probnumber):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.benderscreatesub(probnumber)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersPresubsolve (SCIP* scip, SCIP_BENDERS* benders):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.benderspresubsolve()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersSolvesub (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber, SCIP_Bool* infeasible):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    solution = Solution()
    solution.sol = sol
    result_dict = PyBenders.benderssolvesub(solution, probnumber)
    infeasible[0] = result_dict.get("infeasible", <SCIP_Real>infeasible[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersPostsolve (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, SCIP_Bool infeasible):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    solution = Solution()
    solution.sol = sol
    PyBenders.benderspostsolve(solution, infeasible)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersFreesub (SCIP* scip, SCIP_BENDERS* benders, int probnumber):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersfreesub(probnumber)
    return SCIP_OKAY

#TODO: Really need to ask about the passing and returning of variables
cdef SCIP_RETCODE PyBendersGetvar (SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR* var, SCIP_VAR** mappedvar, int probnumber):
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    variable = Variable()
    variable.var = var
    result_dict = PyBenders.bendersgetvar(variable, probnumber)
    mappedvar[0] = result_dict.get("mappedvar", <SCIP_VAR>mappedvar[0])
    return SCIP_OKAY
