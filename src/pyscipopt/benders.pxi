##@file benders.pxi
#@brief Base class of the Benders decomposition Plugin
cdef class Benders:
    cdef public Model model
    cdef public str name
    cdef SCIP_BENDERS* _benders

    def bendersfree(self):
        '''calls destructor and frees memory of Benders decomposition '''
        pass

    def bendersinit(self):
        '''initializes Benders deconposition'''
        pass

    def bendersexit(self):
        '''calls exit method of Benders decomposition'''
        pass

    def bendersinitpre(self):
        '''informs the Benders decomposition that the presolving process is being started '''
        pass

    def bendersexitpre(self):
        '''informs the Benders decomposition that the presolving process has been completed'''
        pass

    def bendersinitsol(self):
        '''informs Benders decomposition that the branch and bound process is being started '''
        pass

    def bendersexitsol(self):
        '''informs Benders decomposition that the branch and bound process data is being freed'''
        pass

    def benderscreatesub(self, probnumber):
        '''creates the subproblems and registers it with the Benders decomposition struct '''
        print("python error in benderscreatesub: this method needs to be implemented")
        return {}

    def benderspresubsolve(self, solution, enfotype, checkint):
        '''sets the pre subproblem solve callback of Benders decomposition '''
        return {}

    def benderssolvesubconvex(self, solution, probnumber, onlyconvex):
        '''sets convex solve callback of Benders decomposition'''
        return {}

    def benderssolvesub(self, solution, probnumber):
        '''sets solve callback of Benders decomposition '''
        return {}

    def benderspostsolve(self, solution, enfotype, mergecandidates, npriomergecands, checkint, infeasible):
        '''sets post-solve callback of Benders decomposition '''
        return {}

    def bendersfreesub(self, probnumber):
        '''frees the subproblems'''
        pass

    def bendersgetvar(self, variable, probnumber):
        '''Returns the corresponding master or subproblem variable for the given variable. This provides a call back for the variable mapping between the master and subproblems. '''
        print("python error in bendersgetvar: this method needs to be implemented")
        return {}

# local helper functions for the interface
cdef Variable getPyVar(SCIP_VAR* var):
    cdef SCIP_VARDATA* vardata
    vardata = SCIPvarGetData(var)
    return <Variable>vardata


cdef SCIP_RETCODE PyBendersCopy (SCIP* scip, SCIP_BENDERS* benders, SCIP_Bool threadsafe) with gil:
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersFree (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersfree()
    Py_DECREF(PyBenders)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersInit (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersExit (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersInitpre (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersinitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersExitpre (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersexitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersInitsol (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersExitsol (SCIP* scip, SCIP_BENDERS* benders) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersCreatesub (SCIP* scip, SCIP_BENDERS* benders, int probnumber) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.benderscreatesub(probnumber)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersPresubsolve (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, SCIP_BENDERSENFOTYPE type, SCIP_Bool checkint, SCIP_Bool* infeasible, SCIP_Bool* auxviol, SCIP_Bool* skipsolve,  SCIP_RESULT* result) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    if sol == NULL:
        solution = None
    else:
        solution = Solution.create(scip, sol)
    enfotype = type
    result_dict = PyBenders.benderspresubsolve(solution, enfotype, checkint)
    infeasible[0] = result_dict.get("infeasible", False)
    auxviol[0] = result_dict.get("auxviol", False)
    skipsolve[0] = result_dict.get("skipsolve", False)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersSolvesubconvex (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber, SCIP_Bool onlyconvex, SCIP_Real* objective, SCIP_RESULT* result) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    if sol == NULL:
        solution = None
    else:
        solution = Solution.create(scip, sol)
    result_dict = PyBenders.benderssolvesubconvex(solution, probnumber, onlyconvex)
    objective[0] = result_dict.get("objective", 1e+20)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersSolvesub (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber, SCIP_Real* objective, SCIP_RESULT* result) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    if sol == NULL:
        solution = None
    else:
        solution = Solution.create(scip, sol)
    result_dict = PyBenders.benderssolvesub(solution, probnumber)
    objective[0] = result_dict.get("objective", 1e+20)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersPostsolve (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol,
        SCIP_BENDERSENFOTYPE type, int* mergecands, int npriomergecands, int nmergecands, SCIP_Bool checkint,
        SCIP_Bool infeasible, SCIP_Bool* merged) with  gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    if sol == NULL:
        solution = None
    else:
        solution = Solution.create(scip, sol)
    enfotype = type
    mergecandidates = []
    for i in range(nmergecands):
        mergecandidates.append(mergecands[i])
    result_dict = PyBenders.benderspostsolve(solution, enfotype, mergecandidates, npriomergecands, checkint, infeasible)
    merged[0] = result_dict.get("merged", False)
    return SCIP_OKAY

cdef SCIP_RETCODE PyBendersFreesub (SCIP* scip, SCIP_BENDERS* benders, int probnumber) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyBenders.bendersfreesub(probnumber)
    return SCIP_OKAY

#TODO: Really need to ask about the passing and returning of variables
cdef SCIP_RETCODE PyBendersGetvar (SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR* var, SCIP_VAR** mappedvar, int probnumber) with gil:
    cdef SCIP_BENDERSDATA* bendersdata
    bendersdata = SCIPbendersGetData(benders)
    PyBenders = <Benders>bendersdata
    PyVar = getPyVar(var)
    result_dict = PyBenders.bendersgetvar(PyVar, probnumber)
    mappedvariable = <Variable>(result_dict.get("mappedvar", None))
    if mappedvariable is None:
        mappedvar[0] = NULL
    else:
        mappedvar[0] = mappedvariable.scip_var
    return SCIP_OKAY
