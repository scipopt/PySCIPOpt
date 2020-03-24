##@file conshdlr.pxi
#@brief base class of the Constraint Handler plugin

cdef class Conshdlr:
    cdef public Model model
    cdef public str name

    def consfree(self):
        '''calls destructor and frees memory of constraint handler '''
        pass

    def consinit(self, constraints):
        '''calls initialization method of constraint handler '''
        pass

    def consexit(self, constraints):
        '''calls exit method of constraint handler '''
        pass

    def consinitpre(self, constraints):
        '''informs constraint handler that the presolving process is being started '''
        pass

    def consexitpre(self, constraints):
        '''informs constraint handler that the presolving is finished '''
        pass

    def consinitsol(self, constraints):
        '''informs constraint handler that the branch and bound process is being started '''
        pass

    def consexitsol(self, constraints, restart):
        '''informs constraint handler that the branch and bound process data is being freed '''
        pass

    def consdelete(self, constraint):
        '''sets method of constraint handler to free specific constraint data '''
        pass

    def constrans(self, sourceconstraint):
        '''sets method of constraint handler to transform constraint data into data belonging to the transformed problem '''
        return {}

    def consinitlp(self, constraints):
        '''calls LP initialization method of constraint handler to separate all initial active constraints '''
        return {}

    def conssepalp(self, constraints, nusefulconss):
        '''calls separator method of constraint handler to separate LP solution '''
        return {}

    def conssepasol(self, constraints, nusefulconss, solution):
        '''calls separator method of constraint handler to separate given primal solution '''
        return {}

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        '''calls enforcing method of constraint handler for LP solution for all constraints added'''
        print("python error in consenfolp: this method needs to be implemented")
        return {}

    def consenforelax(self, solution, constraints, nusefulconss, solinfeasible):
        '''calls enforcing method of constraint handler for a relaxation solution for all constraints added'''
        print("python error in consenforelax: this method needs to be implemented")
        return {}

    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        '''calls enforcing method of constraint handler for pseudo solution for all constraints added'''
        print("python error in consenfops: this method needs to be implemented")
        return {}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        '''calls feasibility check method of constraint handler '''
        print("python error in conscheck: this method needs to be implemented")
        return {}

    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        '''calls propagation method of constraint handler '''
        return {}

    def conspresol(self, constraints, nrounds, presoltiming,
                   nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                   nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict):
        '''calls presolving method of constraint handler '''
        return result_dict

    def consresprop(self):
        '''sets propagation conflict resolving method of constraint handler '''
        return {}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        '''variable rounding lock method of constraint handler'''
        print("python error in conslock: this method needs to be implemented")
        return {}

    def consactive(self, constraint):
        '''sets activation notification method of constraint handler '''
        pass

    def consdeactive(self, constraint):
        '''sets deactivation notification method of constraint handler '''
        pass

    def consenable(self, constraint):
        '''sets enabling notification method of constraint handler '''
        pass

    def consdisable(self, constraint):
        '''sets disabling notification method of constraint handler '''
        pass

    def consdelvars(self, constraints):
        '''calls variable deletion method of constraint handler'''
        pass

    def consprint(self, constraint):
        '''sets constraint display method of constraint handler '''
        pass

    def conscopy(self):
        '''sets copy method of both the constraint handler and each associated constraint'''
        pass

    def consparse(self):
        '''sets constraint parsing method of constraint handler '''
        pass

    def consgetvars(self, constraint):
        '''sets constraint variable getter method of constraint handler'''
        pass

    def consgetnvars(self, constraint):
        '''sets constraint variable number getter method of constraint handler '''
        return {}

    def consgetdivebdchgs(self):
        '''calls diving solution enforcement callback of constraint handler, if it exists '''
        pass


# local helper functions for the interface
cdef Conshdlr getPyConshdlr(SCIP_CONSHDLR* conshdlr):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    return <Conshdlr>conshdlrdata

cdef Constraint getPyCons(SCIP_CONS* cons):
    cdef SCIP_CONSDATA* consdata
    consdata = SCIPconsGetData(cons)
    return <Constraint>consdata



cdef SCIP_RETCODE PyConshdlrCopy (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_Bool* valid):
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsFree (SCIP* scip, SCIP_CONSHDLR* conshdlr):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyConshdlr.consfree()
    Py_DECREF(PyConshdlr)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInit (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consinit(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExit (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consexit(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitpre (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consinitpre(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExitpre (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consexitpre(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitsol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consinitsol(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExitsol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool restart):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consexitsol(constraints, restart)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDelete (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_CONSDATA** consdata):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    assert <Constraint>consdata[0] == PyCons
    PyConshdlr.consdelete(PyCons)
    consdata[0] = NULL
    Py_DECREF(PyCons)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsTrans (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* sourcecons, SCIP_CONS** targetcons):
    cdef Constraint PyTargetCons
    PyConshdlr = getPyConshdlr(conshdlr)
    PySourceCons = getPyCons(sourcecons)
    result_dict = PyConshdlr.constrans(PySourceCons)

    # create target (transform) constraint: if user doesn't return a constraint, copy PySourceCons
    # otherwise use the created cons
    if "targetcons" in result_dict:
        PyTargetCons = result_dict.get("targetcons")
        targetcons[0] = PyTargetCons.scip_cons
        Py_INCREF(PyTargetCons)
    else:
        PY_SCIP_CALL(SCIPcreateCons(scip, targetcons, str_conversion(PySourceCons.name), conshdlr, <SCIP_CONSDATA*>PySourceCons,
            PySourceCons.isInitial(), PySourceCons.isSeparated(), PySourceCons.isEnforced(), PySourceCons.isChecked(),
            PySourceCons.isPropagated(), PySourceCons.isLocal(), PySourceCons.isModifiable(), PySourceCons.isDynamic(),
            PySourceCons.isRemovable(), PySourceCons.isStickingAtNode()))
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitlp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool* infeasible):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    result_dict = PyConshdlr.consinitlp(constraints)
    infeasible[0] = result_dict.get("infeasible", infeasible[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsSepalp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    result_dict = PyConshdlr.conssepalp(constraints, nusefulconss)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsSepasol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                 SCIP_SOL* sol, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    solution = Solution()
    solution.sol = sol
    result_dict = PyConshdlr.conssepasol(constraints, nusefulconss, solution)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnfolp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                SCIP_Bool solinfeasible, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    result_dict = PyConshdlr.consenfolp(constraints, nusefulconss, solinfeasible)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnforelax (SCIP* scip, SCIP_SOL* sol, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    solution = Solution()
    solution.sol = sol
    result_dict = PyConshdlr.consenforelax(solution, constraints, nusefulconss, solinfeasible)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnfops (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                SCIP_Bool solinfeasible, SCIP_Bool objinfeasible, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    result_dict = PyConshdlr.consenfops(constraints, nusefulconss, solinfeasible, objinfeasible)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsCheck (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_SOL* sol, SCIP_Bool checkintegrality,
                               SCIP_Bool checklprows, SCIP_Bool printreason, SCIP_Bool completely, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    solution = Solution()
    solution.sol = sol
    result_dict = PyConshdlr.conscheck(constraints, solution, checkintegrality, checklprows, printreason, completely)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsProp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, int nmarkedconss,
                              SCIP_PROPTIMING proptiming, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    result_dict = PyConshdlr.consprop(constraints, nusefulconss, nmarkedconss, proptiming)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsPresol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    # dictionary for input/output parameters
    result_dict = {}
    result_dict["nfixedvars"]   = nfixedvars[0]
    result_dict["naggrvars"]    = naggrvars[0]
    result_dict["nchgvartypes"] = nchgvartypes[0]
    result_dict["nchgbds"]      = nchgbds[0]
    result_dict["naddholes"]    = naddholes[0]
    result_dict["ndelconss"]    = ndelconss[0]
    result_dict["naddconss"]    = naddconss[0]
    result_dict["nupgdconss"]   = nupgdconss[0]
    result_dict["nchgcoefs"]    = nchgcoefs[0]
    result_dict["nchgsides"]    = nchgsides[0]
    result_dict["result"]       = result[0]
    PyConshdlr.conspresol(constraints, nrounds, presoltiming,
                          nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                          nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict)
    result[0]       = result_dict["result"]
    nfixedvars[0]   = result_dict["nfixedvars"]
    naggrvars[0]    = result_dict["naggrvars"]
    nchgvartypes[0] = result_dict["nchgvartypes"]
    nchgbds[0]      = result_dict["nchgbds"]
    naddholes[0]    = result_dict["naddholes"]
    ndelconss[0]    = result_dict["ndelconss"]
    naddconss[0]    = result_dict["naddconss"]
    nupgdconss[0]   = result_dict["nupgdconss"]
    nchgcoefs[0]    = result_dict["nchgcoefs"]
    nchgsides[0]    = result_dict["nchgsides"]
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsResprop (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR* infervar, int inferinfo,
                                 SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyConshdlr.consresprop()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsLock (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_LOCKTYPE locktype, int nlockspos, int nlocksneg):
    PyConshdlr = getPyConshdlr(conshdlr)
    if cons == NULL:
        PyConshdlr.conslock(None, locktype, nlockspos, nlocksneg)
    else:
        PyCons = getPyCons(cons)
        PyConshdlr.conslock(PyCons, locktype, nlockspos, nlocksneg)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsActive (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    PyConshdlr.consactive(PyCons)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDeactive (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    PyConshdlr.consdeactive(PyCons)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnable (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    PyConshdlr.consenable(PyCons)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDisable (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    PyConshdlr.consdisable(PyCons)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDelvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    PyConshdlr = getPyConshdlr(conshdlr)
    cdef constraints = []
    for i in range(nconss):
        constraints.append(getPyCons(conss[i]))
    PyConshdlr.consdelvars(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsPrint (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, FILE* file):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    # TODO: pass file
    PyConshdlr.consprint(PyCons)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsCopy (SCIP* scip, SCIP_CONS** cons, const char* name, SCIP* sourcescip, SCIP_CONSHDLR* sourceconshdlr,
                              SCIP_CONS* sourcecons, SCIP_HASHMAP* varmap, SCIP_HASHMAP* consmap, SCIP_Bool initial,
                              SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate, SCIP_Bool local,
                              SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable, SCIP_Bool stickingatnode,
                              SCIP_Bool isglobal, SCIP_Bool* valid):
    # TODO everything!
    PyConshdlr = getPyConshdlr(sourceconshdlr)
    PyConshdlr.conscopy()
    valid[0] = False
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsParse (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** cons, const char* name, const char* str,
                               SCIP_Bool initial, SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate,
                               SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable,
                               SCIP_Bool stickingatnode, SCIP_Bool* success):
    # TODO everything!
    PyConshdlr = getPyConshdlr(conshdlr)
    PyConshdlr.consparse()
    success[0] = False
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR** vars, int varssize, SCIP_Bool* success):
    # TODO
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    PyConshdlr.consgetvars(PyCons)
    success[0] = False
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetnvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int* nvars, SCIP_Bool* success):
    PyConshdlr = getPyConshdlr(conshdlr)
    PyCons = getPyCons(cons)
    result_dict = PyConshdlr.consgetnvars(PyCons)
    nvars[0] = result_dict.get("nvars", 0)
    success[0] = result_dict.get("success", False)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetdivebdchgs (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_DIVESET* diveset, SCIP_SOL* sol,
                                       SCIP_Bool* success, SCIP_Bool* infeasible):
    # TODO
    PyConshdlr = getPyConshdlr(conshdlr)
    PyConshdlr.consgetdivebdchgs()
    success[0] = False
    infeasible[0] = False
    return SCIP_OKAY
