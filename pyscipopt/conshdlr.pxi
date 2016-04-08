cdef SCIP_RETCODE PyConshdlrCopy (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_Bool* valid):
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsFree (SCIP* scip, SCIP_CONSHDLR* conshdlr):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consfree()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInit (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    PyConshdlr.consinit(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExit (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    PyConshdlr.consexit(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitpre (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    PyConshdlr.consinitpre(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExitpre (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    PyConshdlr.consexitpre(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitsol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    PyConshdlr.consinitsol(constraints)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExitsol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool restart):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    PyConshdlr.consexitsol(constraints, restart)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDelete (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_CONSDATA** consdata):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    # TODO
    PyConshdlr.consdelete()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsTrans (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* sourcecons, SCIP_CONS** targetcons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    sourceconstraint = Constraint.create(sourcecons, SCIPconsGetName(sourcecons).decode("utf-8"))
    # TODO
    PyConshdlr.constrans(sourceconstraint)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitlp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool* infeasible):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consinitlp()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsSepalp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    result_dict = PyConshdlr.conssepalp(constraints, nusefulconss)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsSepasol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                 SCIP_SOL* sol, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    solution = Solution()
    solution._solution = sol
    result_dict = PyConshdlr.conssepasol(constraints, nusefulconss, solution)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnfolp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                SCIP_Bool solinfeasible, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    result_dict = PyConshdlr.consenfolp(constraints, nusefulconss, solinfeasible)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnfops (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                SCIP_Bool solinfeasible, SCIP_Bool objinfeasible, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    result_dict = PyConshdlr.consenfops(constraints, nusefulconss, solinfeasible, objinfeasible)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsCheck (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_SOL* sol, SCIP_Bool checkintegrality,
                               SCIP_Bool checklprows, SCIP_Bool printreason, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    solution = Solution()
    solution._solution = sol
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    solution = Solution()
    solution._solution = sol
    result_dict = PyConshdlr.conscheck(constraints, solution, checkintegrality, checklprows, printreason)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsProp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, int nmarkedconss,
                              SCIP_PROPTIMING proptiming, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
    result_dict = PyConshdlr.consprop(constraints, nusefulconss, nmarkedconss, proptiming)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsPresol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    cdef constraints = []
    for i in range(nconss):
        constraints.append(Constraint.create(conss[i], SCIPconsGetName(conss[i]).decode("utf-8")))
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
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    # TODO
    PyConshdlr.consresprop()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsLock (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int nlockspos, int nlocksneg):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    constraint = (Constraint.create(cons, SCIPconsGetName(cons).decode("utf-8")))
    PyConshdlr.conslock(constraint, nlockspos, nlocksneg)
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsActive (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consactive()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDeactive (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consdeactive()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnable (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consenable()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDisable (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consdisable()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDelvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consdelvars()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsPrint (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, FILE* file):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consprint()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsCopy (SCIP* scip, SCIP_CONS** cons, const char* name, SCIP* sourcescip, SCIP_CONSHDLR* sourceconshdlr,
                              SCIP_CONS* sourcecons, SCIP_HASHMAP* varmap, SCIP_HASHMAP* consmap, SCIP_Bool initial,
                              SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate, SCIP_Bool local,
                              SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable, SCIP_Bool stickingatnode,
                              SCIP_Bool isglobal, SCIP_Bool* valid):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(sourceconshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.conscopy()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsParse (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** cons, const char* name, const char* str,
                               SCIP_Bool initial, SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate,
                               SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable,
                               SCIP_Bool stickingatnode, SCIP_Bool* success):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consparse()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR** vars, int varssize, SCIP_Bool* success):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consgetvars()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetnvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int* nvars, SCIP_Bool* success):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consgetnvars()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetdivebdchgs (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_DIVESET* diveset, SCIP_SOL* sol,
                                       SCIP_Bool* success, SCIP_Bool* infeasible):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.consgetdivebdchgs()
    return SCIP_OKAY

cdef class Conshdlr:
    cdef public object data     # storage for the python user
    cdef public Model model
    cdef public str name

    def consfree(self):
        pass

    def consinit(self, constraints):
        pass

    def consexit(self, constraints):
        pass

    def consinitpre(self, constraints):
        pass

    def consexitpre(self, constraints):
        pass

    def consinitsol(self, constraints):
        pass

    def consexitsol(self, constraints, restart):
        pass

    def consdelete(self):
        pass

    def constrans(self, sourceconstraint):
        print("python error in constrans: this method needs to be implemented")
        return {}

    def consinitlp(self):
        pass

    def conssepalp(self, constraints, nusefulconss):
        return {}

    def conssepasol(self, constraints, nusefulconss, solution):
        pass

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        print("python error in consenfolp: this method needs to be implemented")
        return {}

    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        print("python error in consenfops: this method needs to be implemented")
        return {}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason):
        print("python error in conscheck: this method needs to be implemented")
        return {}

    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        pass

    def conspresol(self, constraints, nrounds, presoltiming,
                   nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                   nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict):
        pass

    def consresprop(self):
        pass

    def conslock(self, constraint, nlockspos, nlocksneg):
        print("python error in conslock: this method needs to be implemented")
        return {}

    def consactive(self):
        pass

    def consdeactive(self):
        pass

    def consenable(self):
        pass

    def consdisable(self):
        pass

    def consdelvars(self):
        pass

    def consconsprint(self):
        pass

    def conscopy(self):
        pass

    def consparse(self):
        pass

    def consgetvars(self):
        pass

    def consgetnvars(self):
        pass

    def consgetdivebdchgs(self):
        pass
