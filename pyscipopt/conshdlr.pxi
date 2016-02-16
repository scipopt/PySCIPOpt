cdef SCIP_RETCODE PyConshdlrCopy (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_Bool* valid):
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsFree (SCIP* scip, SCIP_CONSHDLR* conshdlr):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.free()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInit (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.init()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExit (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.exit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitpre (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.initpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExitpre (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.exitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitsol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.initsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsExitsol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool restart):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.exitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDelete (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_CONSDATA** consdata):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.delete()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsTrans (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* sourcecons, SCIP_CONS** targetcons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.trans()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsInitlp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.initlp()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsSepalp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                                                SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.sepalp()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsSepasol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                 SCIP_SOL* sol, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.sepasol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnfolp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                SCIP_Bool solinfeasible, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.enfolp()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnfops (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss,
                                SCIP_Bool solinfeasible, SCIP_Bool objinfeasible, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.enfops()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsCheck (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_SOL* sol, SCIP_Bool checkintegrality,
                               SCIP_Bool checklprows, SCIP_Bool printreason, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.check()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsProp (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, int nmarkedconss,
                              SCIP_PROPTIMING proptiming, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.Prop()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsPresol (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nrounds,
                                SCIP_PRESOLTIMING presoltiming, int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes,
                                int nnewchgbds, int nnewholes, int nnewdelconss, int nnewaddconss, int nnewupgdconss,
                                int nnewchgcoefs, int nnewchgsides, int* nfixedvars, int* naggrvars, int* nchgvartypes,
                                int* nchgbds, int* naddholes, int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs,
                                int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.presol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsResprop (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR* infervar, int inferinfo,
                                 SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.resprop()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsLock (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int nlockspos, int nlocksneg):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.lock()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsActive (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.active()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDeactive (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.deactive()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsEnable (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.enable()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDisable (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.disable()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsDelvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.delvars()
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
    PyConshdlr.copy()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsParse (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** cons, const char* name, const char* str,
                               SCIP_Bool initial, SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate,
                               SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable,
                               SCIP_Bool stickingatnode, SCIP_Bool* success):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.parse()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR** vars, int varssize, SCIP_Bool* success):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.getvars()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetnvars (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int* nvars, SCIP_Bool* success):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.getnvars()
    return SCIP_OKAY

cdef SCIP_RETCODE PyConsGetdivebdchgs (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_DIVESET* diveset, SCIP_SOL* sol,
                                       SCIP_Bool* success, SCIP_Bool* infeasible):
    cdef SCIP_CONSHDLRDATA* conshdlrdata
    conshdlrdata = SCIPconshdlrGetData(conshdlr)
    PyConshdlr = <Conshdlr>conshdlrdata
    PyConshdlr.getdivebdchgs()
    return SCIP_OKAY

cdef class Conshdlr:
    cdef public object data     # storage for the python user
    cdef public Model model

    def free(self):
        pass

    def init(self):
        pass

    def exit(self):
        pass

    def initpre(self):
        pass

    def exitpre(self):
        pass

    def initsol(self):
        pass

    def exitsol(self):
        pass

    def delete(self):
        pass

    def trans(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def initlp(self):
        pass

    def sepalp(self):
        pass

    def sepasol(self):
        pass

    def enfolp(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def enfops(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def check(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def prop(self):
        pass

    def presol(self):
        pass

    def resprop(self):
        pass

    def lock(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def active(self):
        pass

    def deactive(self):
        pass

    def enable(self):
        pass

    def disable(self):
        pass

    def delvars(self):
        pass

    def consprint(self):
        pass

    def copy(self):
        pass

    def parse(self):
        pass

    def getvars(self):
        pass

    def getnvars(self):
        pass

    def getdivebdchgs(self):
        pass
