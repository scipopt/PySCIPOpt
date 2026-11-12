##@file symhdlr.pxi
#@brief Base class of the Symmetry Handler Plugin
cdef class SymComp:
    @staticmethod
    cdef create(SCIP_SYMCOMP* scip_symcomp):
        if scip_symcomp == NULL:
            raise Warning("cannot create SymComp with SCIP_SYMCOMP* == NULL")
        symcomp = SymComp()
        symcomp.scip_symcomp = scip_symcomp
        return symcomp

    @property
    def name(self):
        return bytes(SCIPsymcompGetName(self.scip_symcomp)).decode('utf-8')

    @property
    def data(self):
        return <object>SCIPsymcompGetData(self.scip_symcomp)

    def ptr(self):
        return <size_t>self.scip_symcomp

cdef class Symhdlr:
    cdef public Model model
    cdef public str name
    cdef list _symcompdata

    def symfree(self):
        '''calls destructor and frees memory of symmetry handler'''
        pass

    def syminit(self, symcomps):
        '''initializes symmetry handler'''
        pass

    def symexit(self, symcomps):
        '''calls exit method of symmetry handler'''
        pass

    def syminitsol(self, symcomps):
        '''informs symmetry handler that the branch and bound process is being started'''
        pass

    def symexitsol(self, symcomps, restart):
        '''informs symmetry handler that the branch and bound process data is being freed'''
        pass

    def symtryadd(self, symtype, perms, permvars, permvardomcenter, id, allowbdchgs):
        '''tries to add a symmetry handling method for the given symmetries'''
        raise NotImplementedError("symtryadd() is a fundamental callback and should be implemented in the derived class")

    def symsepalp(self, symcomps, allowlocal, depth):
        '''separates the current LP solution'''
        return {"result": SCIP_DIDNOTRUN}

    def symsepasol(self, solution, symcomps, allowlocal, depth):
        '''separates the given primal solution'''
        return {"result": SCIP_DIDNOTRUN}

    def symprop(self, symcomps, proptiming):
        '''propagates domains'''
        return {"result": SCIP_DIDNOTRUN}

    def symresprop(self, symcomp, confvar, inferinfo, bdtype, relaxedbd):
        '''resolves the given conflicting bound, that was reduced by the given symmetry component'''
        return {"result": SCIP_DIDNOTFIND}

    def sympresol(self, symcomps, nrounds, presoltiming,
                  nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                  nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict):
        '''executes presolving method of symmetry handler'''
        pass


cdef Symhdlr getPySymhdlr(SCIP_SYMHDLR* symhdlr):
    return <Symhdlr>SCIPsymhdlrGetData(symhdlr)

cdef list getPySymComps(SCIP_SYMCOMP** symcomps, int nsymcomps):
    return [SymComp.create(symcomps[i]) for i in range(nsymcomps)]

cdef SCIP_RETCODE PySymhdlrTryAdd (SCIP* scip, SCIP_SYMHDLR* symhdlr, SYM_SYMTYPE symtype, int** perms, int nperms,
                                   SCIP_VAR** permvars, int npermvars, SCIP_Real* permvardomcenter, SCIP_HASHMAP* permvarmap,
                                   SYM_GRAPH* symgraph, int id, SCIP_SYMCOMPDATA** symcompdata, int* naddedconss,
                                   SCIP_Bool allowbdchgs, int* nchgbds, SCIP_Bool* success) noexcept with gil:
    cdef int permlen = npermvars if symtype == SYM_SYMTYPE_PERM else 2 * npermvars
    PySymhdlr = getPySymhdlr(symhdlr)
    pyperms = [[perms[p][i] for i in range(permlen)] for p in range(nperms)]
    pypermvars = [Variable.create(permvars[i]) for i in range(npermvars)]
    if permvardomcenter == NULL:
        pydomcenter = None
    else:
        pydomcenter = [permvardomcenter[i] for i in range(npermvars)]

    result_dict = PySymhdlr.symtryadd(symtype, pyperms, pypermvars, pydomcenter, id, allowbdchgs)
    assert isinstance(result_dict, dict), "symtryadd() must return a dictionary."
    success[0] = result_dict.get("success", False)
    naddedconss[0] = result_dict.get("naddedconss", 0)
    nchgbds[0] = result_dict.get("nchgbds", 0)
    if success[0]:
        data = result_dict.get("data")
        PySymhdlr._symcompdata.append(data)
        symcompdata[0] = <SCIP_SYMCOMPDATA*>data
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrFree (SCIP* scip, SCIP_SYMHDLR* symhdlr) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    PySymhdlr.symfree()
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrInit (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    PySymhdlr.syminit(getPySymComps(symcomps, nsymcomps))
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrExit (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    PySymhdlr.symexit(getPySymComps(symcomps, nsymcomps))
    PySymhdlr._symcompdata.clear()
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrInitsol (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    PySymhdlr.syminitsol(getPySymComps(symcomps, nsymcomps))
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrExitsol (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps, SCIP_Bool restart) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    PySymhdlr.symexitsol(getPySymComps(symcomps, nsymcomps), restart)
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrSepaLP (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps,
                                   SCIP_RESULT* result, SCIP_Bool allowlocal, int depth) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    result_dict = PySymhdlr.symsepalp(getPySymComps(symcomps, nsymcomps), allowlocal, depth)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrSepaSol (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SOL* sol, SCIP_SYMCOMP** symcomps, int nsymcomps,
                                    SCIP_RESULT* result, SCIP_Bool allowlocal, int depth) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    solution = Solution.create(scip, sol)
    result_dict = PySymhdlr.symsepasol(solution, getPySymComps(symcomps, nsymcomps), allowlocal, depth)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrProp (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps,
                                 SCIP_PROPTIMING proptiming, SCIP_RESULT* result) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    result_dict = PySymhdlr.symprop(getPySymComps(symcomps, nsymcomps), proptiming)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrResProp (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP* symcomp, SCIP_VAR* infervar, int inferinfo,
                                    SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
    confvar = Variable.create(infervar)
    result_dict = PySymhdlr.symresprop(SymComp.create(symcomp), confvar, inferinfo, boundtype, relaxedbd)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PySymhdlrPresol (SCIP* scip, SCIP_SYMHDLR* symhdlr, SCIP_SYMCOMP** symcomps, int nsymcomps, int nrounds,
                                   SCIP_PRESOLTIMING presoltiming, int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes,
                                   int nnewchgbds, int nnewholes, int nnewdelconss, int nnewaddconss, int nnewupgdconss,
                                   int nnewchgcoefs, int nnewchgsides, int* nfixedvars, int* naggrvars, int* nchgvartypes,
                                   int* nchgbds, int* naddholes, int* ndelconss, int* naddconss, int* nupgdconss,
                                   int* nchgcoefs, int* nchgsides, SCIP_RESULT* result) noexcept with gil:
    PySymhdlr = getPySymhdlr(symhdlr)
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
    PySymhdlr.sympresol(getPySymComps(symcomps, nsymcomps), nrounds, presoltiming,
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
