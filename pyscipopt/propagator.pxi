cdef SCIP_RETCODE PyPropCopy (SCIP* scip, SCIP_PROP* prop):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropFree (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propfree()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInit (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExit (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInitpre (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propinitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExitpre (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propexitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInitsol (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExitsol (SCIP* scip, SCIP_PROP* prop, SCIP_Bool restart):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propexitsol(restart)
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropPresol (SCIP* scip, SCIP_PROP* prop, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    result_dict = {}
    result_dict = PyProp.proppresol(nrounds, presoltiming)
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

cdef SCIP_RETCODE PyPropExec (SCIP* scip, SCIP_PROP* prop, SCIP_PROPTIMING proptiming, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    returnvalues = PyProp.propexec()
    result_dict = returnvalues
    result[0] = result_dict.get("result", SCIP_DIDNOTFIND)
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropResProp (SCIP* scip, SCIP_PROP* prop, SCIP_VAR* infervar, int inferinfo,
                                 SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    returnvalues = PyProp.resprop()
    result_dict = returnvalues
    result[0] = result_dict.get("result", SCIP_SUCCESS)
    return SCIP_OKAY

cdef class Prop:
    cdef public object data     # storage for the python user
    cdef public Model model

    def propfree(self):
        pass

    def propinit(self):
        pass

    def propexit(self):
        pass

    def propinitsol(self):
        pass

    def propexitsol(self, restart):
        pass

    def propinitpre(self):
        pass

    def propexitpre(self):
        pass

    def proppresol(self, nrounds, presoltiming):
        # this method needs to be implemented by the user
        return {}

    def propexec(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def propresprop(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}
