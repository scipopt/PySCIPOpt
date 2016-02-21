cdef SCIP_RETCODE PyPropCopy (SCIP* scip, SCIP_PROP* prop):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropFree (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.free()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInit (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.init()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExit (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.exit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInitpre (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.initpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExitpre (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.exitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInitsol (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.initsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExitsol (SCIP* scip, SCIP_PROP* prop, SCIP_Bool restart):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.exitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropPresol (SCIP* scip, SCIP_PROP* prop, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    returnvalues = PyProp.presol()
    result_dict = returnvalues
    result[0] = result_dict.get("result", SCIP_SUCCESS)
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

    def free(self):
        pass

    def init(self):
        pass

    def exit(self):
        pass

    def initsol(self):
        pass

    def exitsol(self):
        pass

    def initpre(self):
        pass

    def exitpre(self):
        pass

    def presol(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def propexec(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}

    def resprop(self):
        # this method needs to be implemented by the user
        return {"result": SCIP_DIDNOTRUN}
