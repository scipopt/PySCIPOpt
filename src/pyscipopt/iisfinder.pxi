##@file iisfinder.pxi
#@brief Base class of the IIS finder Plugin
cdef class IISfinder:
    cdef public Model model
    cdef public str name
    cdef SCIP_IIS* _iisfinder

    def iisfinderfree(self):
        '''calls destructor and frees memory of iis finder'''
        pass
        
    def iisfinderexec(self):
        '''calls execution method of iis finder'''
        raise NotImplementedError("iisfinderexec() is a fundamental callback and should be implemented in the derived class")


cdef SCIP_RETCODE PyiisfinderCopy (SCIP* scip, SCIP_IISFINDER* iisfinder) noexcept with gil:
    return SCIP_OKAY

cdef SCIP_RETCODE PyiisfinderFree (SCIP* scip, SCIP_IISFINDER* iisfinder) noexcept with gil:
    cdef SCIP_IISFINDERDATA* iisfinderdata
    iisfinderdata = SCIPiisfinderGetData(iisfinder)
    PyIIS = <IISfinder>iisfinderdata
    PyIIS.iisfinderfree()
    Py_DECREF(PyIIS)
    return SCIP_OKAY

cdef SCIP_RETCODE PyiisfinderExec (SCIP_IIS* iis, SCIP_IISFINDER* iisfinder, SCIP_Real timelim, SCIP_Longint nodelim, SCIP_Bool removebounds, SCIP_Bool silent, SCIP_RESULT* result) noexcept with gil:
    cdef SCIP_IISFINDERDATA* iisfinderdata
    iisfinderdata = SCIPiisfinderGetData(iisfinder)
    PyIIS = <IISfinder>iisfinderdata
    result_dict = PyIIS.iisfinderexec()
    assert isinstance(result_dict, dict), "iisfinderexec() must return a dictionary."
    #TODO
    assert False
    # lowerbound[0] = result_dict.get("lowerbound", <SCIP_Real>lowerbound[0])
    # result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    # return SCIP_OKAY