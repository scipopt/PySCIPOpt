##@file pricer.pxi
#@brief Base class of the Pricers Plugin
cdef class Pricer:
    cdef public Model model

    def pricerfree(self):
        '''calls destructor and frees memory of variable pricer '''
        pass

    def pricerinit(self):
        '''initializes variable pricer'''
        pass

    def pricerexit(self):
        '''calls exit method of variable pricer'''
        pass

    def pricerinitsol(self):
        '''informs variable pricer that the branch and bound process is being started '''
        pass

    def pricerexitsol(self):
        '''informs variable pricer that the branch and bound process data is being freed''' 
        pass

    def pricerredcost(self):
        '''calls reduced cost pricing method of variable pricer'''
        print("python error in pricerredcost: this method needs to be implemented")
        return {}

    def pricerfarkas(self):
        '''calls Farkas pricing method of variable pricer'''
        return {}



cdef SCIP_RETCODE PyPricerCopy (SCIP* scip, SCIP_PRICER* pricer, SCIP_Bool* valid):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerFree (SCIP* scip, SCIP_PRICER* pricer):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    PyPricer.pricerfree()
    Py_DECREF(PyPricer)
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerInit (SCIP* scip, SCIP_PRICER* pricer):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    PyPricer.pricerinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerExit (SCIP* scip, SCIP_PRICER* pricer):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    PyPricer.pricerexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerInitsol (SCIP* scip, SCIP_PRICER* pricer):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    PyPricer.pricerinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerExitsol (SCIP* scip, SCIP_PRICER* pricer):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    PyPricer.pricerexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerRedcost (SCIP* scip, SCIP_PRICER* pricer, SCIP_Real* lowerbound, SCIP_Bool* stopearly, SCIP_RESULT* result):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    result_dict = PyPricer.pricerredcost()
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    lowerbound[0] = result_dict.get("lowerbound", <SCIP_Real>lowerbound[0])
    stopearly[0] = result_dict.get("stopearly", <SCIP_Bool>stopearly[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerFarkas (SCIP* scip, SCIP_PRICER* pricer, SCIP_RESULT* result):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    result[0] = PyPricer.pricerfarkas().get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY
