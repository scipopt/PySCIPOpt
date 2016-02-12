
cdef SCIP_RETCODE PyPricerCopy (SCIP* scip, SCIP_PRICER* pricer, SCIP_Bool* valid):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerFree (SCIP* scip, SCIP_PRICER* pricer):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerInit (SCIP* scip, SCIP_PRICER* pricer):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerExit (SCIP* scip, SCIP_PRICER* pricer):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerInitsol (SCIP* scip, SCIP_PRICER* pricer):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerExitsol (SCIP* scip, SCIP_PRICER* pricer):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerRedcost (SCIP* scip, SCIP_PRICER* pricer, SCIP_Real* lowerbound, SCIP_Bool* stopearly, SCIP_RESULT* result):
    cdef SCIP_PRICERDATA* pricerdata
    pricerdata = SCIPpricerGetData(pricer)
    PyPricer = <Pricer>pricerdata
    returnvalues = PyPricer.redcost()
    result_dict = returnvalues
    result[0] = result_dict.get("result", SCIP_SUCCESS)
    lowerbound[0] = result_dict.get("lowerbound", -1e20)
    stopearly[0] = result_dict.get("stopearly", False)
    return SCIP_OKAY

cdef SCIP_RETCODE PyPricerFarkas (SCIP* scip, SCIP_PRICER* pricer, SCIP_RESULT* result):
    return SCIP_OKAY

cdef class Pricer:
    cdef public object data     # storage for the python user
    cdef public Model model

    def init(self):
        pass

    def redcost(self):
        print("It is necessary for the user to write a custom redcost function.")
        print("The redcost function should be included in the users own python files.")
        return {"result": SCIP_SUCCESS}

    def farkas(self):
        pass
