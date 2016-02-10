cimport pyscipopt.scip as scip

class pricerdata(object):
    pass

cdef class Pricer:
    cdef scip.SCIP_PRICER* _pricer

    cdef scip.SCIP_PRICERDATA* _pricerdata

    def getPricerData(self):
        return pricerdata

def init(Model model, Pricer pricer):
    pass

def redcost(Model model, Pricer pricer):
    print("It is necessary for the user to write a custom py_scip_redcost function.")
    print("The py_scip_redcost function should be included in the users own python files.")
    return scip.scip_result.success

def farkas():
    pass

cdef SCIP_RETCODE scipPricerInit(SCIP* _scip, SCIP_PRICER* _pricer):
    s = Model()
    s._scip = _scip
    pricer = Pricer()
    pricer._pricer = _pricer
    init(s, pricer)
    del s
    del pricer
    return scip.SCIP_OKAY

cdef SCIP_RETCODE scipPricerRedcost(SCIP* _scip,
                                    SCIP_PRICER* _pricer,
                                    SCIP_Real* _lowerbound,
                                    SCIP_Bool* _stopearly,
                                    SCIP_RESULT* _result):
    s = Model()
    s._scip = _scip
    pricer = Pricer()
    pricer._pricer = _pricer
    _result[0] = redcost(s, pricer)
    del s
    del pricer
    return scip.SCIP_OKAY

cdef SCIP_RETCODE scipPricerFarkas(SCIP* _scip, SCIP_PRICER* _pricer, SCIP_RESULT* _result):
    _result[0] = farkas()
    return scip.SCIP_OKAY
