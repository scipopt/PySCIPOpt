# Copyright (C) 2012-2013 ZIB
#   see file 'LICENSE' for details.

cimport pyscipopt.scip as scip

class py_pricerdata(object):
    pass

cdef class Pricer:
    cdef scip.SCIP_PRICER* _pricer

    cdef scip.SCIP_PRICERDATA* _pricerdata

    def getPricerData(self):
        return py_pricerdata


def py_scip_init(Model model, Pricer pricer):
    pass

def py_scip_redcost(Model model, Pricer pricer):
    print "It is necessary for the user to write a custom py_scip_redcost function."
    print "The py_scip_redcost function should be included in the users own python files."
    return scip.scip_result.success

def py_scip_farkas():
    pass


cdef SCIP_RETCODE scipPricerInit(SCIP* _scip, SCIP_PRICER* _pricer):
    s = Model()
    s._scip = _scip
    pricer = Pricer()
    pricer._pricer = _pricer
    py_scip_init(s, pricer)
    del s
    del pricer
    return scip.SCIP_OKAY



cdef SCIP_RETCODE scipPricerRedcost(SCIP* _scip, SCIP_PRICER* _pricer, SCIP_Real* _lowerbound, SCIP_Bool* _stopearly, SCIP_RESULT* _result):
    s = Model()
    s._scip = _scip
    pricer = Pricer()
    pricer._pricer = _pricer
    _result[0] = py_scip_redcost(s, pricer)
    del s
    del pricer
    return scip.SCIP_OKAY


cdef SCIP_RETCODE scipPricerFarkas(SCIP* _scip, SCIP_PRICER* _pricer, SCIP_RESULT* _result):
    _result[0] = py_scip_farkas()
    return scip.SCIP_OKAY
