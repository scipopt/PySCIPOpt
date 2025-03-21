from pyscipopt import LP
from pyscipopt import SCIP_LPPARAM

def test_lp():
    # create LP instance, minimizing by default
    myLP = LP()

    # get default int and real parameters, some solver-specific parameters are commented.
    defaultLPParFromScratch = myLP.getIntParam(SCIP_LPPARAM.FROMSCRATCH)
    defaultLPParScaling = myLP.getIntParam(SCIP_LPPARAM.SCALING)
    defaultLPParPricing = myLP.getIntParam(SCIP_LPPARAM.PRICING)
    defaultLPParLpinfo = myLP.getIntParam(SCIP_LPPARAM.LPINFO)
    defaultLPParLpitlim = myLP.getIntParam(SCIP_LPPARAM.LPITLIM)
    defaultLPParFeastol = myLP.getRealParam(SCIP_LPPARAM.FEASTOL)
    defaultLPParDualfeastol = myLP.getRealParam(SCIP_LPPARAM.DUALFEASTOL)
    defaultLPParObjlim = myLP.getRealParam(SCIP_LPPARAM.OBJLIM)
    defaultLPParLptilim = myLP.getRealParam(SCIP_LPPARAM.LPTILIM)

    # try the following nondefault parameters
    tryLPParFromScratch = 0 if defaultLPParFromScratch == 1 else 1
    tryLPParScaling = 0 if defaultLPParScaling == 1 else 1
    tryLPParPricing = 0 if defaultLPParPricing == 1 else 1
    tryLPParLpinfo = 0 if defaultLPParLpinfo == 1 else 1
    tryLPParLpitlim = max(defaultLPParLpitlim - 1, 0)
    tryLPParFeastol = defaultLPParFeastol + 0.1
    tryLPParDualfeastol = defaultLPParDualfeastol + 0.1
    tryLPParObjlim = defaultLPParObjlim + 1.0
    tryLPParLptilim = defaultLPParLptilim + 1.0

    myLP.setIntParam(SCIP_LPPARAM.FROMSCRATCH, tryLPParFromScratch)
    myLP.setIntParam(SCIP_LPPARAM.SCALING, tryLPParScaling)
    myLP.setIntParam(SCIP_LPPARAM.PRICING, tryLPParPricing)
    myLP.setIntParam(SCIP_LPPARAM.LPINFO, tryLPParLpinfo)
    myLP.setIntParam(SCIP_LPPARAM.LPITLIM, tryLPParLpitlim)
    myLP.setRealParam(SCIP_LPPARAM.FEASTOL, tryLPParFeastol)
    myLP.setRealParam(SCIP_LPPARAM.DUALFEASTOL, tryLPParDualfeastol)
    myLP.setRealParam(SCIP_LPPARAM.OBJLIM, tryLPParObjlim)
    myLP.setRealParam(SCIP_LPPARAM.LPTILIM, tryLPParLptilim)

    assert tryLPParFromScratch == myLP.getIntParam(SCIP_LPPARAM.FROMSCRATCH)
    assert tryLPParScaling == myLP.getIntParam(SCIP_LPPARAM.SCALING)
    assert tryLPParPricing == myLP.getIntParam(SCIP_LPPARAM.PRICING)
    assert tryLPParLpinfo == myLP.getIntParam(SCIP_LPPARAM.LPINFO)
    assert tryLPParLpitlim == myLP.getIntParam(SCIP_LPPARAM.LPITLIM)
    assert tryLPParFeastol == myLP.getRealParam(SCIP_LPPARAM.FEASTOL)
    assert tryLPParDualfeastol == myLP.getRealParam(SCIP_LPPARAM.DUALFEASTOL)
    assert tryLPParObjlim == myLP.getRealParam(SCIP_LPPARAM.OBJLIM)
    assert tryLPParLptilim == myLP.getRealParam(SCIP_LPPARAM.LPTILIM)

    # set back default parameters
    myLP.setIntParam(SCIP_LPPARAM.FROMSCRATCH, defaultLPParFromScratch)
    myLP.setIntParam(SCIP_LPPARAM.SCALING, defaultLPParScaling)
    myLP.setIntParam(SCIP_LPPARAM.PRICING, defaultLPParPricing)
    myLP.setIntParam(SCIP_LPPARAM.LPINFO, defaultLPParLpinfo)
    myLP.setIntParam(SCIP_LPPARAM.LPITLIM, defaultLPParLpitlim)
    myLP.setRealParam(SCIP_LPPARAM.FEASTOL, defaultLPParFeastol)
    myLP.setRealParam(SCIP_LPPARAM.DUALFEASTOL, defaultLPParDualfeastol)
    myLP.setRealParam(SCIP_LPPARAM.OBJLIM, defaultLPParObjlim)
    myLP.setRealParam(SCIP_LPPARAM.LPTILIM, defaultLPParLptilim)

    # create cols w/o coefficients, 0 objective coefficient and 0,\infty bounds
    myLP.addCols(2 * [[]])

    # create rows
    myLP.addRow(entries = [(0,1),(1,2)] ,lhs = 5)
    lhs, rhs = myLP.getSides()
    assert lhs[0] == 5.0
    assert rhs[0] == myLP.infinity()

    assert(myLP.ncols() == 2)
    myLP.chgObj(0, 1.0)
    myLP.chgObj(1, 4.0)

    solval = myLP.solve()

    assert(myLP.isOptimal())
    assert(myLP.getPrimal() is not None)
    assert(myLP.getDual() is not None)
    assert(myLP.getRedcost() is not None)
    assert(myLP.getActivity() is not None)
    assert round(myLP.getObjVal() == solval)

    assert round(5.0 == solval)
