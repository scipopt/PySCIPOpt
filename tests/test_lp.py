from pyscipopt import LP
from pyscipopt import SCIP_LPPARAM

def test_lp():
    # create LP instance, minimizing by default
    myLP = LP()

    # get default int parameters.
    lpParFromScratch = myLP.getIntParam(SCIP_LPPARAM.FROMSCRATCH)
    lpParScaling = myLP.getIntParam(SCIP_LPPARAM.SCALING)
    lpParPricing = myLP.getIntParam(SCIP_LPPARAM.PRICING)
    lpParLpinfo = myLP.getIntParam(SCIP_LPPARAM.LPINFO)
    lpParLpitlim = myLP.getIntParam(SCIP_LPPARAM.LPITLIM)
    # lpParFastmip = myLP.getIntParam(SCIP_LPPARAM.FASTMIP)


    # get default real parameters
    lpParFeastol = myLP.getRealParam(SCIP_LPPARAM.FEASTOL)
    lpParDualfeastol = myLP.getRealParam(SCIP_LPPARAM.DUALFEASTOL)
    # lpParBarrierconvtol = myLP.getRealParam(SCIP_LPPARAM.BARRIERCONVTOL)
    lpParObjlim = myLP.getRealParam(SCIP_LPPARAM.OBJLIM)
    lpParLptilim = myLP.getRealParam(SCIP_LPPARAM.LPTILIM)

    # set int parameters back
    myLP.setIntParam(SCIP_LPPARAM.FROMSCRATCH, lpParFromScratch)
    myLP.setIntParam(SCIP_LPPARAM.SCALING, lpParScaling)
    myLP.setIntParam(SCIP_LPPARAM.PRICING, lpParPricing)
    myLP.setIntParam(SCIP_LPPARAM.LPINFO, lpParLpinfo)
    myLP.setIntParam(SCIP_LPPARAM.LPITLIM, lpParLpitlim)
    # myLP.setIntParam(SCIP_LPPARAM.FASTMIP, lpParFastmip)

    # set real parameters back
    myLP.setRealParam(SCIP_LPPARAM.FEASTOL, lpParFeastol)
    myLP.setRealParam(SCIP_LPPARAM.DUALFEASTOL, lpParDualfeastol)
    # myLP.setRealParam(SCIP_LPPARAM.BARRIERCONVTOL, lpParBarrierconvtol)
    myLP.setRealParam(SCIP_LPPARAM.OBJLIM, lpParObjlim)
    myLP.setRealParam(SCIP_LPPARAM.LPTILIM, lpParLptilim)

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

    assert round(5.0 == solval)
