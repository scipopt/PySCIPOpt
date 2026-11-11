from fractions import Fraction

import pytest

from pyscipopt import LPExact, SCIP_BASESTAT, SCIP_LPPARAM

F = Fraction
INF = float("inf")


def _exact_lp_supported():
    try:
        LPExact()
    except ValueError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _exact_lp_supported(), reason="SCIP was built without an exact LP solver")


def all_fractions(values):
    return all(type(v) is Fraction for v in values)


def test_lpexact_build_solve_and_modify():
    lp = LPExact("exact")
    assert repr(lp) == "exact"
    assert lp.ncols() == 0 and lp.nrows() == 0
    assert lp.infinity() == INF
    assert lp.isInfinity(lp.infinity()) and not lp.isInfinity(F(10) ** 30)

    lp.addCols([[], []], objs=[F(1, 3), F(1, 7)])
    lp.addRow([(0, F(2, 3)), (1, F(5, 7))], lhs=F(1, 11))
    assert lp.ncols() == 2 and lp.nrows() == 1
    assert lp.getBounds() == ([0, 0], [INF, INF])
    assert lp.getSides() == ([F(1, 11)], [INF])
    assert lp.getBounds(1, 0) is None

    objval = lp.solve()
    assert lp.isOptimal() and lp.isPrimalFeasible() and lp.isDualFeasible()
    assert objval == F(1, 55) and type(objval) is Fraction
    assert lp.getObjVal() == F(1, 55)
    assert lp.getPrimal() == [0, F(7, 55)]
    assert lp.getDual() == [F(1, 5)]
    assert lp.getRedcost() == [F(1, 5), 0]
    assert lp.getActivity() == [F(1, 11)]
    assert all_fractions(lp.getPrimal() + lp.getDual() + lp.getRedcost() + lp.getActivity())
    assert lp.getNIterations() >= 0
    assert lp.getPrimalRay() is None and lp.getDualRay() is None

    assert len(lp.getBasisInds()) == lp.nrows()
    cstat, rstat = lp.getBase()
    assert len(cstat) == 2 and len(rstat) == 1
    assert set(cstat) <= {SCIP_BASESTAT.LOWER, SCIP_BASESTAT.BASIC, SCIP_BASESTAT.UPPER, SCIP_BASESTAT.ZERO}
    lp.setBase(cstat, rstat)
    assert lp.solve(dual=False) == F(1, 55)
    assert lp.getBase() == (cstat, rstat)
    with pytest.raises(ValueError):
        lp.setBase([SCIP_BASESTAT.BASIC], rstat)

    lp.chgObj(0, F(1, 10))
    assert lp.solve() == F(3, 220)
    lp.chgBound(0, 0, F(1, 22))
    assert lp.solve() == F(1, 60)
    assert lp.getBounds(0, 0) == ([0], [F(1, 22)])
    lp.chgSide(0, F(1, 11), F(1, 11))
    assert lp.solve() == F(1, 60)
    assert lp.getSides() == ([F(1, 11)], [F(1, 11)])
    lp.chgCoef(0, 1, F(10, 7))
    assert lp.solve() == F(1, 110)
    assert lp.getPrimal() == [0, F(7, 110)]

    lp.addCol([(0, 1)], obj=F(1, 2), lb=0, ub=1)
    assert lp.ncols() == 3
    assert lp.solve() == F(1, 110) and lp.getPrimal()[2] == 0
    lp.delCols(2, 2)
    lp.addRows([[(0, 1), (1, 1)]], lhss=[-INF], rhss=[F(1, 2)])
    assert lp.nrows() == 2
    assert lp.getSides(1, 1) == ([-INF], [F(1, 2)])
    assert lp.solve() == F(1, 110)
    lp.delRows(1, 1)
    assert lp.nrows() == 1

    lp.addCol([], obj="1/9")
    assert lp.solve() == F(1, 110)

    with pytest.raises(ValueError):
        lp.addCol([], obj=float("nan"))
    with pytest.raises(TypeError):
        lp.addRow([(0, None)])
    assert lp.ncols() == 3 and lp.nrows() == 1

    lp.clear()
    assert lp.ncols() == 0 and lp.nrows() == 0

    with pytest.raises(Warning):
        LPExact(sense="sideways")


def test_lpexact_unbounded_and_infeasible():
    lp = LPExact(sense="maximize")
    lp.addCol([], obj=1)
    lp.solve()
    assert not lp.isOptimal()
    ray = lp.getPrimalRay()
    assert len(ray) == 1 and ray[0] > 0 and all_fractions(ray)

    lp = LPExact()
    lp.addCol([], lb=0, ub=1)
    lp.addRow([(0, 1)], lhs=2)
    lp.solve()
    assert not lp.isOptimal() and not lp.isPrimalFeasible()
    farkas = lp.getDualRay()
    assert len(farkas) == 1 and farkas[0] != 0 and all_fractions(farkas)


def test_lpexact_params_and_files(tmp_path):
    lp = LPExact()
    for param, value in [(SCIP_LPPARAM.FROMSCRATCH, 1), (SCIP_LPPARAM.LPITLIM, 7), (SCIP_LPPARAM.SCALING, 0)]:
        lp.setIntParam(param, value)
        assert lp.getIntParam(param) == value
    for param, value in [(SCIP_LPPARAM.LPTILIM, 12.5), (SCIP_LPPARAM.OBJLIM, 3.0)]:
        lp.setRealParam(param, value)
        assert lp.getRealParam(param) == value
    with pytest.raises(KeyError):
        lp.getRealParam(SCIP_LPPARAM.BARRIERCONVTOL)
    with pytest.raises(KeyError):
        lp.setIntParam(SCIP_LPPARAM.THREADS, 2)

    lp.addCols([[], []], objs=[1, 4])
    lp.addRow([(0, 1), (1, 2)], lhs=5)
    path = tmp_path / "exact.lp"
    lp.writeLP(str(path))
    assert path.exists()

    other = LPExact()
    other.readLP(str(path))
    assert other.ncols() == 2 and other.nrows() == 1
    assert other.solve() == 5
