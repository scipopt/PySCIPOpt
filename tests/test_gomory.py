import pytest

from pyscipopt import Model, Sepa, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import is_memory_freed

class GMI(Sepa):

    def __init__(self):
        self.ncuts = 0

    def getGMIFromRow(self, cols, rows, binvrow, binvarow, primsol):
        """ Given the row (binvarow, binvrow) of the tableau, computes gomory cut

        :param primsol: is the rhs of the tableau row.
        :param cols:    are the variables
        :param rows:    are the slack variables
        :param binvrow: components of the tableau row associated to the basis inverse
        :param binvarow: components of the tableau row associated to the basis inverse * A

        The GMI is given by
         sum(f_j x_j                  , j in J_I s.t. f_j <= f_0) +
         sum((1-f_j)*f_0/(1 - f_0) x_j, j in J_I s.t. f_j  > f_0) +
         sum(a_j x_j,                 , j in J_C s.t. a_j >=   0) -
         sum(a_j*f_0/(1-f_0) x_j      , j in J_C s.t. a_j  <   0) >= f_0.
        where J_I are the integer non-basic variables and J_C are the continuous.
        f_0 is the fractional part of primsol
        a_j is the j-th coefficient of the row and f_j its fractional part
        Note: we create -% <= -f_0 !!
        Note: this formula is valid for a problem of the form Ax = b, x>= 0. Since we do not have
        such problem structure in general, we have to (implicitely) transform whatever we are given
        to that form. Specifically, non-basic variables at their lower bound are shifted so that the lower
        bound is 0 and non-basic at their upper bound are complemented.
        """

        # initialize
        cutcoefs = [0] * len(cols)
        cutrhs = 0

        # get scip
        scip = self.model

        # Compute cut fractionality f0 and f0/(1-f0)
        f0 = scip.frac(primsol)
        ratiof0compl = f0/(1-f0)

        # rhs of the cut is the fractional part of the LP solution for the basic variable
        cutrhs = -f0

        #  Generate cut coefficients for the original variables
        for c in range(len(cols)):
             col = cols[c]
             assert col is not None # is this the equivalent of col != NULL? does it even make sense to have this assert?
             status = col.getBasisStatus()

             # Get simplex tableau coefficient
             if  status == "lower":
                 # Take coefficient if nonbasic at lower bound
                 rowelem = binvarow[c]
             elif status == "upper":
                 # Flip coefficient if nonbasic at upper bound: x --> u - x
                 rowelem = -binvarow[c]
             else:
                 # variable is nonbasic free at zero -> cut coefficient is zero, skip OR
                 # variable is basic, skip
                 assert status == "zero" or status == "basic"
                 continue

             # Integer variables
             if col.isIntegral():
                 # warning: because of numerics cutelem < 0 is possible (though the fractional part is, mathematically, always positive)
                 # However, when cutelem < 0 it is also very close to 0, enough that isZero(cutelem) is true, so we ignore
                 # the coefficient (see below)
                 cutelem = scip.frac(rowelem)

                 if cutelem > f0:
                     #  sum((1-f_j)*f_0/(1 - f_0) x_j, j in J_I s.t. f_j  > f_0) +
                     cutelem = -((1.0 - cutelem) * ratiof0compl)
                 else:
                     #  sum(f_j x_j                  , j in J_I s.t. f_j <= f_0) +
                     cutelem = -cutelem
             else:
                 # Continuous variables
                 if rowelem < 0.0:
                     # -sum(a_j*f_0/(1-f_0) x_j      , j in J_C s.t. a_j  <   0) >= f_0.
                     cutelem = rowelem * ratiof0compl
                 else:
                     #  sum(a_j x_j,                 , j in J_C s.t. a_j >=   0) -
                     cutelem = -rowelem

             # cut is define when variables are in [0, infty). Translate to general bounds
             if not scip.isZero(cutelem):
                 if col.getBasisStatus() == "upper":
                     cutelem = -cutelem
                     cutrhs += cutelem * col.getUb()
                 else:
                     cutrhs += cutelem * col.getLb()
                 # Add coefficient to cut in dense form
                 cutcoefs[col.getLPPos()] = cutelem

        # Generate cut coefficients for the slack variables; skip basic ones
        for c in range(len(rows)):
            row = rows[c]
            assert row != None
            status = row.getBasisStatus()

            # free slack variable shouldn't appear
            assert status != "zero"

            # Get simplex tableau coefficient
            if status == "lower":
                # Take coefficient if nonbasic at lower bound
                rowelem = binvrow[row.getLPPos()]
                # But if this is a >= or ranged constraint at the lower bound, we have to flip the row element
                if not scip.isInfinity(-row.getLhs()):
                    rowelem = -rowelem
            elif status == "upper":
                # Take element if nonbasic at upper bound - see notes at beginning of file: only nonpositive slack variables
                # can be nonbasic at upper, therefore they should be flipped twice and we can take the element directly.
                rowelem = binvrow[row.getLPPos()]
            else:
                assert status == "basic"
                continue

            # if row is integral we can strengthen the cut coefficient
            if row.isIntegral() and not row.isModifiable():
                 # warning: because of numerics cutelem < 0 is possible (though the fractional part is, mathematically, always positive)
                 # However, when cutelem < 0 it is also very close to 0, enough that isZero(cutelem) is true (see later)
                 cutelem = scip.frac(rowelem)

                 if cutelem > f0:
                     #  sum((1-f_j)*f_0/(1 - f_0) x_j, j in J_I s.t. f_j  > f_0) +
                     cutelem = -((1.0 - cutelem) * ratiof0compl)
                 else:
                     #  sum(f_j x_j                  , j in J_I s.t. f_j <= f_0) +
                     cutelem = -cutelem
            else:
                # Continuous variables
                if rowelem < 0.0:
                    # -sum(a_j*f_0/(1-f_0) x_j      , j in J_C s.t. a_j  <   0) >= f_0.
                    cutelem = rowelem * ratiof0compl
                else:
                    #  sum(a_j x_j,                 , j in J_C s.t. a_j >=   0) -
                    cutelem = -rowelem

            # cut is define in original variables, so we replace slack by its definition
            if not scip.isZero(cutelem):
                # get lhs/rhs
                rlhs = row.getLhs()
                rrhs = row.getRhs()
                assert scip.isLE(rlhs, rrhs)
                assert not scip.isInfinity(rlhs) or not scip.isInfinity(rrhs)

                # If the slack variable is fixed, we can ignore this cut coefficient
                if scip.isFeasZero(rrhs - rlhs):
                  continue

                # Unflip slack variable and adjust rhs if necessary: row at lower means the slack variable is at its upper bound.
                # Since SCIP adds +1 slacks, this can only happen when constraints have a finite lhs
                if row.getBasisStatus() == "lower":
                    assert not scip.isInfinity(-rlhs)
                    cutelem = -cutelem

                rowcols = row.getCols()
                rowvals = row.getVals()

                assert len(rowcols) == len(rowvals)

                # Eliminate slack variable: rowcols is sorted: [columns in LP, columns not in LP]
                for i in range(row.getNLPNonz()):
                    cutcoefs[rowcols[i].getLPPos()] -= cutelem * rowvals[i]

                act = scip.getRowLPActivity(row)
                rhsslack = rrhs - act
                if scip.isFeasZero(rhsslack):
                    assert row.getBasisStatus() == "upper" # cutelem != 0 and row active at upper bound -> slack at lower, row at upper
                    cutrhs -= cutelem * (rrhs - row.getConstant())
                else:
                    assert scip.isFeasZero(act - rlhs)
                    cutrhs -= cutelem * (rlhs - row.getConstant())

        return cutcoefs, cutrhs

    def sepaexeclp(self):
        result = SCIP_RESULT.DIDNOTRUN
        scip = self.model

        if not scip.isLPSolBasic():
            return {"result": result}

        #TODO: add SCIPgetNLPBranchCands
        # get var data ---> this is the same as getVars!
        vars = scip.getVars(transformed = True)

        # get LP data
        cols = scip.getLPColsData()
        rows = scip.getLPRowsData()

        # exit if LP is trivial
        if len(cols) == 0 or len(rows) == 0:
            return {"result": result}

        result = SCIP_RESULT.DIDNOTFIND

        # get basis indices
        basisind = scip.getLPBasisInd()

        # For all basic columns (not slacks) belonging to integer variables, try to generate a gomory cut
        for i in range(len(rows)):
            tryrow = False
            c = basisind[i]
            #primsol = SCIP_INVALID

            #print("Row %d/%d basic index: %d"%(i, len(rows),c))
            #print("Row %d with value %f"%(i, cols[c].getPrimsol() if c >=0 else scip.getRowActivity(rows[-c-1])))
            if c >= 0:
                assert c < len(cols)
                var = cols[c].getVar()

                if var.vtype() != "CONTINUOUS":
                    primsol = cols[c].getPrimsol()
                    assert scip.getSolVal(None, var) == primsol

                    #print("var ", var," is not continuous. primsol = ", primsol)
                    if 0.005 <= scip.frac(primsol) <= 1 - 0.005:
                        #print("####trying gomory cut for col <%s> [%g] row %i"%(var, primsol, i))
                        tryrow = True

            # generate the cut!
            if tryrow:
                # get the row of B^-1 for this basic integer variable with fractional solution value
                binvrow = scip.getLPBInvRow(i)

                # get the tableau row for this basic integer variable with fractional solution value
                binvarow = scip.getLPBInvARow(i)

                # get cut's coefficients
                cutcoefs, cutrhs = self.getGMIFromRow(cols, rows, binvrow, binvarow, primsol)

                #########################################
                #### This code is for testing only!! ####
                # the first two cuts are -x1 + 2y <= 0 and -x2 +2y <= 0
                # the next two (when both previous ones are added) are -2x1 + 6y <=0 and -2x2 + 6y <=0
                assert scip.isZero(cutcoefs[0]) or scip.isZero(cutcoefs[1])
                if self.ncuts < 2:
                    assert scip.isZero(cutcoefs[0] + 1) or scip.isZero(cutcoefs[1] + 1)
                    assert scip.isZero(cutcoefs[2] - 2)
                else:
                    assert scip.isZero(cutcoefs[0] + 2) or scip.isZero(cutcoefs[1] + 2)
                    assert scip.isZero(cutcoefs[2] - 6)
                ##########################################

                # add cut
                # TODO: here it might make sense just to have a function `addCut` just like `addCons`. Or maybe better `createCut`
                # so that then one can ask stuff about it, like its efficacy, etc. This function would receive all coefficients
                # and basically do what we do here: cacheRowExtension etc up to releaseRow
                cut = scip.createEmptyRowSepa(self, "gmi%d_x%d"%(self.ncuts,c if c >= 0 else -c-1), lhs = None, rhs = cutrhs)
                scip.cacheRowExtensions(cut)

                for j in range(len(cutcoefs)):
                    if scip.isZero(cutcoefs[j]): # maybe here we need isFeasZero
                        continue
                    #print("var : ", cols[j].getVar(), " coef: ", cutcoefs[j])
                    #print("cut.lhs : ", cut.getLhs(), " cut.rhs: ", cut.getRhs())
                    scip.addVarToRow(cut, cols[j].getVar(), cutcoefs[j])

                if cut.getNNonz() == 0:
                    assert scip.isFeasNegative(cutrhs)
                    #print("Gomory cut is infeasible: 0 <= ", cutrhs)
                    return {"result": SCIP_RESULT.CUTOFF}


                # Only take efficacious cuts, except for cuts with one non-zero coefficient (= bound changes)
                # the latter cuts will be handeled internally in sepastore.
                if cut.getNNonz() == 1 or scip.isCutEfficacious(cut):
                    #print(" -> gomory cut for <%s>:  rhs=%f, eff=%f"%(cols[c].getVar() if c >= 0 else rows[-c-1].getName(),
                     #   cutrhs, scip.getCutEfficacy(cut)))

                    #SCIPdebugMessage(" -> found gomory cut <%s>: act=%f, rhs=%f, norm=%f, eff=%f, min=%f, max=%f (range=%f)\n",
                    #   cutname, SCIPgetRowLPActivity(scip, cut), SCIProwGetRhs(cut), SCIProwGetNorm(cut),
                    #   SCIPgetCutEfficacy(scip, NULL, cut),
                    #   SCIPgetRowMinCoef(scip, cut), SCIPgetRowMaxCoef(scip, cut),
                    #   SCIPgetRowMaxCoef(scip, cut)/SCIPgetRowMinCoef(scip, cut))

                    # flush all changes before adding the cut
                    scip.flushRowExtensions(cut)

                    infeasible = scip.addCut(cut, forcecut=True)
                    self.ncuts += 1

                    if infeasible:
                       result = SCIP_RESULT.CUTOFF
                    else:
                       result = SCIP_RESULT.SEPARATED
                scip.releaseRow(cut)

        return {"result": result}

def model():
    # create solver instance
    s = Model()

    # include separator
    sepa = GMI()
    s.includeSepa(sepa, "python_gmi", "generates gomory mixed integer cuts", priority = 1000, freq = 1)

    # turn off presolve
    s.setPresolve(SCIP_PARAMSETTING.OFF)
    # turn off heuristics
    s.setHeuristics(SCIP_PARAMSETTING.OFF)
    # turn off propagation
    s.setIntParam("propagating/maxrounds", 0)
    s.setIntParam("propagating/maxroundsroot", 0)
    # turn off some cuts
    s.setIntParam("separating/strongcg/freq", -1)
    s.setIntParam("separating/gomory/freq", -1)
    s.setIntParam("separating/aggregation/freq", -1)
    s.setIntParam("separating/mcf/freq", -1)
    s.setIntParam("separating/closecuts/freq", -1)
    s.setIntParam("separating/clique/freq", -1)
    s.setIntParam("separating/zerohalf/freq", -1)

    # only two rounds of cuts
    s.setIntParam("separating/maxroundsroot", 2)

    return s

# we use Cook Kannan and Schrijver's example
def test_CKS():
    s = model()
    # add variables
    x1 = s.addVar("x1", vtype='I')
    x2 = s.addVar("x2", vtype='I')
    y = s.addVar("y", obj=-1, vtype='C')

    # add constraint
    s.addCons(-x1      + y <=  0)
    s.addCons(    - x2 + y <=  0)
    s.addCons( x1 + x2 + y <=  2)

    # solve problem
    s.optimize()
    s.printStatistics()
