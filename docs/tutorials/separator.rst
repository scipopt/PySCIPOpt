##########################
Seperator (Cutting Planes)
##########################

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, Sepa, SCIP_RESULT

  scip = Model()

.. contents:: Contents


What is a Separator?
=====================

A separator is an algorithm for generating cutting planes (often abbreviated as cuts).
A cut is an inequality that does not remove any feasible solutions of the optimization problem but is intended
to remove some fractional solutions from the relaxation. For the purpose of this introduction we restrict ourselves
to linear cuts in this paper. A cut would then be denoted by a an array of coefficients
(:math:`\boldsymbol{\alpha} \in \mathbb{R}^{n}`) on each variable and a right-hand-side value
(:math:`\beta \in \mathbb{R}`).

.. math::

  \boldsymbol{\alpha}^{T}\mathbf{x} \leq \beta

The purpose of a separator is to find cuts that tighten the relaxation of an optimization problem.
Most commonly, cuts are found that separate the current fractional feasible solution to the relaxation,
thereby making a tighter relaxation. For this reason algorithms that find cuts are often called separators.


Gomory Mixed-Integer Cut Example
================================

In this example we show to generate one of the most prolific class of cutting planes, the
Gomory mixed-integer (GMI) cut. This example looks complicated, and that's simply because it is.
While in theory a GMI cut can be quickly written down, that assumes that your basis information is nicely
accessible in the form that you want and that your problem is in standard form. For this reason the code
needs to be quite substantial to construct the cuts from scratch.

.. note:: Separators are one of the most difficult components of a MIP solver to program due to their correctness
  being difficult to verify, and numerics being a constant issue.

.. code-block:: python

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

          # Generate cut coefficients for the original variables
          for c in range(len(cols)):
              col = cols[c]
              assert col is not None # is this the equivalent of col != NULL? does it even make sense to have this assert?
              status = col.getBasisStatus()

              # Get simplex tableau coefficient
              if status == "lower":
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
                      # sum((1-f_j)*f_0/(1 - f_0) x_j, j in J_I s.t. f_j  > f_0) +
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

              if c >= 0:
                  assert c < len(cols)
                  var = cols[c].getVar()

                  if var.vtype() != "CONTINUOUS":
                      primsol = cols[c].getPrimsol()
                      assert scip.getSolVal(None, var) == primsol

                      if 0.005 <= scip.frac(primsol) <= 1 - 0.005:
                          tryrow = True

              # generate the cut!
              if tryrow:
                  # get the row of B^-1 for this basic integer variable with fractional solution value
                  binvrow = scip.getLPBInvRow(i)

                  # get the tableau row for this basic integer variable with fractional solution value
                  binvarow = scip.getLPBInvARow(i)

                  # get cut's coefficients
                  cutcoefs, cutrhs = self.getGMIFromRow(cols, rows, binvrow, binvarow, primsol)

                  # add cut
                  cut = scip.createEmptyRowSepa(self, "gmi%d_x%d"%(self.ncuts,c if c >= 0 else -c-1), lhs = None, rhs = cutrhs)
                  scip.cacheRowExtensions(cut)

                  for j in range(len(cutcoefs)):
                      if scip.isZero(cutcoefs[j]): # maybe here we need isFeasZero
                          continue
                      scip.addVarToRow(cut, cols[j].getVar(), cutcoefs[j])

                  if cut.getNNonz() == 0:
                      assert scip.isFeasNegative(cutrhs)
                      return {"result": SCIP_RESULT.CUTOFF}


                  # Only take efficacious cuts, except for cuts with one non-zero coefficient (= bound changes)
                  # the latter cuts will be handeled internally in sepastore.
                  if cut.getNNonz() == 1 or scip.isCutEfficacious(cut):

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

The GMI separator can then be included using the following code:

.. code-block:: python

  sepa = GMI()
  scip.includeSepa(sepa, "python_gmi", "generates gomory mixed integer cuts", priorityS=1000, freq=1)

