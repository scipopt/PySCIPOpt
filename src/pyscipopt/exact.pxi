def _asRational(value):
    """Convert value to a Fraction, keeping +/- infinity as a float."""
    try:
        return Fraction(value)
    except OverflowError:
        return float(value)

def _asFiniteRational(value):
    value = _asRational(value)
    if isinstance(value, float):
        raise ValueError("value must be finite, got %s" % value)
    return value

cdef SCIP_RATIONAL* _createRational(value) except NULL:
    """Create a SCIP_RATIONAL holding value; the caller must free it with SCIPrationalFree."""
    cdef SCIP_RATIONAL* rational = NULL

    value = _asRational(value)
    if isinstance(value, float):
        desc = b"inf" if value > 0 else b"-inf"
    else:
        desc = str(value).encode()

    PY_SCIP_CALL(SCIPrationalCreate(&rational))
    SCIPrationalSetString(rational, desc)
    return rational

cdef object _fromRational(SCIP_RATIONAL* rational):
    """Convert a SCIP_RATIONAL to a Fraction, or to a float for +/- infinity."""
    cdef int length = SCIPrationalStrLen(rational)
    cdef char* buf = <char*> malloc(length + 1)

    if buf == NULL:
        raise MemoryError()
    SCIPrationalToString(rational, buf, length + 1)
    desc = buf[:length].decode("ascii")
    free(buf)

    if desc == "+infinity":
        return float("inf")
    if desc == "-infinity":
        return -float("inf")
    return Fraction(desc)

cdef _checkExactSol(Solution sol):
    if sol.sol != NULL and not SCIPsolIsExact(sol.sol):
        raise ValueError("solution has no exact values")

cdef class ColumnExact:
    """Base class holding a pointer to corresponding SCIP_COLEXACT."""

    @staticmethod
    cdef create(SCIP_COLEXACT* scipcolexact):
        """
        Main method for creating a ColumnExact class. Is used instead of __init__.

        Parameters
        ----------
        scipcolexact : SCIP_COLEXACT*
            A pointer to the SCIP_COLEXACT

        Returns
        -------
        col : ColumnExact
            The Python representative of the SCIP_COLEXACT

        """
        if scipcolexact == NULL:
            raise Warning("cannot create ColumnExact with SCIP_COLEXACT* == NULL")
        col = ColumnExact()
        col.scip_col_exact = scipcolexact
        return col

cdef class RowExact:
    """Base class holding a pointer to corresponding SCIP_ROW."""

    @staticmethod
    cdef create(SCIP_ROWEXACT* sciprowexact):
        """
        Main method for creating a RowExact class. Is used instead of __init__.

        Parameters
        ----------
        sciprow : SCIP_ROWEXACT*
            A pointer to the SCIP_ROWEXACT

        Returns
        -------
        row : Row
            The Python representative of the SCIP_ROWEXACT

        """
        if sciprowexact == NULL:
            raise Warning("cannot create Row with SCIP_ROWEXACT* == NULL")
        row_exact = RowExact()
        row_exact.scip_row_exact = sciprowexact
        return row_exact


cdef class ExactVariable(Variable):
    """Variable of an ExactModel: bounds, objective and LP value are rationals."""

    def getObj(self):
        """
        Retrieve the exact objective coefficient of the variable.

        Returns
        -------
        Fraction

        """
        return _fromRational(SCIPvarGetObjExact(self.scip_var))

    def getLbOriginal(self):
        """
        Retrieve the exact original lower bound of the variable.

        Returns
        -------
        Fraction or float
            a Fraction, or -inf if the variable is unbounded

        """
        return _fromRational(SCIPvarGetLbOriginalExact(self.scip_var))

    def getUbOriginal(self):
        """
        Retrieve the exact original upper bound of the variable.

        Returns
        -------
        Fraction or float
            a Fraction, or inf if the variable is unbounded

        """
        return _fromRational(SCIPvarGetUbOriginalExact(self.scip_var))

    def getLbGlobal(self):
        """
        Retrieve the exact global lower bound of the variable.

        Returns
        -------
        Fraction or float
            a Fraction, or -inf if the variable is unbounded

        """
        return _fromRational(SCIPvarGetLbGlobalExact(self.scip_var))

    def getUbGlobal(self):
        """
        Retrieve the exact global upper bound of the variable.

        Returns
        -------
        Fraction or float
            a Fraction, or inf if the variable is unbounded

        """
        return _fromRational(SCIPvarGetUbGlobalExact(self.scip_var))

    def getLbLocal(self):
        """
        Retrieve the exact local lower bound of the variable.

        Returns
        -------
        Fraction or float
            a Fraction, or -inf if the variable is unbounded

        """
        return _fromRational(SCIPvarGetLbLocalExact(self.scip_var))

    def getUbLocal(self):
        """
        Retrieve the exact local upper bound of the variable.

        Returns
        -------
        Fraction or float
            a Fraction, or inf if the variable is unbounded

        """
        return _fromRational(SCIPvarGetUbLocalExact(self.scip_var))

    def getLPSol(self):
        """
        Returns the current LP solution value of the variable as a Fraction.

        Returns
        -------
        Fraction

        """
        cdef SCIP_RATIONAL* rational = NULL

        PY_SCIP_CALL(SCIPrationalCreate(&rational))
        SCIPvarGetLPSolExact(self.scip_var, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getCol(self):
        """
        Returns the exact LP column of the variable.

        Returns
        -------
        ColumnExact

        """
        return ColumnExact.create(SCIPvarGetColExact(self.scip_var))


cdef class ExactModel(Model):
    """
    Model that solves exactly over the rationals.

    Variables, constraints and solutions carry exact data. Methods accept rational
    inputs as int, Fraction, str (e.g. "1/3") or float and return Fractions, or
    +/- float("inf") for infinite values.
    """

    def __init__(self, problemName='model', defaultPlugins=True, createscip=True, threadsafe=False):
        """
        Parameters
        ----------
        problemName : str, optional
            name of the problem (default 'model')
        defaultPlugins : bool, optional
            use default plugins (default True)
        createscip : bool, optional
            initialize the Model object and creates a SCIP instance
        threadsafe : bool, optional
            False if data can be safely shared between the source and target problem

        """
        if not PYSCIPOPT_WITH_EXACTSOLVE:
            raise ValueError("SCIP was built without exact solving support, ExactModel needs a SCIP build with exact solving enabled")
        super().__init__(problemName, defaultPlugins, createscip=createscip, threadsafe=threadsafe)

    cdef _configureScip(self):
        PY_SCIP_CALL(SCIPenableExactSolving(self._scip, True))

    def setObjective(self, expr, sense='minimize', clear='true'):
        """
        Not available in exact solving mode: expressions hold floating-point coefficients.
        Set objective coefficients with addVar(obj=...) or chgVarObj() instead.
        """
        raise TypeError("ExactModel.setObjective is not available because expressions hold floating-point coefficients; set objective coefficients with addVar(obj=...) or chgVarObj()")

    def addVar(self, name='', vtype='C', lb=0, ub=None, obj=0, pricedVar=False, pricedVarScore=1.0, deletable=False):
        """
        Create a new variable with exact (rational) bounds and objective coefficient.

        Parameters
        ----------
        name : str, optional
            name of the variable, generic if empty (Default value = '')
        vtype : str, optional
            type of the variable: 'C' continuous, 'I' integer, or 'B' binary
            (Default value = 'C')
        lb : Fraction, int, str or None, optional
            lower bound of the variable, use None for -infinity (Default value = 0)
        ub : Fraction, int, str or None, optional
            upper bound of the variable, use None for +infinity, or 1 for binary variables
            (Default value = None)
        obj : Fraction, int or str, optional
            objective coefficient of the variable (Default value = 0)

        Returns
        -------
        Variable

        Notes
        -----
        Values are converted with fractions.Fraction, so "1/3" and Fraction(1, 3) are exact.
        Floats are taken at their exact binary value, e.g. 0.1 is not 1/10.

        """
        cdef SCIP_VAR* scip_var = NULL
        cdef SCIP_RATIONAL* lbexact = NULL
        cdef SCIP_RATIONAL* ubexact = NULL
        cdef SCIP_RATIONAL* objexact = NULL
        cdef SCIP_VARTYPE vartype
        cdef SCIP_RETCODE rc

        if pricedVar or deletable:
            raise ValueError("priced and deletable variables are not supported in exact solving mode")
        if name == '':
            name = 'x'+str(SCIPgetNVars(self._scip)+1)
        cname = str_conversion(name)

        vtype = vtype.upper()
        if vtype in ['C', 'CONTINUOUS']:
            vartype = SCIP_VARTYPE_CONTINUOUS
        elif vtype in ['B', 'BINARY']:
            vartype = SCIP_VARTYPE_BINARY
            if ub is None:
                ub = 1
        elif vtype in ['I', 'INTEGER']:
            vartype = SCIP_VARTYPE_INTEGER
        else:
            raise Warning("unrecognized variable type")

        lb = _asRational(-float("inf") if lb is None else lb)
        ub = _asRational(float("inf") if ub is None else ub)
        obj = _asFiniteRational(obj)
        if lb > ub:
            raise ValueError("lower bound %s exceeds upper bound %s" % (lb, ub))

        PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname,
                                        max(float(lb), -SCIPinfinity(self._scip)),
                                        min(float(ub), SCIPinfinity(self._scip)),
                                        float(obj), vartype))
        lbexact = _createRational(lb)
        ubexact = _createRational(ub)
        objexact = _createRational(obj)
        rc = SCIPaddVarExactData(self._scip, scip_var, lbexact, ubexact, objexact)
        SCIPrationalFree(&lbexact)
        SCIPrationalFree(&ubexact)
        SCIPrationalFree(&objexact)
        if rc != SCIP_OKAY:
            SCIPreleaseVar(self._scip, &scip_var)
        PY_SCIP_CALL(rc)
        PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

        pyVar = self._getOrCreateVar(scip_var)
        SCIPvarSetData(scip_var, <SCIP_VARDATA*>pyVar)
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &scip_var))
        return pyVar

    def chgVarObj(self, Variable var, newobj):
        """
        Change the exact objective coefficient of a variable.

        Parameters
        ----------
        var : Variable
            variable to change the objective coefficient of
        newobj : Fraction, int or str
            new objective coefficient

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        rational = _createRational(_asFiniteRational(newobj))
        rc = SCIPchgVarObjExact(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgVarLb(self, Variable var, lb):
        """
        Change the exact lower bound of a variable.

        Parameters
        ----------
        var : Variable
            variable to change the bound of
        lb : Fraction, int, str or None
            new lower bound, use None for -infinity

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        rational = _createRational(-float("inf") if lb is None else lb)
        rc = SCIPchgVarLbExact(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgVarUb(self, Variable var, ub):
        """
        Change the exact upper bound of a variable.

        Parameters
        ----------
        var : Variable
            variable to change the bound of
        ub : Fraction, int, str or None
            new upper bound, use None for +infinity

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        rational = _createRational(float("inf") if ub is None else ub)
        rc = SCIPchgVarUbExact(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgVarLbGlobal(self, Variable var, lb):
        """
        Changes the lower bound of the variable exactly.

        Parameters
        ----------
        var : Variable
        lb : Rational
            new bound (None for -infinity)

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc

        rational = _createRational(-float("inf") if lb is None else lb)
        rc = SCIPchgVarLbGlobalExact(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgVarUbGlobal(self, Variable var, ub):
        """
        Changes the upper bound of the variable exactly.

        Parameters
        ----------
        var : Variable
        ub : Rational
            new bound (None for +infinity)

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc

        rational = _createRational(float("inf") if ub is None else ub)
        rc = SCIPchgVarUbGlobalExact(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def fixVar(self, Variable var, val):
        """
        Fixes the variable to a rational value in the current node.

        Parameters
        ----------
        var : Variable
        val : Rational

        Returns
        -------
        infeasible : bool
        fixed : bool

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool fixed

        rational = _createRational(_asFiniteRational(val))
        rc = SCIPfixVarExact(self._scip, var.scip_var, rational, &infeasible, &fixed)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)
        return infeasible, fixed

    def tightenVarLb(self, Variable var, lb, force=False):
        """
        Tightens the lower bound of the variable exactly in the current node.

        Parameters
        ----------
        var : Variable
        lb : Rational
        force : bool, optional
            ignored; the exact SCIP function has no force parameter

        Returns
        -------
        infeasible : bool
        tightened : bool

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened

        rational = _createRational(_asFiniteRational(lb))
        rc = SCIPtightenVarLbExact(self._scip, var.scip_var, rational, &infeasible, &tightened)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)
        return infeasible, tightened

    def tightenVarUb(self, Variable var, ub, force=False):
        """
        Tightens the upper bound of the variable exactly in the current node.

        Parameters
        ----------
        var : Variable
        ub : Rational
        force : bool, optional
            ignored; the exact SCIP function has no force parameter

        Returns
        -------
        infeasible : bool
        tightened : bool

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened

        rational = _createRational(_asFiniteRational(ub))
        rc = SCIPtightenVarUbExact(self._scip, var.scip_var, rational, &infeasible, &tightened)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)
        return infeasible, tightened

    def aggregateVars(self, Variable varx, Variable vary, coefx=1, coefy=-1, rhs=0):
        """
        Aggregates two variables exactly: coefx * varx + coefy * vary == rhs.

        Returns
        -------
        infeasible : bool
        redundant : bool
        aggregated : bool

        """
        cdef SCIP_RATIONAL* scalarx
        cdef SCIP_RATIONAL* scalary
        cdef SCIP_RATIONAL* rhsexact
        cdef SCIP_RETCODE rc
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool redundant
        cdef SCIP_Bool aggregated

        scalarx = _createRational(_asFiniteRational(coefx))
        scalary = _createRational(_asFiniteRational(coefy))
        rhsexact = _createRational(_asFiniteRational(rhs))
        rc = SCIPaggregateVarsExact(self._scip, varx.scip_var, vary.scip_var, scalarx, scalary, rhsexact, &infeasible, &redundant, &aggregated)
        SCIPrationalFree(&scalarx)
        SCIPrationalFree(&scalary)
        SCIPrationalFree(&rhsexact)
        PY_SCIP_CALL(rc)
        return infeasible, redundant, aggregated

    def startDive(self):
        """Initiates exact LP diving."""
        PY_SCIP_CALL(SCIPstartExactDive(self._scip))

    def endDive(self):
        """Quits exact LP diving."""
        PY_SCIP_CALL(SCIPendExactDive(self._scip))

    def solveDiveLP(self, itlim=-1):
        """
        Solves the exact diving LP.

        Returns
        -------
        lperror : bool
        cutoff : bool

        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL(SCIPsolveExactDiveLP(self._scip, itlim, &lperror, &cutoff))
        return lperror, cutoff

    def chgVarLbDive(self, Variable var, newbound):
        """Changes the lower bound of the variable in the exact dive."""
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc

        rational = _createRational(-float("inf") if newbound is None else newbound)
        rc = SCIPchgVarLbExactDive(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgVarUbDive(self, Variable var, newbound):
        """Changes the upper bound of the variable in the exact dive."""
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc

        rational = _createRational(float("inf") if newbound is None else newbound)
        rc = SCIPchgVarUbExactDive(self._scip, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def tightenVarLbGlobal(self, var, lb, force=False):
        """Not available in exact solving mode."""
        raise ValueError("tightenVarLbGlobal is not available in exact solving mode")

    def tightenVarUbGlobal(self, var, ub, force=False):
        """Not available in exact solving mode."""
        raise ValueError("tightenVarUbGlobal is not available in exact solving mode")

    def chgVarLbNode(self, node, var, lb):
        """Not available in exact solving mode."""
        raise ValueError("chgVarLbNode is not available in exact solving mode")

    def chgVarUbNode(self, node, var, ub):
        """Not available in exact solving mode."""
        raise ValueError("chgVarUbNode is not available in exact solving mode")

    def chgVarObjDive(self, var, newobj):
        """Not available in exact solving mode."""
        raise ValueError("chgVarObjDive is not available in exact solving mode")

    def chgVarLbProbing(self, var, lb):
        """Not available in exact solving mode."""
        raise ValueError("chgVarLbProbing is not available in exact solving mode")

    def chgVarUbProbing(self, var, ub):
        """Not available in exact solving mode."""
        raise ValueError("chgVarUbProbing is not available in exact solving mode")

    def chgVarObjProbing(self, var, newobj):
        """Not available in exact solving mode."""
        raise ValueError("chgVarObjProbing is not available in exact solving mode")

    def fixVarProbing(self, var, fixedval):
        """Not available in exact solving mode."""
        raise ValueError("fixVarProbing is not available in exact solving mode")
