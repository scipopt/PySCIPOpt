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

cdef _checkExactVar(Variable var, original=False):
    if not SCIPvarIsExact(var.scip_var):
        raise ValueError("variable <%s> has no exact data" % var.name)
    if original and not SCIPvarIsOriginal(var.scip_var):
        raise ValueError("variable <%s> is not an original variable" % var.name)

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
