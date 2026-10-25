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

    def addConsLinear(self, vars, vals, lhs, rhs, name="",
                           initial=True, separate=True, enforce=True, check=True,
                           propagate=True, local=False, modifiable=False, dynamic=False,
                           removable=False, stickingatnode=False):
        """
        Add an exact linear constraint lhs <= sum(vals[i] * vars[i]) <= rhs.

        Parameters
        ----------
        vars : list of Variable
            variables of the constraint
        vals : list of Fraction, int or str
            coefficients of the variables
        lhs : Fraction, int, str or None
            left-hand side, use None for -infinity
        rhs : Fraction, int, str or None
            right-hand side, use None for +infinity
        name : str, optional
            name of the constraint, generic if empty (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint

        """
        cdef _VarArray wrapper = _VarArray(vars)
        cdef int nvars = wrapper.size
        cdef SCIP_RATIONAL** valsexact = NULL
        cdef SCIP_RATIONAL* lhsexact = NULL
        cdef SCIP_RATIONAL* rhsexact = NULL
        cdef SCIP_CONS* scip_cons = NULL
        cdef SCIP_RETCODE rc
        cdef int i

        if len(vals) != nvars:
            raise ValueError("number of variables (%i) and coefficients (%i) must be the same" % (nvars, len(vals)))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        vals = [_asFiniteRational(val) for val in vals]
        lhs = _asRational(-float("inf") if lhs is None else lhs)
        rhs = _asRational(float("inf") if rhs is None else rhs)

        valsexact = <SCIP_RATIONAL**> malloc(max(nvars, 1) * sizeof(SCIP_RATIONAL*))
        if valsexact == NULL:
            raise MemoryError()
        for i in range(nvars):
            valsexact[i] = _createRational(vals[i])
        lhsexact = _createRational(lhs)
        rhsexact = _createRational(rhs)

        rc = SCIPcreateConsExactLinear(
            self._scip, &scip_cons, str_conversion(name), nvars, wrapper.ptr, valsexact,
            lhsexact, rhsexact, initial, separate, enforce, check, propagate, local, modifiable,
            dynamic, removable, stickingatnode)
        for i in range(nvars):
            SCIPrationalFree(&valsexact[i])
        free(valsexact)
        SCIPrationalFree(&lhsexact)
        SCIPrationalFree(&rhsexact)
        PY_SCIP_CALL(rc)

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        return pyCons

    def _checkExactLinear(self, Constraint cons):
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype != 'exactlinear':
            raise ValueError("expected a constraint of type exactlinear, got %s" % constype)

    def addCoefLinear(self, Constraint cons, Variable var, value):
        """
        Add a variable with an exact coefficient to an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint
        var : Variable
            variable to add
        value : Fraction, int or str
            coefficient of the variable

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        self._checkExactLinear(cons)
        rational = _createRational(_asFiniteRational(value))
        rc = SCIPaddCoefExactLinear(self._scip, cons.scip_cons, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgCoefLinear(self, Constraint cons, Variable var, value):
        """
        Change the exact coefficient of a variable in an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint
        var : Variable
            variable whose coefficient is changed
        value : Fraction, int or str
            new coefficient of the variable

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        self._checkExactLinear(cons)
        rational = _createRational(_asFiniteRational(value))
        rc = SCIPchgCoefExactLinear(self._scip, cons.scip_cons, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgLhs(self, Constraint cons, lhs):
        """
        Change the exact left-hand side of an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint
        lhs : Fraction, int, str or None
            new left-hand side, use None for -infinity

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        self._checkExactLinear(cons)
        rational = _createRational(-float("inf") if lhs is None else lhs)
        rc = SCIPchgLhsExactLinear(self._scip, cons.scip_cons, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def chgRhs(self, Constraint cons, rhs):
        """
        Change the exact right-hand side of an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint
        rhs : Fraction, int, str or None
            new right-hand side, use None for +infinity

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        self._checkExactLinear(cons)
        rational = _createRational(float("inf") if rhs is None else rhs)
        rc = SCIPchgRhsExactLinear(self._scip, cons.scip_cons, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def getLhs(self, Constraint cons):
        """
        Retrieve the exact left-hand side of an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint

        Returns
        -------
        Fraction or float
            a Fraction, or -inf if the constraint has no left-hand side

        """
        self._checkExactLinear(cons)
        return _fromRational(SCIPgetLhsExactLinear(self._scip, cons.scip_cons))

    def getRhs(self, Constraint cons):
        """
        Retrieve the exact right-hand side of an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint

        Returns
        -------
        Fraction or float
            a Fraction, or inf if the constraint has no right-hand side

        """
        self._checkExactLinear(cons)
        return _fromRational(SCIPgetRhsExactLinear(self._scip, cons.scip_cons))

    def getValsLinear(self, Constraint cons):
        """
        Retrieve the exact coefficients of an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint

        Returns
        -------
        dict of str to Fraction

        """
        cdef SCIP_VAR** vars
        cdef SCIP_RATIONAL** vals
        cdef int i

        self._checkExactLinear(cons)
        vars = SCIPgetVarsExactLinear(self._scip, cons.scip_cons)
        vals = SCIPgetValsExactLinear(self._scip, cons.scip_cons)

        valsdict = {}
        for i in range(SCIPgetNVarsExactLinear(self._scip, cons.scip_cons)):
            valsdict[bytes(SCIPvarGetName(vars[i])).decode('utf-8')] = _fromRational(vals[i])

        return valsdict

    def getActivity(self, Constraint cons, Solution sol = None):
        """
        Retrieve the exact activity of an exact linear constraint in a given solution.

        Parameters
        ----------
        cons : Constraint
            exact linear constraint
        sol : Solution or None, optional
            solution to compute the activity in, or None for the current LP/pseudo solution
            (Default value = None)

        Returns
        -------
        Fraction

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc
        cdef SCIP_SOL* scip_sol = NULL

        self._checkExactLinear(cons)
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        if sol is not None:
            _checkExactSol(sol)
            scip_sol = sol.sol

        rational = _createRational(0)
        rc = SCIPgetActivityExactLinear(self._scip, cons.scip_cons, scip_sol, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)
        return result

    def addObjoffset(self, offset, solutions=False):
        """
        Add an exact constant to the objective function of the original problem.

        Parameters
        ----------
        offset : Fraction, int or str
            offset to add

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc

        if solutions:
            raise ValueError("addObjoffset(solutions=True) is not available in exact solving mode")
        if SCIPgetStage(self._scip) != SCIP_STAGE_PROBLEM:
            raise Warning("addObjoffset can only be called in stage PROBLEM in exact solving mode")
        rational = _createRational(_asFiniteRational(offset))
        rc = SCIPaddOrigObjoffsetExact(self._scip, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def getObjoffset(self, original=True):
        """
        Retrieve the exact objective offset of the original problem.

        Returns
        -------
        Fraction

        """
        if not original:
            raise ValueError("only the original objective offset is available in exact solving mode")
        if not SCIP_STAGE_PROBLEM <= SCIPgetStage(self._scip) <= SCIP_STAGE_SOLVED:
            raise Warning("getObjoffset cannot be called in stage %i." % SCIPgetStage(self._scip))
        return _fromRational(SCIPgetOrigObjoffsetExact(self._scip))

    def addCons(self, cons, name='', **kwargs):
        """
        Not available in exact solving mode: expressions hold floating-point coefficients.
        Use addConsLinear(vars, vals, lhs, rhs) instead.
        """
        raise TypeError("ExactModel.addCons is not available because expressions hold floating-point coefficients; use addConsLinear(vars, vals, lhs, rhs)")

    def delCoefLinear(self, Constraint cons, Variable var):
        """
        Deletes a variable from an exact linear constraint.

        Parameters
        ----------
        cons : Constraint
        var : Variable

        """
        self._checkExactLinear(cons)
        PY_SCIP_CALL(SCIPdelCoefExactLinear(self._scip, cons.scip_cons, var.scip_var))

    def getSlack(self, Constraint cons, Solution sol = None, side = None):
        """
        Retrieve the slack of an exact linear constraint in the given solution.

        Parameters
        ----------
        cons : Constraint
        sol : Solution or None, optional
            solution to use, None to use the current node's solution (Default value = None)
        side : str or None, optional
            whether to use 'lhs' or 'rhs' for ranged constraints, None to return minimum (Default value = None)

        Returns
        -------
        Fraction or float

        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        lhs = self.getLhs(cons)
        rhs = self.getRhs(cons)
        activity = self.getActivity(cons, sol)
        lhsslack = activity - lhs
        rhsslack = rhs - activity
        if side == 'lhs':
            return lhsslack
        elif side == 'rhs':
            return rhsslack
        else:
            return min(lhsslack, rhsslack)

    def getRowLinear(self, Constraint cons):
        """
        Retrieve the floating-point LP row of an exact linear constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        Row

        """
        self._checkExactLinear(cons)
        return Row.create(SCIPgetRowExactLinear(self._scip, cons.scip_cons))

    def getDualsolLinear(self, Constraint cons):
        """
        Retrieve the dual solution of an exact linear constraint as a Fraction.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        Fraction

        """
        cdef SCIP_RATIONAL* rational = NULL

        self._checkExactLinear(cons)
        transcons = <Constraint>self.getTransformedCons(cons) if cons.isOriginal() else cons
        PY_SCIP_CALL(SCIPrationalCreate(&rational))
        SCIPgetFpDualsolExactLinear(self._scip, transcons.scip_cons, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getDualfarkasLinear(self, Constraint cons):
        """
        Retrieve the dual Farkas value of an exact linear constraint as a Fraction.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        Fraction

        """
        cdef SCIP_RATIONAL* rational = NULL

        self._checkExactLinear(cons)
        transcons = <Constraint>self.getTransformedCons(cons) if cons.isOriginal() else cons
        PY_SCIP_CALL(SCIPrationalCreate(&rational))
        SCIPgetFpDualfarkasExactLinear(self._scip, transcons.scip_cons, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getConsVars(self, Constraint constraint):
        """
        Gets variables in constraint; exact linear constraints are handled exactly.

        Parameters
        ----------
        constraint : Constraint

        Returns
        -------
        list of Variable

        """
        cdef SCIP_VAR** vars
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(constraint.scip_cons))).decode('UTF-8')
        if constype != 'exactlinear':
            return super().getConsVars(constraint)
        vars = SCIPgetVarsExactLinear(self._scip, constraint.scip_cons)
        return [self._getOrCreateVar(vars[i]) for i in range(SCIPgetNVarsExactLinear(self._scip, constraint.scip_cons))]

    def getConsVals(self, Constraint constraint):
        """
        Gets coefficients in constraint; exact linear constraints return Fractions.

        Parameters
        ----------
        constraint : Constraint

        Returns
        -------
        list of Fraction

        """
        cdef SCIP_RATIONAL** vals
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(constraint.scip_cons))).decode('UTF-8')
        if constype != 'exactlinear':
            return super().getConsVals(constraint)
        vals = SCIPgetValsExactLinear(self._scip, constraint.scip_cons)
        return [_fromRational(vals[i]) for i in range(SCIPgetNVarsExactLinear(self._scip, constraint.scip_cons))]

    def _checkBoundStage(self, method):
        if not SCIP_STAGE_TRANSFORMED <= SCIPgetStage(self._scip) <= SCIP_STAGE_EXITSOLVE:
            raise Warning("%s cannot be called in stage %i." % (method, SCIPgetStage(self._scip)))

    def createSol(self, Heur heur = None, initlp=False):
        """
        Create a new exact primal solution in the transformed space.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)
        initlp : bool, optional
            Should the created solution be initialised to the current LP solution instead of all zeros

        Returns
        -------
        Solution

        """
        cdef SCIP_HEUR* _heur
        cdef SCIP_SOL* _sol

        if isinstance(heur, Heur):
            n = str_conversion(heur.name)
            _heur = SCIPfindHeur(self._scip, n)
        else:
            _heur = NULL
        if not initlp:
            PY_SCIP_CALL(SCIPcreateSolExact(self._scip, &_sol, _heur))
        else:
            PY_SCIP_CALL(SCIPcreateLPSolExact(self._scip, &_sol, _heur))
        return Solution.create(self._scip, _sol)

    def createOrigSol(self, Heur heur = None):
        """Not available in exact solving mode."""
        raise ValueError("createOrigSol is not available in exact solving mode")

    def createPartialSol(self, Heur heur = None):
        """Not available in exact solving mode."""
        raise ValueError("createPartialSol is not available in exact solving mode")

    def setSolVal(self, Solution solution, Variable var, val):
        """
        Set a variable in a solution to a rational value.

        Parameters
        ----------
        solution : Solution
        var : Variable
        val : Rational

        """
        cdef SCIP_RATIONAL* rational
        cdef SCIP_RETCODE rc

        assert solution.sol != NULL, "Cannot set value to a freed solution."
        rational = _createRational(_asFiniteRational(val))
        rc = SCIPsetSolValExact(self._scip, solution.sol, var.scip_var, rational)
        SCIPrationalFree(&rational)
        PY_SCIP_CALL(rc)

    def getSolVal(self, Solution sol, expr):
        """
        Retrieve the exact value of a variable in the given solution.

        Parameters
        ----------
        sol : Solution or None
            solution, None for the current LP/pseudo solution
        expr : Variable or MatrixVariable

        Returns
        -------
        Fraction or np.ndarray

        """
        cdef SCIP_RATIONAL* rational

        if sol is None:
            sol = Solution.create(self._scip, NULL)
        sol._checkStage("SCIPgetSolVal")
        _checkExactSol(sol)
        if isinstance(expr, MatrixVariable):
            values = np.empty(expr.shape, dtype=object)
            for idx in np.ndindex(expr.shape):
                values[idx] = self.getSolVal(sol, expr[idx])
            return values
        if not isinstance(expr, Variable):
            raise TypeError("exact solution values are only available for variables; evaluate expressions from the variable values")
        rational = _createRational(0)
        SCIPgetSolValExact(self._scip, sol.sol, (<Variable>expr).scip_var, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getVal(self, expr):
        """
        Retrieve the exact value of a variable in the best known solution.

        Parameters
        ----------
        expr : Variable or MatrixVariable

        Returns
        -------
        Fraction or np.ndarray

        """
        cdef SCIP_SOL* current_best_sol

        if SCIPgetStage(self._scip) in {SCIP_STAGE_INIT, SCIP_STAGE_FREE}:
            raise Warning("Method cannot be called in stage ", self.getStage())
        current_best_sol = SCIPgetBestSol(self._scip)
        if self._bestSol is None or self._bestSol.sol != current_best_sol:
            self._bestSol = Solution.create(self._scip, current_best_sol)
        if self._bestSol.sol == NULL and SCIPgetStage(self._scip) != SCIP_STAGE_SOLVING:
            raise Warning("No solution available")
        return self.getSolVal(self._bestSol, expr)

    def getSolObjVal(self, Solution sol, original=True):
        """
        Retrieve the exact objective value of the solution.

        Parameters
        ----------
        sol : Solution or None
        original : bool, optional
            objective value in original space; only True is supported, SCIP exports no exact
            transformed objective value (Default value = True)

        Returns
        -------
        Fraction

        """
        cdef SCIP_RATIONAL* rational

        if not original:
            raise ValueError("the transformed objective value is not available in exact solving mode")
        if sol is None:
            sol = Solution.create(self._scip, NULL)
        sol._checkStage("getSolObjVal")
        _checkExactSol(sol)
        rational = _createRational(0)
        SCIPgetSolOrigObjExact(self._scip, sol.sol, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def trySol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, free=True):
        """
        Check given exact primal solution for feasibility and try to add it to the storage.

        Returns
        -------
        stored : bool

        """
        cdef SCIP_Bool stored

        if SCIPgetStage(self._scip) < SCIP_STAGE_TRANSFORMED:
            raise Warning("trySol cannot be called before the problem is transformed")
        if free:
            PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        else:
            PY_SCIP_CALL(SCIPtrySol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        return stored

    def printSol(self, Solution solution=None, write_zeros=False):
        """
        Print the given exact primal solution.

        Parameters
        ----------
        solution : Solution or None, optional
            solution to print, None for the current LP/pseudo solution
        write_zeros : bool, optional
            include variables that are set to zero

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")
        if solution is None:
            PY_SCIP_CALL(SCIPprintSolExact(self._scip, NULL, NULL, write_zeros))
        else:
            PY_SCIP_CALL(SCIPprintSolExact(self._scip, solution.sol, NULL, write_zeros))
        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def printBestSol(self, write_zeros=False):
        """Print the best exact primal solution."""
        cdef SCIP_SOL* best = SCIPgetBestSol(self._scip)

        if best == NULL:
            return super().printBestSol(write_zeros)
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")
        PY_SCIP_CALL(SCIPprintSolExact(self._scip, best, NULL, write_zeros))
        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def writeSol(self, Solution solution, filename="origprob.sol", write_zeros=False):
        """Write the given exact primal solution to a file."""
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintSolExact(self._scip, solution.sol, cfile, write_zeros))
        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def writeBestSol(self, filename="origprob.sol", write_zeros=False):
        """Write the best exact primal solution to a file."""
        cdef SCIP_SOL* best = SCIPgetBestSol(self._scip)

        if best == NULL:
            return super().writeBestSol(filename, write_zeros)
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintSolExact(self._scip, best, cfile, write_zeros))
        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def getPrimalbound(self):
        """
        Retrieve the exact best primal bound (objective value of the best solution).

        Returns
        -------
        Fraction or float

        """
        cdef SCIP_RATIONAL* rational

        self._checkBoundStage("getPrimalbound")
        rational = _createRational(0)
        SCIPgetPrimalboundExact(self._scip, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getDualbound(self):
        """
        Retrieve the exact best dual bound.

        Returns
        -------
        Fraction or float

        """
        cdef SCIP_RATIONAL* rational

        self._checkBoundStage("getDualbound")
        rational = _createRational(0)
        SCIPgetDualboundExact(self._scip, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getUpperbound(self):
        """
        Retrieve the exact global upper (primal) bound in transformed space.

        Returns
        -------
        Fraction or float

        """
        cdef SCIP_RATIONAL* rational

        self._checkBoundStage("getUpperbound")
        rational = _createRational(0)
        SCIPgetUpperboundExact(self._scip, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getLowerbound(self):
        """
        Retrieve the exact global lower (dual) bound in transformed space.

        Returns
        -------
        Fraction or float

        """
        cdef SCIP_RATIONAL* rational

        self._checkBoundStage("getLowerbound")
        rational = _createRational(0)
        SCIPgetLowerboundExact(self._scip, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getLPObjVal(self):
        """
        Retrieve the objective value of the current exact LP.

        Returns
        -------
        Fraction or float

        """
        cdef SCIP_RATIONAL* rational

        self._checkBoundStage("getLPObjVal")
        rational = _createRational(0)
        SCIPgetLPExactObjval(self._scip, rational)
        result = _fromRational(rational)
        SCIPrationalFree(&rational)
        return result

    def getCutoffbound(self):
        """
        Retrieve the exact global cutoff bound.

        Returns
        -------
        Fraction or float

        """
        self._checkBoundStage("getCutoffbound")
        return _fromRational(SCIPgetCutoffboundExact(self._scip))
