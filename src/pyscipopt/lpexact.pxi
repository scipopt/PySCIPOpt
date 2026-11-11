##@file lpexact.pxi
#@brief LP solving in exact rational arithmetic through SCIP's exact LP interface

_INF = float("inf")


cdef int _setRational(SCIP_RATIONAL* rational, value) except -1:
    if isinstance(value, float) and abs(value) == _INF:
        SCIPrationalSetString(rational, b"inf" if value > 0 else b"-inf")
    else:
        SCIPrationalSetString(rational, str(Fraction(value)).encode())
    return 0


cdef SCIP_RATIONAL** _createRationals(int n) except NULL:
    cdef SCIP_RATIONAL** array = NULL

    PY_SCIP_CALL(SCIPrationalCreateArray(&array, n))
    if array == NULL:
        raise MemoryError()
    return array


cdef SCIP_RATIONAL** _createRationalArray(values) except NULL:
    cdef int n = len(values)
    cdef SCIP_RATIONAL** array = _createRationals(n)
    cdef int i

    try:
        for i in range(n):
            _setRational(array[i], values[i])
    except BaseException:
        SCIPrationalFreeArray(&array, n)
        raise
    return array


cdef void _freeRationals(SCIP_RATIONAL*** array, int n) noexcept:
    if array[0] != NULL:
        SCIPrationalFreeArray(array, n)


cdef list _fromRationals(SCIP_RATIONAL** array, int n):
    return [_fromRational(array[i]) for i in range(n)]


cdef int* _createInts(values) except NULL:
    cdef int n = len(values)
    cdef int* array = <int*> malloc(max(n, 1) * sizeof(int))
    cdef int i

    if array == NULL:
        raise MemoryError()
    try:
        for i in range(n):
            array[i] = values[i]
    except BaseException:
        free(array)
        raise
    return array


cdef list _getSolPart(SCIP_LPIEXACT* lpi, int n, int part):
    cdef SCIP_RATIONAL** vector = NULL

    if n == 0:
        return []
    try:
        vector = _createRationals(n)
        if part == 0:
            PY_SCIP_CALL(SCIPlpiExactGetSol(lpi, NULL, vector, NULL, NULL, NULL))
        elif part == 1:
            PY_SCIP_CALL(SCIPlpiExactGetSol(lpi, NULL, NULL, vector, NULL, NULL))
        elif part == 2:
            PY_SCIP_CALL(SCIPlpiExactGetSol(lpi, NULL, NULL, NULL, vector, NULL))
        else:
            PY_SCIP_CALL(SCIPlpiExactGetSol(lpi, NULL, NULL, NULL, NULL, vector))
        return _fromRationals(vector, n)
    finally:
        _freeRationals(&vector, n)


cdef class LPExact:
    """LP solved in exact rational arithmetic by SCIP's exact LP interface.

    Values are given as Fraction, int, float or str and returned as Fraction;
    infinite bounds and sides are float("inf") and -float("inf").
    """
    cdef SCIP_LPIEXACT* lpi
    cdef readonly str name

    def __init__(self, name="LP", sense="minimize"):
        """
        Parameters
        ----------
        name : str, optional
            name of the problem (Default value = "LP")
        sense : str, optional
            "minimize" or "maximize" (Default value = "minimize")

        """
        if SCIPlpiExactGetSolverName() == b"NONE":
            raise ValueError("SCIP was built without an exact LP solver, LPExact needs a SCIP build with exact SoPlex")
        self.name = name
        n = str_conversion(name)
        if sense == "minimize":
            PY_SCIP_CALL(SCIPlpiExactCreate(&self.lpi, NULL, n, SCIP_OBJSEN_MINIMIZE))
        elif sense == "maximize":
            PY_SCIP_CALL(SCIPlpiExactCreate(&self.lpi, NULL, n, SCIP_OBJSEN_MAXIMIZE))
        else:
            raise Warning("unrecognized objective sense")

    def __dealloc__(self):
        if self.lpi != NULL:
            PY_SCIP_CALL(SCIPlpiExactFree(&self.lpi))

    def __repr__(self):
        return self.name

    def writeLP(self, filename):
        """Write the LP to a file."""
        PY_SCIP_CALL(SCIPlpiExactWriteLP(self.lpi, str_conversion(filename)))

    def readLP(self, filename):
        """Read the LP from a file."""
        PY_SCIP_CALL(SCIPlpiExactReadLP(self.lpi, str_conversion(filename)))

    def infinity(self):
        """Return the value representing infinity: float("inf")."""
        return _INF

    def isInfinity(self, val):
        """Return whether val is positive infinity."""
        return val == _INF

    def addCol(self, entries, obj=0, lb=0, ub=None):
        """Add a single column.

        Parameters
        ----------
        entries : list of tuple
            (row index, coefficient) pairs of the nonzero entries of the column
        obj : Fraction or int or float or str, optional
            objective coefficient (Default value = 0)
        lb : Fraction or int or float or str or None, optional
            lower bound, None for -infinity (Default value = 0)
        ub : Fraction or int or float or str or None, optional
            upper bound, None for infinity (Default value = None)

        """
        self.addCols([entries], [obj], [lb], [ub])

    def addCols(self, entrieslist, objs=None, lbs=None, ubs=None):
        """Add multiple columns.

        Parameters
        ----------
        entrieslist : list of list of tuple
            (row index, coefficient) pairs of the nonzero entries of each column
        objs : list, optional
            objective coefficients (Default value = all 0)
        lbs : list, optional
            lower bounds, None for -infinity (Default value = all 0)
        ubs : list, optional
            upper bounds, None for infinity (Default value = all infinity)

        """
        cdef int ncols = len(entrieslist)
        cdef int nnonz
        cdef SCIP_RATIONAL** c_objs = NULL
        cdef SCIP_RATIONAL** c_lbs = NULL
        cdef SCIP_RATIONAL** c_ubs = NULL
        cdef SCIP_RATIONAL** c_vals = NULL
        cdef int* c_beg = NULL
        cdef int* c_inds = NULL

        if ncols == 0:
            return
        objs = [0] * ncols if objs is None else list(objs)
        lbs = [0] * ncols if lbs is None else [-_INF if lb is None else lb for lb in lbs]
        ubs = [_INF] * ncols if ubs is None else [_INF if ub is None else ub for ub in ubs]
        if not len(objs) == len(lbs) == len(ubs) == ncols:
            raise ValueError("objs, lbs and ubs must have one entry per column")
        begs, inds, vals = [], [], []
        for entries in entrieslist:
            begs.append(len(inds))
            for ind, val in entries:
                inds.append(ind)
                vals.append(val)
        nnonz = len(vals)

        try:
            c_objs = _createRationalArray(objs)
            c_lbs = _createRationalArray(lbs)
            c_ubs = _createRationalArray(ubs)
            if nnonz > 0:
                c_vals = _createRationalArray(vals)
                c_beg = _createInts(begs)
                c_inds = _createInts(inds)
            PY_SCIP_CALL(SCIPlpiExactAddCols(self.lpi, ncols, c_objs, c_lbs, c_ubs, NULL, nnonz, c_beg, c_inds, c_vals))
        finally:
            free(c_inds)
            free(c_beg)
            _freeRationals(&c_vals, nnonz)
            _freeRationals(&c_ubs, ncols)
            _freeRationals(&c_lbs, ncols)
            _freeRationals(&c_objs, ncols)

    def delCols(self, firstcol, lastcol):
        """Delete the columns firstcol..lastcol."""
        PY_SCIP_CALL(SCIPlpiExactDelCols(self.lpi, firstcol, lastcol))

    def addRow(self, entries, lhs=0, rhs=None):
        """Add a single row.

        Parameters
        ----------
        entries : list of tuple
            (column index, coefficient) pairs of the nonzero entries of the row
        lhs : Fraction or int or float or str or None, optional
            left-hand side, None for -infinity (Default value = 0)
        rhs : Fraction or int or float or str or None, optional
            right-hand side, None for infinity (Default value = None)

        """
        self.addRows([entries], [lhs], [rhs])

    def addRows(self, entrieslist, lhss=None, rhss=None):
        """Add multiple rows.

        Parameters
        ----------
        entrieslist : list of list of tuple
            (column index, coefficient) pairs of the nonzero entries of each row
        lhss : list, optional
            left-hand sides, None for -infinity (Default value = all 0)
        rhss : list, optional
            right-hand sides, None for infinity (Default value = all infinity)

        """
        cdef int nrows = len(entrieslist)
        cdef int nnonz
        cdef SCIP_RATIONAL** c_lhss = NULL
        cdef SCIP_RATIONAL** c_rhss = NULL
        cdef SCIP_RATIONAL** c_vals = NULL
        cdef int* c_beg = NULL
        cdef int* c_inds = NULL

        if nrows == 0:
            return
        lhss = [0] * nrows if lhss is None else [-_INF if lhs is None else lhs for lhs in lhss]
        rhss = [_INF] * nrows if rhss is None else [_INF if rhs is None else rhs for rhs in rhss]
        if not len(lhss) == len(rhss) == nrows:
            raise ValueError("lhss and rhss must have one entry per row")
        begs, inds, vals = [], [], []
        for entries in entrieslist:
            begs.append(len(inds))
            for ind, val in entries:
                inds.append(ind)
                vals.append(val)
        nnonz = len(vals)

        try:
            c_lhss = _createRationalArray(lhss)
            c_rhss = _createRationalArray(rhss)
            if nnonz > 0:
                c_vals = _createRationalArray(vals)
                c_beg = _createInts(begs)
                c_inds = _createInts(inds)
            PY_SCIP_CALL(SCIPlpiExactAddRows(self.lpi, nrows, c_lhss, c_rhss, NULL, nnonz, c_beg, c_inds, c_vals))
        finally:
            free(c_inds)
            free(c_beg)
            _freeRationals(&c_vals, nnonz)
            _freeRationals(&c_rhss, nrows)
            _freeRationals(&c_lhss, nrows)

    def delRows(self, firstrow, lastrow):
        """Delete the rows firstrow..lastrow."""
        PY_SCIP_CALL(SCIPlpiExactDelRows(self.lpi, firstrow, lastrow))

    def getBounds(self, firstcol=0, lastcol=None):
        """Return the lower and upper bounds of the columns firstcol..lastcol, or None if the range is empty.

        Returns
        -------
        tuple of (list of Fraction, list of Fraction) or None

        """
        cdef SCIP_RATIONAL** c_lbs = NULL
        cdef SCIP_RATIONAL** c_ubs = NULL
        cdef int n

        if lastcol is None:
            lastcol = self.ncols() - 1
        if firstcol > lastcol:
            return None
        n = lastcol - firstcol + 1
        try:
            c_lbs = _createRationals(n)
            c_ubs = _createRationals(n)
            PY_SCIP_CALL(SCIPlpiExactGetBounds(self.lpi, firstcol, lastcol, c_lbs, c_ubs))
            return _fromRationals(c_lbs, n), _fromRationals(c_ubs, n)
        finally:
            _freeRationals(&c_ubs, n)
            _freeRationals(&c_lbs, n)

    def getSides(self, firstrow=0, lastrow=None):
        """Return the left- and right-hand sides of the rows firstrow..lastrow, or None if the range is empty.

        Returns
        -------
        tuple of (list of Fraction, list of Fraction) or None

        """
        cdef SCIP_RATIONAL** c_lhss = NULL
        cdef SCIP_RATIONAL** c_rhss = NULL
        cdef int n

        if lastrow is None:
            lastrow = self.nrows() - 1
        if firstrow > lastrow:
            return None
        n = lastrow - firstrow + 1
        try:
            c_lhss = _createRationals(n)
            c_rhss = _createRationals(n)
            PY_SCIP_CALL(SCIPlpiExactGetRows(self.lpi, firstrow, lastrow, c_lhss, c_rhss, NULL, NULL, NULL, NULL))
            return _fromRationals(c_lhss, n), _fromRationals(c_rhss, n)
        finally:
            _freeRationals(&c_rhss, n)
            _freeRationals(&c_lhss, n)

    def chgObj(self, col, obj):
        """Change the objective coefficient of a column."""
        cdef int c_col = col
        cdef SCIP_RATIONAL** c_obj = NULL

        try:
            c_obj = _createRationalArray([obj])
            PY_SCIP_CALL(SCIPlpiExactChgObj(self.lpi, 1, &c_col, c_obj))
        finally:
            _freeRationals(&c_obj, 1)

    def chgCoef(self, row, col, newval):
        """Change a single coefficient of the constraint matrix."""
        cdef SCIP_RATIONAL* c_val = _createRational(newval)

        try:
            PY_SCIP_CALL(SCIPlpiExactChgCoef(self.lpi, row, col, c_val))
        finally:
            SCIPrationalFree(&c_val)

    def chgBound(self, col, lb, ub):
        """Change the lower and upper bound of a column; None means infinite."""
        cdef int c_col = col
        cdef SCIP_RATIONAL** c_lb = NULL
        cdef SCIP_RATIONAL** c_ub = NULL

        try:
            c_lb = _createRationalArray([-_INF if lb is None else lb])
            c_ub = _createRationalArray([_INF if ub is None else ub])
            PY_SCIP_CALL(SCIPlpiExactChgBounds(self.lpi, 1, &c_col, c_lb, c_ub))
        finally:
            _freeRationals(&c_ub, 1)
            _freeRationals(&c_lb, 1)

    def chgSide(self, row, lhs, rhs):
        """Change the left- and right-hand side of a row; None means infinite."""
        cdef int c_row = row
        cdef SCIP_RATIONAL** c_lhs = NULL
        cdef SCIP_RATIONAL** c_rhs = NULL

        try:
            c_lhs = _createRationalArray([-_INF if lhs is None else lhs])
            c_rhs = _createRationalArray([_INF if rhs is None else rhs])
            PY_SCIP_CALL(SCIPlpiExactChgSides(self.lpi, 1, &c_row, c_lhs, c_rhs))
        finally:
            _freeRationals(&c_rhs, 1)
            _freeRationals(&c_lhs, 1)

    def clear(self):
        """Remove all columns and rows."""
        PY_SCIP_CALL(SCIPlpiExactClear(self.lpi))

    def nrows(self):
        """Return the number of rows."""
        cdef int nrows

        PY_SCIP_CALL(SCIPlpiExactGetNRows(self.lpi, &nrows))
        return nrows

    def ncols(self):
        """Return the number of columns."""
        cdef int ncols

        PY_SCIP_CALL(SCIPlpiExactGetNCols(self.lpi, &ncols))
        return ncols

    def solve(self, dual=True):
        """Solve the LP and return the optimal objective value, or None if the LP was not solved to optimality.

        Parameters
        ----------
        dual : bool, optional
            use the dual instead of the primal simplex (Default value = True)

        Returns
        -------
        Fraction or None

        """
        cdef SCIP_RATIONAL* objval = NULL

        if dual:
            PY_SCIP_CALL(SCIPlpiExactSolveDual(self.lpi))
        else:
            PY_SCIP_CALL(SCIPlpiExactSolvePrimal(self.lpi))
        if not SCIPlpiExactIsOptimal(self.lpi):
            return None
        PY_SCIP_CALL(SCIPrationalCreate(&objval))
        try:
            PY_SCIP_CALL(SCIPlpiExactGetObjval(self.lpi, objval))
            return _fromRational(objval)
        finally:
            SCIPrationalFree(&objval)

    def isOptimal(self):
        """Return whether the last solve reached optimality."""
        return SCIPlpiExactIsOptimal(self.lpi)

    def isPrimalFeasible(self):
        """Return whether the LP is proven primal feasible."""
        return SCIPlpiExactIsPrimalFeasible(self.lpi)

    def isDualFeasible(self):
        """Return whether the LP is proven dual feasible."""
        return SCIPlpiExactIsDualFeasible(self.lpi)

    def getObjVal(self):
        """Return the objective value of the last solve."""
        cdef SCIP_RATIONAL* objval = NULL

        PY_SCIP_CALL(SCIPrationalCreate(&objval))
        try:
            PY_SCIP_CALL(SCIPlpiExactGetSol(self.lpi, objval, NULL, NULL, NULL, NULL))
            return _fromRational(objval)
        finally:
            SCIPrationalFree(&objval)

    def getPrimal(self):
        """Return the primal solution of the last solve as a list of Fraction."""
        return _getSolPart(self.lpi, self.ncols(), 0)

    def getDual(self):
        """Return the dual solution of the last solve as a list of Fraction."""
        return _getSolPart(self.lpi, self.nrows(), 1)

    def getActivity(self):
        """Return the row activities of the last solve as a list of Fraction."""
        return _getSolPart(self.lpi, self.nrows(), 2)

    def getRedcost(self):
        """Return the reduced costs of the last solve as a list of Fraction."""
        return _getSolPart(self.lpi, self.ncols(), 3)

    def getPrimalRay(self):
        """Return a primal ray as a list of Fraction if the LP is proven unbounded, None otherwise."""
        cdef int ncols
        cdef SCIP_RATIONAL** c_ray = NULL

        if not SCIPlpiExactHasPrimalRay(self.lpi):
            return None
        ncols = self.ncols()
        try:
            c_ray = _createRationals(ncols)
            PY_SCIP_CALL(SCIPlpiExactGetPrimalRay(self.lpi, c_ray))
            return _fromRationals(c_ray, ncols)
        finally:
            _freeRationals(&c_ray, ncols)

    def getDualRay(self):
        """Return a dual Farkas ray as a list of Fraction if the LP is proven infeasible, None otherwise."""
        cdef int nrows
        cdef SCIP_RATIONAL** c_ray = NULL

        if not SCIPlpiExactHasDualRay(self.lpi):
            return None
        nrows = self.nrows()
        try:
            c_ray = _createRationals(nrows)
            PY_SCIP_CALL(SCIPlpiExactGetDualfarkas(self.lpi, c_ray))
            return _fromRationals(c_ray, nrows)
        finally:
            _freeRationals(&c_ray, nrows)

    def getNIterations(self):
        """Return the number of simplex iterations of the last solve."""
        cdef int niters

        PY_SCIP_CALL(SCIPlpiExactGetIterations(self.lpi, &niters))
        return niters

    def getBasisInds(self):
        """Return the indices of the basic columns and rows; index i >= 0 is column i, index i < 0 is row -i-1."""
        cdef int nrows = self.nrows()
        cdef int* c_binds = <int*> malloc(max(nrows, 1) * sizeof(int))
        cdef int i

        if c_binds == NULL:
            raise MemoryError()
        try:
            PY_SCIP_CALL(SCIPlpiExactGetBasisInd(self.lpi, c_binds))
            return [c_binds[i] for i in range(nrows)]
        finally:
            free(c_binds)

    def getBase(self):
        """Return the basis status of the columns and of the rows as SCIP_BASESTAT values.

        Returns
        -------
        tuple of (list of int, list of int)

        """
        cdef int ncols = self.ncols()
        cdef int nrows = self.nrows()
        cdef int* c_cstat = NULL
        cdef int* c_rstat = NULL
        cdef int i

        try:
            c_cstat = <int*> malloc(max(ncols, 1) * sizeof(int))
            c_rstat = <int*> malloc(max(nrows, 1) * sizeof(int))
            if c_cstat == NULL or c_rstat == NULL:
                raise MemoryError()
            PY_SCIP_CALL(SCIPlpiExactGetBase(self.lpi, c_cstat, c_rstat))
            return [c_cstat[i] for i in range(ncols)], [c_rstat[i] for i in range(nrows)]
        finally:
            free(c_rstat)
            free(c_cstat)

    def setBase(self, cstat, rstat):
        """Set the basis status of the columns and of the rows from SCIP_BASESTAT values.

        Parameters
        ----------
        cstat : list of int
            one status per column
        rstat : list of int
            one status per row

        """
        cdef int ncols = self.ncols()
        cdef int nrows = self.nrows()
        cdef int* c_cstat = NULL
        cdef int* c_rstat = NULL

        if len(cstat) != ncols:
            raise ValueError(f"cstat has length {len(cstat)}, expected {ncols}")
        if len(rstat) != nrows:
            raise ValueError(f"rstat has length {len(rstat)}, expected {nrows}")
        try:
            c_cstat = _createInts(cstat)
            c_rstat = _createInts(rstat)
            PY_SCIP_CALL(SCIPlpiExactSetBase(self.lpi, c_cstat, c_rstat))
        finally:
            free(c_rstat)
            free(c_cstat)

    def setIntParam(self, param, value):
        """Set an int parameter; raises KeyError if the exact LP solver does not support it.

        Parameters
        ----------
        param : SCIP_LPPARAM
        value : int

        """
        PY_SCIP_CALL(SCIPlpiExactSetIntpar(self.lpi, param, value))

    def setRealParam(self, param, value):
        """Set a real parameter; raises KeyError if the exact LP solver does not support it.

        Parameters
        ----------
        param : SCIP_LPPARAM
        value : float

        """
        PY_SCIP_CALL(SCIPlpiExactSetRealpar(self.lpi, param, value))

    def getIntParam(self, param):
        """Return an int parameter; raises KeyError if the exact LP solver does not support it.

        Parameters
        ----------
        param : SCIP_LPPARAM

        Returns
        -------
        int

        """
        cdef int value

        PY_SCIP_CALL(SCIPlpiExactGetIntpar(self.lpi, param, &value))
        return value

    def getRealParam(self, param):
        """Return a real parameter; raises KeyError if the exact LP solver does not support it.

        Parameters
        ----------
        param : SCIP_LPPARAM

        Returns
        -------
        float

        """
        cdef SCIP_Real value

        PY_SCIP_CALL(SCIPlpiExactGetRealpar(self.lpi, param, &value))
        return value
