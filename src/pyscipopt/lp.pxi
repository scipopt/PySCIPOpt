##@file lp.pxi
#@brief Base class of the LP Plugin
cdef class LP:
    cdef SCIP_LPI* lpi
    cdef readonly str name

    def __init__(self, name="LP", sense="minimize"):
        """
        Keyword arguments:
        name -- the name of the problem (default 'LP')
        sense -- objective sense (default minimize)
        """
        self.name = name
        n = str_conversion(name)
        if sense == "minimize":
            PY_SCIP_CALL(SCIPlpiCreate(&(self.lpi), NULL, n, SCIP_OBJSENSE_MINIMIZE))
        elif sense == "maximize":
            PY_SCIP_CALL(SCIPlpiCreate(&(self.lpi), NULL, n, SCIP_OBJSENSE_MAXIMIZE))
        else:
            raise Warning("unrecognized objective sense")

    def __dealloc__(self):
        PY_SCIP_CALL(SCIPlpiFree(&(self.lpi)))

    def __repr__(self):
        return self.name

    def writeLP(self, filename):
        """Writes LP to a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
        PY_SCIP_CALL(SCIPlpiWriteLP(self.lpi, filename))

    def readLP(self, filename):
        """Reads LP from a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
        PY_SCIP_CALL(SCIPlpiReadLP(self.lpi, filename))

    def infinity(self):
        """Returns infinity value of the LP.
        """
        return SCIPlpiInfinity(self.lpi)

    def isInfinity(self, val):
        """Checks if a given value is equal to the infinity value of the LP.

        Keyword arguments:
        val -- value that should be checked
        """
        return SCIPlpiIsInfinity(self.lpi, val)

    def addCol(self, entries, obj = 0.0, lb = 0.0, ub = None):
        """Adds a single column to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple consists of a row index and a coefficient
        obj     -- objective coefficient (default 0.0)
        lb      -- lower bound (default 0.0)
        ub      -- upper bound (default infinity)
        """
        nnonz = len(entries)

        cdef SCIP_Real* c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef SCIP_Real c_obj
        cdef SCIP_Real c_lb
        cdef SCIP_Real c_ub
        cdef int c_beg

        c_obj = obj
        c_lb = lb
        c_ub = ub if ub != None else self.infinity()
        c_beg = 0

        for i,entry in enumerate(entries):
            c_inds[i] = entry[0]
            c_coefs[i] = entry[1]

        PY_SCIP_CALL(SCIPlpiAddCols(self.lpi, 1, &c_obj, &c_lb, &c_ub, NULL, nnonz, &c_beg, c_inds, c_coefs))

        free(c_coefs)
        free(c_inds)

    def addCols(self, entrieslist, objs = None, lbs = None, ubs = None):
        """Adds multiple columns to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a row index
        objs  -- objective coefficient (default 0.0)
        lbs   -- lower bounds (default 0.0)
        ubs   -- upper bounds (default infinity)
        """

        ncols = len(entrieslist)
        nnonz = sum(len(entries) for entries in entrieslist)

        cdef SCIP_Real* c_objs   = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_lbs    = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_ubs    = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_coefs
        cdef int* c_inds
        cdef int* c_beg


        if nnonz > 0:
            c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
            c_inds = <int*>malloc(nnonz * sizeof(int))
            c_beg  = <int*>malloc(ncols * sizeof(int))

            tmp = 0
            for i,entries in enumerate(entrieslist):
                c_objs[i] = objs[i] if objs != None else 0.0
                c_lbs[i] = lbs[i] if lbs != None else 0.0
                c_ubs[i] = ubs[i] if ubs != None else self.infinity()
                c_beg[i] = tmp

                for entry in entries:
                    c_inds[tmp] = entry[0]
                    c_coefs[tmp] = entry[1]
                    tmp += 1

            PY_SCIP_CALL(SCIPlpiAddCols(self.lpi, ncols, c_objs, c_lbs, c_ubs, NULL, nnonz, c_beg, c_inds, c_coefs))

            free(c_beg)
            free(c_inds)
            free(c_coefs)
        else:
            for i in range(len(entrieslist)):
                c_objs[i] = objs[i] if objs != None else 0.0
                c_lbs[i] = lbs[i] if lbs != None else 0.0
                c_ubs[i] = ubs[i] if ubs != None else self.infinity()

            PY_SCIP_CALL(SCIPlpiAddCols(self.lpi, ncols, c_objs, c_lbs, c_ubs, NULL, 0, NULL, NULL, NULL))

        free(c_ubs)
        free(c_lbs)
        free(c_objs)

    def delCols(self, firstcol, lastcol):
        """Deletes a range of columns from the LP.

        Keyword arguments:
        firstcol -- first column to delete
        lastcol  -- last column to delete
        """
        PY_SCIP_CALL(SCIPlpiDelCols(self.lpi, firstcol, lastcol))

    def addRow(self, entries, lhs=0.0, rhs=None):
        """Adds a single row to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple contains a coefficient and a column index
        lhs     -- left-hand side of the row (default 0.0)
        rhs     -- right-hand side of the row (default infinity)
        """
        beg = 0
        nnonz = len(entries)

        cdef SCIP_Real* c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef SCIP_Real c_lhs
        cdef SCIP_Real c_rhs
        cdef int c_beg

        c_lhs = lhs
        c_rhs = rhs if rhs != None else self.infinity()
        c_beg = 0

        for i,entry in enumerate(entries):
            c_inds[i] = entry[0]
            c_coefs[i] = entry[1]

        PY_SCIP_CALL(SCIPlpiAddRows(self.lpi, 1, &c_lhs, &c_rhs, NULL, nnonz, &c_beg, c_inds, c_coefs))

        free(c_coefs)
        free(c_inds)

    def addRows(self, entrieslist, lhss = None, rhss = None):
        """Adds multiple rows to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a column index
        lhss        -- left-hand side of the row (default 0.0)
        rhss        -- right-hand side of the row (default infinity)
        """
        nrows = len(entrieslist)
        nnonz = sum(len(entries) for entries in entrieslist)

        cdef SCIP_Real* c_lhss  = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef SCIP_Real* c_rhss  = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef SCIP_Real* c_coefs = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef int* c_beg  = <int*>malloc(nrows * sizeof(int))

        tmp = 0
        for i,entries in enumerate(entrieslist):
            c_lhss[i] = lhss[i] if lhss != None else 0.0
            c_rhss[i] = rhss[i] if rhss != None else self.infinity()
            c_beg[i]  = tmp

            for entry in entries:
                c_inds[tmp] = entry[0]
                c_coefs[tmp] = entry[1]
                tmp += 1

        PY_SCIP_CALL(SCIPlpiAddRows(self.lpi, nrows, c_lhss, c_rhss, NULL, nnonz, c_beg, c_inds, c_coefs))

        free(c_beg)
        free(c_inds)
        free(c_coefs)
        free(c_lhss)
        free(c_rhss)

    def delRows(self, firstrow, lastrow):
        """Deletes a range of rows from the LP.

        Keyword arguments:
        firstrow -- first row to delete
        lastrow  -- last row to delete
        """
        PY_SCIP_CALL(SCIPlpiDelRows(self.lpi, firstrow, lastrow))

    def getBounds(self, firstcol = 0, lastcol = None):
        """Returns all lower and upper bounds for a range of columns.

        Keyword arguments:
        firstcol -- first column (default 0)
        lastcol  -- last column (default ncols - 1)
        """
        lastcol = lastcol if lastcol != None else self.ncols() - 1

        if firstcol > lastcol:
            return None

        ncols = lastcol - firstcol + 1
        cdef SCIP_Real* c_lbs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_ubs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetBounds(self.lpi, firstcol, lastcol, c_lbs, c_ubs))

        lbs = []
        ubs = []

        for i in range(ncols):
            lbs.append(c_lbs[i])
            ubs.append(c_ubs[i])

        free(c_ubs)
        free(c_lbs)

        return lbs, ubs

    def getSides(self, firstrow = 0, lastrow = None):
        """Returns all left- and right-hand sides for a range of rows.

        Keyword arguments:
        firstrow -- first row (default 0)
        lastrow  -- last row (default nrows - 1)
        """
        lastrow = lastrow if lastrow != None else self.nrows() - 1

        if firstrow > lastrow:
            return None

        nrows = lastrow - firstrow + 1
        cdef SCIP_Real* c_lhss = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef SCIP_Real* c_rhss = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSides(self.lpi, firstrow, lastrow, c_lhss, c_rhss))

        lhss = []
        rhss = []

        for i in range(firstrow, lastrow + 1):
            lhss.append(c_lhss[i])
            rhss.append(c_rhss[i])

        free(c_rhss)
        free(c_lhss)

        return lhss, rhss

    def chgObj(self, col, obj):
        """Changes objective coefficient of a single column.

        Keyword arguments:
        col -- column to change
        obj -- new objective coefficient
        """
        cdef int c_col = col
        cdef SCIP_Real c_obj = obj
        PY_SCIP_CALL(SCIPlpiChgObj(self.lpi, 1, &c_col, &c_obj))

    def chgCoef(self, row, col, newval):
        """Changes a single coefficient in the LP.

        Keyword arguments:
        row -- row to change
        col -- column to change
        newval -- new coefficient
        """
        PY_SCIP_CALL(SCIPlpiChgCoef(self.lpi, row, col, newval))

    def chgBound(self, col, lb, ub):
        """Changes the lower and upper bound of a single column.

        Keyword arguments:
        col -- column to change
        lb  -- new lower bound
        ub  -- new upper bound
        """
        cdef int c_col = col
        cdef SCIP_Real c_lb = lb
        cdef SCIP_Real c_ub = ub
        PY_SCIP_CALL(SCIPlpiChgBounds(self.lpi, 1, &c_col, &c_lb, &c_ub))

    def chgSide(self, row, lhs, rhs):
        """Changes the left- and right-hand side of a single row.

        Keyword arguments:
        row -- row to change
        lhs -- new left-hand side
        rhs -- new right-hand side
        """
        cdef int c_row = row
        cdef SCIP_Real c_lhs = lhs
        cdef SCIP_Real c_rhs = rhs
        PY_SCIP_CALL(SCIPlpiChgSides(self.lpi, 1, &c_row, &c_lhs, &c_rhs))

    def clear(self):
        """Clears the whole LP."""
        PY_SCIP_CALL(SCIPlpiClear(self.lpi))

    def nrows(self):
        """Returns the number of rows."""
        cdef int nrows
        PY_SCIP_CALL(SCIPlpiGetNRows(self.lpi, &nrows))
        return nrows

    def ncols(self):
        """Returns the number of columns."""
        cdef int ncols
        PY_SCIP_CALL(SCIPlpiGetNCols(self.lpi, &ncols))
        return ncols

    def solve(self, dual=True):
        """Solves the current LP.

        Keyword arguments:
        dual -- use the dual or primal Simplex method (default: dual)
        """
        if dual:
            PY_SCIP_CALL(SCIPlpiSolveDual(self.lpi))
        else:
            PY_SCIP_CALL(SCIPlpiSolvePrimal(self.lpi))

        cdef SCIP_Real objval
        PY_SCIP_CALL(SCIPlpiGetObjval(self.lpi, &objval))
        return objval

    def getPrimal(self):
        """Returns the primal solution of the last LP solve."""
        ncols = self.ncols()
        cdef SCIP_Real* c_primalsol = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, c_primalsol, NULL, NULL, NULL))
        primalsol = [0.0] * ncols
        for i in range(ncols):
            primalsol[i] = c_primalsol[i]
        free(c_primalsol)

        return primalsol

    def isPrimalFeasible(self):
        """Returns True iff LP is proven to be primal feasible."""
        return SCIPlpiIsPrimalFeasible(self.lpi)

    def getDual(self):
        """Returns the dual solution of the last LP solve."""
        nrows = self.nrows()
        cdef SCIP_Real* c_dualsol = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, NULL, c_dualsol, NULL, NULL))
        dualsol = [0.0] * nrows
        for i in range(nrows):
            dualsol[i] = c_dualsol[i]
        free(c_dualsol)

        return dualsol

    def isDualFeasible(self):
        """Returns True iff LP is proven to be dual feasible."""
        return SCIPlpiIsDualFeasible(self.lpi)

    def getPrimalRay(self):
        """Returns a primal ray if possible, None otherwise."""
        if not SCIPlpiHasPrimalRay(self.lpi):
            return None
        ncols = self.ncols()
        cdef SCIP_Real* c_ray  = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetPrimalRay(self.lpi, c_ray))
        ray = [0.0] * ncols
        for i in range(ncols):
            ray[i] = c_ray[i]
        free(c_ray)

        return ray

    def getDualRay(self):
        """Returns a dual ray if possible, None otherwise."""
        if not SCIPlpiHasDualRay(self.lpi):
            return None
        nrows = self.nrows()
        cdef SCIP_Real* c_ray  = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetDualfarkas(self.lpi, c_ray))
        ray = [0.0] * nrows
        for i in range(nrows):
            ray[i] = c_ray[i]
        free(c_ray)

        return ray

    def getNIterations(self):
        """Returns the number of LP iterations of the last LP solve."""
        cdef int niters
        PY_SCIP_CALL(SCIPlpiGetIterations(self.lpi, &niters))
        return niters

    def getRedcost(self):
        """Returns the reduced cost vector of the last LP solve."""
        ncols = self.ncols()

        cdef SCIP_Real* c_redcost = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSol(self.lpi, NULL, NULL, NULL, NULL, c_redcost))

        redcost = []
        for i in range(ncols):
            redcost[i].append(c_redcost[i])

        free(c_redcost)
        return redcost

    def getBasisInds(self):
        """Returns the indices of the basic columns and rows; index i >= 0 corresponds to column i, index i < 0 to row -i-1"""
        nrows = self.nrows()
        cdef int* c_binds  = <int*> malloc(nrows * sizeof(int))

        PY_SCIP_CALL(SCIPlpiGetBasisInd(self.lpi, c_binds))

        binds = []
        for i in range(nrows):
            binds.append(c_binds[i])

        free(c_binds)
        return binds
