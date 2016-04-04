from os.path import abspath
import sys

#cimport pyscipopt.scip as scip
cimport pyscipopt.linexpr
from pyscipopt.linexpr cimport LinExpr, LinCons

from libc.stdlib cimport malloc, free

include "pricer.pxi"
include "conshdlr.pxi"
include "presol.pxi"
include "sepa.pxi"
include "propagator.pxi"
include "heuristic.pxi"
include "branchrule.pxi"


# for external user functions use def; for functions used only inside the interface (starting with _) use cdef
# todo: check whether this is currently done like this

if sys.version_info >= (3, 0):
    str_conversion = lambda x:bytes(x,'utf-8')
else:
    str_conversion = lambda x:x

def scipErrorHandler(function):
    def wrapper(*args, **kwargs):
        return PY_SCIP_CALL(function(*args, **kwargs))
    return wrapper

# Mapping the SCIP_RESULT enum to a python class
# This is required to return SCIP_RESULT in the python code
# In __init__.py this is imported as SCIP_RESULT to keep the
# original naming scheme using capital letters
cdef class PY_SCIP_RESULT:
    DIDNOTRUN   =   1
    DELAYED     =   2
    DIDNOTFIND  =   3
    FEASIBLE    =   4
    INFEASIBLE  =   5
    UNBOUNDED   =   6
    CUTOFF      =   7
    SEPARATED   =   8
    NEWROUND    =   9
    REDUCEDOM   =  10
    CONSADDED   =  11
    CONSSHANGED =  12
    BRANCHED    =  13
    SOLVELP     =  14
    FOUNDSOL    =  15
    SUSPENDED   =  16
    SUCCESS     =  17


cdef class PY_SCIP_PARAMSETTING:
    DEFAULT     = 0
    AGRESSIVE   = 1
    FAST        = 2
    OFF         = 3

cdef class PY_SCIP_STATUS:
    UNKNOWN        =  0
    USERINTERRUPT  =  1
    NODELIMIT      =  2
    TOTALNODELIMIT =  3
    STALLNODELIMIT =  4
    TIMELIMIT      =  5
    MEMLIMIT       =  6
    GAPLIMIT       =  7
    SOLLIMIT       =  8
    BESTSOLLIMIT   =  9
    RESTARTLIMIT   = 10
    OPTIMAL        = 11
    INFEASIBLE     = 12
    UNBOUNDED      = 13
    INFORUNBD      = 14

cdef class PY_SCIP_PROPTIMING:
    BEFORELP     = 0X001U
    DURINGLPLOOP = 0X002U
    AFTERLPLOOP  = 0X004U
    AFTERLPNODE  = 0X008U

cdef class PY_SCIP_PRESOLTIMING:
    NONE       = 0x000u
    FAST       = 0x002u
    MEDIUM     = 0x004u
    EXHAUSTIVE = 0x008u

cdef class PY_SCIP_HEURTIMING:
    BEFORENODE        = 0x001u
    DURINGLPLOOP      = 0x002u
    AFTERLPLOOP       = 0x004u
    AFTERLPNODE       = 0x008u
    AFTERPSEUDONODE   = 0x010u
    AFTERLPPLUNGE     = 0x020u
    AFTERPSEUDOPLUNGE = 0x040u
    DURINGPRICINGLOOP = 0x080u
    BEFOREPRESOL      = 0x100u
    DURINGPRESOLLOOP  = 0x200u
    AFTERPROPLOOP     = 0x400u

def PY_SCIP_CALL(SCIP_RETCODE rc):
    if rc == SCIP_OKAY:
        pass
    elif rc == SCIP_ERROR:
        raise Exception('SCIP: unspecified error!')
    elif rc == SCIP_NOMEMORY:
        raise MemoryError('SCIP: insufficient memory error!')
    elif rc == SCIP_READERROR:
        raise IOError('SCIP: read error!')
    elif rc == SCIP_WRITEERROR:
        raise IOError('SCIP: write error!')
    elif rc == SCIP_NOFILE:
        raise IOError('SCIP: file not found error!')
    elif rc == SCIP_FILECREATEERROR:
        raise IOError('SCIP: cannot create file!')
    elif rc == SCIP_LPERROR:
        raise Exception('SCIP: error in LP solver!')
    elif rc == SCIP_NOPROBLEM:
        raise Exception('SCIP: no problem exists!')
    elif rc == SCIP_INVALIDCALL:
        raise Exception('SCIP: method cannot be called at this time'
                            + ' in solution process!')
    elif rc == SCIP_INVALIDDATA:
        raise Exception('SCIP: error in input data!')
    elif rc == SCIP_INVALIDRESULT:
        raise Exception('SCIP: method returned an invalid result code!')
    elif rc == SCIP_PLUGINNOTFOUND:
        raise Exception('SCIP: a required plugin was not found !')
    elif rc == SCIP_PARAMETERUNKNOWN:
        raise KeyError('SCIP: the parameter with the given name was not found!')
    elif rc == SCIP_PARAMETERWRONGTYPE:
        raise LookupError('SCIP: the parameter is not of the expected type!')
    elif rc == SCIP_PARAMETERWRONGVAL:
        raise ValueError('SCIP: the value is invalid for the given parameter!')
    elif rc == SCIP_KEYALREADYEXISTING:
        raise KeyError('SCIP: the given key is already existing in table!')
    elif rc == SCIP_MAXDEPTHLEVEL:
        raise Exception('SCIP: maximal branching depth level exceeded!')
    else:
        raise Exception('SCIP: unknown return code!')
    return rc


cdef class Col:
    """Base class holding a pointer to corresponding SCIP_COL"""
    cdef SCIP_COL* _col


cdef class Row:
    """Base class holding a pointer to corresponding SCIP_ROW"""
    cdef SCIP_ROW* _row


cdef class Solution:
    """Base class holding a pointer to corresponding SCIP_SOL"""
    cdef SCIP_SOL* _solution


cdef class Var:
    """Base class holding a pointer to corresponding SCIP_VAR"""
    cdef SCIP_VAR* _var

cdef class Lpi:
    """Base class holding a pointer to corresponding SCIP_LPI"""
    cdef SCIP_LPI* _lpi

class LP:
    def __init__(self, name="LP", objsen=SCIP_OBJSENSE_MINIMIZE):
        """
        Keyword arguments:
        name -- the name of the problem (default 'LP')
        objsen -- objective sense (default minimize)
        """
        self.lpi = Lpi()
        self.name = name
        cdef Lpi lpi
        lpi = self.lpi
        PY_SCIP_CALL(SCIPlpiCreate(&(lpi._lpi), NULL, name, objsen))

    def __del__(self):
        cdef Lpi lpi
        cdef SCIP_LPI* _lpi
        lpi = self.lpi
        _lpi = lpi._lpi
        PY_SCIP_CALL(SCIPlpiFree(&_lpi))

    def __repr__(self):
        return self.name

    def writeLP(self, filename):
        """Writes LP to a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
        cdef Lpi lpi
        lpi = self.lpi
        PY_SCIP_CALL(SCIPlpiWriteLP(lpi._lpi, filename))

    def readLP(self, filename):
        """Reads LP from a file.

        Keyword arguments:
        filename -- the name of the file to be used
        """
        cdef Lpi lpi
        lpi = self.lpi
        PY_SCIP_CALL(SCIPlpiReadLP(lpi._lpi, filename))

    def infinity(self):
        """Returns infinity value of the LP.
        """
        cdef Lpi lpi
        lpi = self.lpi
        return SCIPlpiInfinity(lpi._lpi)

    def isInfinity(self, val):
        """Checks if a given value is equal to the infinity value of the LP.

        Keyword arguments:
        val -- value that should be checked
        """
        cdef Lpi lpi
        lpi = self.lpi
        return SCIPlpiIsInfinity(lpi._lpi, val)

    def addCol(self, entries, obj = 0.0, lb = 0.0, ub = None):
        """Adds a single column to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple consists of a coefficient and a row index
        obj     -- objective coefficient (default 0.0)
        lb      -- lower bound (default 0.0)
        ub      -- upper bound (default infinity)
        """
        cdef Lpi lpi
        lpi = self.lpi

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

        PY_SCIP_CALL(SCIPlpiAddCols(lpi._lpi, 1, &c_obj, &c_lb, &c_ub, NULL, nnonz, &c_beg, c_inds, c_coefs))

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
        cdef Lpi lpi
        lpi = self.lpi

        ncols = len(entrieslist)
        nnonz = sum(len(entries) for entries in entrieslist)

        cdef SCIP_Real* c_objs   = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_lbs    = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_ubs    = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_coefs  = <SCIP_Real*> malloc(nnonz * sizeof(SCIP_Real))
        cdef int* c_inds = <int*>malloc(nnonz * sizeof(int))
        cdef int* c_beg  = <int*>malloc(ncols * sizeof(int))

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

        PY_SCIP_CALL(SCIPlpiAddCols(lpi._lpi, ncols, c_objs, c_lbs, c_ubs, NULL, nnonz, c_beg, c_inds, c_coefs))

        free(c_beg)
        free(c_inds)
        free(c_coefs)
        free(c_ubs)
        free(c_lbs)
        free(c_objs)

    def delCols(self, firstcol, lastcol):
        """Deletes a range of columns from the LP.

        Keyword arguments:
        firstcol -- first column to delete
        lastcol  -- last column to delete
        """
        cdef Lpi lpi
        lpi = self.lpi
        PY_SCIP_CALL(SCIPlpiDelCols(lpi._lpi, firstcol, lastcol))

    def addRow(self, entries, lhs=0.0, rhs=None):
        """Adds a single row to the LP.

        Keyword arguments:
        entries -- list of tuples, each tuple contains a coefficient and a column index
        lhs     -- left-hand side of the row (default 0.0)
        rhs     -- right-hand side of the row (default infinity)
        """
        cdef Lpi lpi
        lpi = self.lpi

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

        PY_SCIP_CALL(SCIPlpiAddRows(lpi._lpi, 1, &c_lhs, &c_rhs, NULL, nnonz, &c_beg, c_inds, c_coefs))

        free(c_coefs)
        free(c_inds)

    def addRows(self, entrieslist, lhss = None, rhss = None):
        """Adds multiple rows to the LP.

        Keyword arguments:
        entrieslist -- list containing lists of tuples, each tuple contains a coefficient and a column index
        lhss        -- left-hand side of the row (default 0.0)
        rhss        -- right-hand side of the row (default infinity)
        """
        cdef Lpi lpi
        lpi = self.lpi

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

        PY_SCIP_CALL(SCIPlpiAddRows(lpi._lpi, nrows, c_lhss, c_rhss, NULL, nnonz, c_beg, c_inds, c_coefs))

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
        cdef Lpi lpi
        lpi = self.lpi
        PY_SCIP_CALL(SCIPlpiDelRows(lpi._lpi, firstrow, lastrow))

    def getBounds(self, firstcol = 0, lastcol = None):
        """Returns all lower and upper bounds for a range of columns.

        Keyword arguments:
        firstcol -- first column (default 0)
        lastcol  -- last column (default ncols - 1)
        """
        cdef Lpi lpi
        lpi = self.lpi

        lastcol = lastcol if lastcol != None else self.ncols() - 1

        if firstcol > lastcol:
            return None

        ncols = lastcol - firstcol + 1
        cdef SCIP_Real* c_lbs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef SCIP_Real* c_ubs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetBounds(lpi._lpi, firstcol, lastcol, c_lbs, c_ubs))

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
        cdef Lpi lpi
        lpi = self.lpi

        lastrow = lastrow if lastrow != None else self.nrows() - 1

        if firstrow > lastrow:
            return None

        nrows = lastrow - firstrow + 1
        cdef SCIP_Real* c_lhss = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef SCIP_Real* c_rhss = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSides(lpi._lpi, firstrow, lastrow, c_lhss, c_rhss))

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
        cdef Lpi lpi
        lpi = self.lpi

        cdef int c_col = col
        cdef SCIP_Real c_obj = obj
        PY_SCIP_CALL(SCIPlpiChgObj(lpi._lpi, 1, &c_col, &c_obj))

    def chgCoef(self, row, col, newval):
        """Changes a single coefficient in the LP.

        Keyword arguments:
        row -- row to change
        col -- column to change
        newval -- new coefficient
        """
        cdef Lpi lpi
        lpi = self.lpi

        PY_SCIP_CALL(SCIPlpiChgCoef(lpi._lpi, row, col, newval))

    def chgBound(self, col, lb, ub):
        """Changes the lower and upper bound of a single column.

        Keyword arguments:
        col -- column to change
        lb  -- new lower bound
        ub  -- new upper bound
        """
        cdef Lpi lpi
        lpi = self.lpi

        cdef int c_col = col
        cdef SCIP_Real c_lb = lb
        cdef SCIP_Real c_ub = ub
        PY_SCIP_CALL(SCIPlpiChgBounds(lpi._lpi, 1, &c_col, &c_lb, &c_ub))

    def chgSide(self, row, lhs, rhs):
        """Changes the left- and right-hand side of a single row.

        Keyword arguments:
        row -- row to change
        lhs -- new left-hand side
        rhs -- new right-hand side
        """
        cdef Lpi lpi
        lpi = self.lpi

        cdef int c_row = row
        cdef SCIP_Real c_lhs = lhs
        cdef SCIP_Real c_rhs = rhs
        PY_SCIP_CALL(SCIPlpiChgSides(lpi._lpi, 1, &c_row, &c_lhs, &c_rhs))

    def clear(self):
        """Clears the whole LP."""
        cdef Lpi lpi
        lpi = self.lpi
        PY_SCIP_CALL(SCIPlpiClear(lpi._lpi))

    def nrows(self):
        """Returns the number of rows."""
        cdef Lpi lpi
        lpi = self.lpi

        cdef int nrows
        PY_SCIP_CALL(SCIPlpiGetNRows(lpi._lpi, &nrows))
        return nrows

    def ncols(self):
        """Returns the number of columns."""
        cdef Lpi lpi
        lpi = self.lpi

        cdef int ncols
        PY_SCIP_CALL(SCIPlpiGetNCols(lpi._lpi, &ncols))
        return ncols

    def solve(self, dual=True):
        """Solves the current LP.

        Keyword arguments:
        dual -- use the dual or primal Simplex method (default: dual)
        """
        cdef Lpi lpi
        lpi = self.lpi

        if dual:
            PY_SCIP_CALL(SCIPlpiSolveDual(lpi._lpi))
        else:
            PY_SCIP_CALL(SCIPlpiSolvePrimal(lpi._lpi))

        cdef SCIP_Real objval
        PY_SCIP_CALL(SCIPlpiGetObjval(lpi._lpi, &objval))
        return objval

    def getPrimal(self):
        """Returns the primal solution of the last LP solve."""
        cdef Lpi lpi
        lpi = self.lpi

        ncols = self.ncols()

        cdef SCIP_Real* c_primalsol = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSol(lpi._lpi, NULL, c_primalsol, NULL, NULL, NULL))

        primalsol = [0.0] * ncols
        for i in range(ncols):
            primalsol[i] = c_primalsol[i]

        free(c_primalsol)

        return primalsol

    def getDual(self):
        """Returns the dual solution of the last LP solve."""
        cdef Lpi lpi
        lpi = self.lpi

        nrows = self.nrows()

        cdef SCIP_Real* c_dualsol = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetSol(lpi._lpi, NULL, NULL, c_dualsol, NULL, NULL))

        dualsol = [0.0] * nrows
        for i in range(nrows):
            dualsol[i] = c_dualsol[i]

        free(c_dualsol)

        return dualsol

    def getPrimalRay(self):
        """Returns a primal ray if possible, None otherwise."""
        cdef Lpi lpi
        lpi = self.lpi

        if not SCIPlpiHasPrimalRay(lpi._lpi):
            return None

        ncols = self.ncols()
        cdef SCIP_Real* c_ray  = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetPrimalRay(lpi._lpi, c_ray))

        ray = [0.0] * ncols
        for i in range(ncols):
            ray[i] = c_ray[i]

        free(c_ray)

        return ray

    def getDualRay(self):
        """Returns a dual ray if possible, None otherwise."""
        cdef Lpi lpi
        lpi = self.lpi

        if not SCIPlpiHasDualRay(lpi._lpi):
            return None

        nrows = self.nrows()
        cdef SCIP_Real* c_ray  = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPlpiGetDualfarkas(lpi._lpi, c_ray))

        ray = [0.0] * nrows
        for i in range(nrows):
            ray[i] = c_ray[i]

        free(c_ray)

        return ray

    def getNIterations(self):
        """Returns a the number of LP iterations of the last LP solve."""
        cdef Lpi lpi
        lpi = self.lpi

        cdef int niters
        PY_SCIP_CALL(SCIPlpiGetIterations(lpi._lpi, &niters))
        return niters

cdef class Variable(LinExpr):
    """Is a linear expression and has SCIP_VAR*"""
    cdef public var
    cdef public name

    def __init__(self, name=None):
        self.var = Var()
        self.name = name
        LinExpr.__init__(self, {(self,) : 1.0})

    def __hash__(self):
        return hash(id(self))

    def __richcmp__(self, other, op):
        if op == 0: # <
            return id(self) < id(other)
        elif op == 4: # > 
            return id(self) > id(other)

    def __repr__(self):
        return self.name

    def vtype(self):
        cdef Var v
        cdef SCIP_VAR* _var
        v = self.var
        _var = v._var
        vartype = SCIPvarGetType(_var)
        if vartype == SCIP_VARTYPE_BINARY:
            return "BINARY"
        elif vartype == SCIP_VARTYPE_INTEGER:
            return "INTEGER"
        elif vartype == SCIP_VARTYPE_CONTINUOUS or vartype == SCIP_VARTYPE_IMPLINT:
            return "CONTINUOUS"

    def isOriginal(self):
        cdef Var v
        cdef SCIP_VAR* _var
        v = self.var
        _var = v._var
        return SCIPvarIsOriginal(_var)

    def isInLP(self):
        cdef Var v
        cdef SCIP_VAR* _var
        v = self.var
        _var = v._var
        return SCIPvarIsInLP(_var)

    def getCol(self):
        cdef Var v
        cdef SCIP_VAR* _var
        v = self.var
        _var = v._var
        col = Col()
        cdef SCIP_COL* _col
        _col = col._col
        _col = SCIPvarGetCol(_var)
        return col

cdef pythonizeVar(SCIP_VAR* scip_var, name):
    var = Variable(name)
    cdef Var v
    v = var.var
    v._var = scip_var
    return var

cdef class Cons:
    cdef SCIP_CONS* _cons

class Constraint:
    def __init__(self, name=None):
        self.cons = Cons()
        self.name = name

    def __repr__(self):
        return self.name

    def isOriginal(self):
        cdef Cons c
        cdef SCIP_CONS* _cons
        c = self.cons
        _cons = c._cons
        return SCIPconsIsOriginal(_cons)


cdef pythonizeCons(SCIP_CONS* scip_cons, name):
    cons = Constraint(name)
    cdef Cons c
    c = cons.cons
    c._cons = scip_cons
    return cons

# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
cdef class Model:
    cdef SCIP* _scip
    # store best solution to get the solution values easier
    cdef SCIP_SOL* _bestSol
    # can be used to store problem data
    cdef public object data

    def __init__(self, problemName='model', defaultPlugins=True):
        """
        Keyword arguments:
        problemName -- the name of the problem (default 'model')
        defaultPlugins -- use default plugins? (default True)
        """
        self.create()
        if defaultPlugins:
            self.includeDefaultPlugins()
        self.createProbBasic(problemName)

    def __del__(self):
        self.freeTransform()
        self.freeProb()
        self.free()

    @scipErrorHandler
    def create(self):
        return SCIPcreate(&self._scip)

    @scipErrorHandler
    def includeDefaultPlugins(self):
        return SCIPincludeDefaultPlugins(self._scip)

    @scipErrorHandler
    def createProbBasic(self, problemName='model'):
        n = str_conversion(problemName)
        return SCIPcreateProbBasic(self._scip, n)

    @scipErrorHandler
    def free(self):
        return SCIPfree(&self._scip)

    @scipErrorHandler
    def freeProb(self):
        return SCIPfreeProb(self._scip)

    @scipErrorHandler
    def freeTransform(self):
        return SCIPfreeTransform(self._scip)

    def printVersion(self):
        SCIPprintVersion(self._scip, NULL)

    def getTotalTime(self):
        return SCIPgetTotalTime(self._scip)

    def getSolvingTime(self):
        return SCIPgetSolvingTime(self._scip)

    def getReadingTime(self):
        return SCIPgetReadingTime(self._scip)

    def getPresolvingTime(self):
        return SCIPgetPresolvingTime(self._scip)

    def infinity(self):
        """Retrieve 'infinity' value."""
        return SCIPinfinity(self._scip)

    def epsilon(self):
        """Return epsilon for e.g. equality checks"""
        return SCIPepsilon(self._scip)

    def feastol(self):
        """Return feasibility tolerance"""
        return SCIPfeastol(self._scip)

    #@scipErrorHandler       We'll be able to use decorators when we
    #                        interface the relevant classes (SCIP_VAR, ...)
    cdef _createVarBasic(self, SCIP_VAR** scip_var, name,
                        lb, ub, obj, SCIP_VARTYPE varType):
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, scip_var,
                           n, lb, ub, obj, varType))

    cdef _addVar(self, SCIP_VAR* scip_var):
        PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

    cdef _addPricedVar(self, SCIP_VAR* scip_var):
        PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, 1.0))

    cdef _createConsLinear(self, SCIP_CONS** cons, name, nvars,
                                SCIP_VAR** vars, SCIP_Real* vals, lhs, rhs,
                                initial=True, separate=True, enforce=True, check=True,
                                propagate=True, local=False, modifiable=False, dynamic=False,
                                removable=False, stickingatnode=False):
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPcreateConsLinear(self._scip, cons,
                                                    n, nvars, vars, vals,
                                                    lhs, rhs, initial, separate, enforce,
                                                    check, propagate, local, modifiable,
                                                    dynamic, removable, stickingatnode) )

    cdef _createConsSOS1(self, SCIP_CONS** cons, name, nvars,
                              SCIP_VAR** vars, SCIP_Real* weights,
                              initial=True, separate=True, enforce=True, check=True,
                              propagate=True, local=False, dynamic=False, removable=False,
                              stickingatnode=False):
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPcreateConsSOS1(self._scip, cons,
                                                    n, nvars, vars, weights,
                                                    initial, separate, enforce,
                                                    check, propagate, local, dynamic, removable,
                                                    stickingatnode) )

    cdef _createConsSOS2(self, SCIP_CONS** cons, name, nvars,
                              SCIP_VAR** vars, SCIP_Real* weights,
                              initial=True, separate=True, enforce=True, check=True,
                              propagate=True, local=False, dynamic=False, removable=False,
                              stickingatnode=False):
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPcreateConsSOS2(self._scip, cons,
                                                    n, nvars, vars, weights,
                                                    initial, separate, enforce,
                                                    check, propagate, local, dynamic, removable,
                                                    stickingatnode) )

    cdef _addCoefLinear(self, SCIP_CONS* cons, SCIP_VAR* var, val):
        PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, cons, var, val))

    cdef _addCons(self, SCIP_CONS* cons):
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons))

    cdef _addVarSOS1(self, SCIP_CONS* cons, SCIP_VAR* var, weight):
        PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, cons, var, weight))

    cdef _appendVarSOS1(self, SCIP_CONS* cons, SCIP_VAR* var):
        PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, cons, var))

    cdef _addVarSOS2(self, SCIP_CONS* cons, SCIP_VAR* var, weight):
        PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, cons, var, weight))

    cdef _appendVarSOS2(self, SCIP_CONS* cons, SCIP_VAR* var):
        PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, cons, var))

    cdef _writeVarName(self, SCIP_VAR* var):
        PY_SCIP_CALL(SCIPwriteVarName(self._scip, NULL, var, False))

    cdef _releaseVar(self, SCIP_VAR* var):
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &var))

    cdef _releaseCons(self, SCIP_CONS* cons):
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &cons))


    # Objective function

    def setMinimize(self):
        """Set the objective sense to maximization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        """Set the objective sense to minimization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    def setObjlimit(self, objlimit):
        """Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Keyword arguments:
        objlimit -- limit on the objective function
        """
        PY_SCIP_CALL(SCIPsetObjlimit(self._scip, objlimit))

    def setObjective(self, coeffs, sense = 'minimize'):
        """Establish the objective function, either as a variable dictionary or as a linear expression.

        Keyword arguments:
        coeffs -- the coefficients
        sense -- the objective sense (default 'minimize')
        """
        cdef SCIP_Real coeff
        cdef Var v
        cdef SCIP_VAR* _var
        if isinstance(coeffs, LinExpr):
            # transform linear expression into variable dictionary
            terms = coeffs.terms
            coeffs = {t[0]:c for t, c in terms.items() if c != 0.0}
        elif coeffs == 0:
            coeffs = {}
        for k in coeffs:
            coeff = <SCIP_Real>coeffs[k]
            v = k.var
            PY_SCIP_CALL(SCIPchgVarObj(self._scip, v._var, coeff))
        if sense == 'maximize':
            self.setMaximize()
        else:
            self.setMinimize()

    # Setting parameters
    def setPresolve(self, setting):
        """Set presolving parameter settings.

        Keyword arguments:
        setting -- the parameter settings
        """
        PY_SCIP_CALL(SCIPsetPresolving(self._scip, setting, True))

    # Write original problem to file
    def writeProblem(self, filename='origprob.cip'):
        """Write original problem to a file.

        Keyword arguments:
        filename -- the name of the file to be used (default 'origprob.cip')
        """
        if filename.find('.') < 0:
            filename = filename + '.cip'
            ext = str_conversion('cip')
        else:
            ext = str_conversion(filename.split('.')[1])
        fn = str_conversion(filename)
        PY_SCIP_CALL(SCIPwriteOrigProblem(self._scip, fn, ext, False))
        print('wrote original problem to file ' + filename)

    # Variable Functions

    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, pricedVar = False):
        """Create a new variable.

        Keyword arguments:
        name -- the name of the variable (default '')
        vtype -- the typ of the variable (default 'C')
        lb -- the lower bound of the variable (default 0.0)
        ub -- the upper bound of the variable (default None)
        obj -- the objective value of the variable (default 0.0)
        pricedVar -- is the variable a pricing candidate? (default False)
        """
        if ub is None:
            ub = SCIPinfinity(self._scip)
        cdef SCIP_VAR* scip_var
        if vtype in ['C', 'CONTINUOUS']:
            self._createVarBasic(&scip_var, name, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS)
        elif vtype in ['B', 'BINARY']:
            lb = 0.0
            ub = 1.0
            self._createVarBasic(&scip_var, name, lb, ub, obj, SCIP_VARTYPE_BINARY)
        elif vtype in ['I', 'INTEGER']:
            self._createVarBasic(&scip_var, name, lb, ub, obj, SCIP_VARTYPE_INTEGER)

        if pricedVar:
            self._addPricedVar(scip_var)
        else:
            self._addVar(scip_var)

        self._releaseVar(scip_var)
        return pythonizeVar(scip_var, name)

    def releaseVar(self, var):
        """Release the variable.

        Keyword arguments:
        var -- the variable
        """
        cdef SCIP_VAR* _var
        cdef Var v
        v = var.var
        _var = v._var
        self._releaseVar(_var)

    def getTransformedVar(self, var):
        """Retrieve the transformed variable.

        Keyword arguments:
        var -- the variable
        """
        cdef SCIP_VAR* _tvar
        cdef Var v
        cdef Var tv
        v = var.var
        tv = var.var
        _tvar = tv._var
        PY_SCIP_CALL(SCIPtransformVar(self._scip, v._var, &_tvar))
        name = <bytes> SCIPvarGetName(_tvar)
        return pythonizeVar(_tvar, name)


    def chgVarLb(self, var, lb=None):
        """Changes the lower bound of the specified variable.

        Keyword arguments:
        var -- the variable
        lb -- the lower bound (default None)
        """
        cdef SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <SCIP_VAR*>v._var

        if lb is None:
           lb = -SCIPinfinity(self._scip)

        PY_SCIP_CALL(SCIPchgVarLb(self._scip, _var, lb))

    def chgVarUb(self, var, ub=None):
        """Changes the upper bound of the specified variable.

        Keyword arguments:
        var -- the variable
        ub -- the upper bound (default None)
        """
        cdef SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <SCIP_VAR*>v._var

        if ub is None:
           ub = SCIPinfinity(self._scip)

        PY_SCIP_CALL(SCIPchgVarLb(self._scip, _var, ub))

    def chgVarType(self, var, vtype):
        cdef SCIP_VAR* _var
        cdef Var v
        cdef SCIP_Bool infeasible
        v = var.var
        _var = <SCIP_VAR*>v._var
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, _var, SCIP_VARTYPE_CONTINUOUS, &infeasible))
        elif vtype in ['B', 'BINARY']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, _var, SCIP_VARTYPE_BINARY, &infeasible))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, _var, SCIP_VARTYPE_INTEGER, &infeasible))
        else:
            print('wrong variable type: ',vtype)
        if infeasible:
            print('could not change variable type of variable ',<bytes> SCIPvarGetName(_var))

    def getVars(self, transformed=False):
        """Retrieve all variables.

        Keyword arguments:
        transformed -- get transformed variables instead of original
        """
        cdef SCIP_VAR** _vars
        cdef SCIP_VAR* _var
        cdef int _nvars
        vars = []

        if transformed:
            _vars = SCIPgetVars(self._scip)
            _nvars = SCIPgetNVars(self._scip)
        else:
            _vars = SCIPgetOrigVars(self._scip)
            _nvars = SCIPgetNOrigVars(self._scip)

        for i in range(_nvars):
            _var = _vars[i]
            name = SCIPvarGetName(_var).decode("utf-8")
            vars.append(pythonizeVar(_var, name))

        return vars


    # Constraint functions
    # . By default the lhs is set to 0.0.
    # If the lhs is to be unbounded, then you set lhs to None.
    # By default the rhs is unbounded.
    def addCons(self, coeffs, lhs=0.0, rhs=None, name="cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, modifiable=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add a linear or quadratic constraint.

        Keyword arguments:
        coeffs -- list of coefficients
        lhs -- the left hand side (default 0.0)
        rhs -- the right hand side (default None)
        name -- the name of the constraint (default 'cons')
        initial -- should the LP relaxation of constraint be in the initial LP? (default True)
        separate -- should the constraint be separated during LP processing? (default True)
        enforce -- should the constraint be enforced during node processing? (default True)
        check -- should the constraint be checked for feasibility? (default True)
        propagate -- should the constraint be propagated during node processing? (default True)
        local -- is the constraint only valid locally? (default False)
        modifiable -- is the constraint modifiable (subject to column generation)? (default False)
        dynamic -- is the constraint subject to aging? (default False)
        removable -- hould the relaxation be removed from the LP due to aging or cleanup? (default False)
        stickingatnode -- should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (default False)
        """
        if isinstance(coeffs, LinCons):
            kwargs = dict(lhs=lhs, rhs=rhs, name=name,
                          initial=initial, separate=separate, enforce=enforce,
                          check=check, propagate=propagate, local=local,
                          modifiable=modifiable, dynamic=dynamic,
                          removable=removable, stickingatnode=stickingatnode)
            deg = coeffs.expr.degree()
            if deg <= 1:
                return self._addLinCons(coeffs, **kwargs)
            elif deg <= 2:
                return self._addQuadCons(coeffs, **kwargs)
            else:
                raise NotImplementedError('Constraints of degree %d!' % deg)

        if lhs is None:
            lhs = -SCIPinfinity(self._scip)
        if rhs is None:
            rhs = SCIPinfinity(self._scip)
        cdef SCIP_CONS* scip_cons
        self._createConsLinear(&scip_cons, name, 0, NULL, NULL, lhs, rhs,
                                initial, separate, enforce, check, propagate,
                                local, modifiable, dynamic, removable, stickingatnode)
        cdef SCIP_Real coeff
        cdef Var v
        cdef SCIP_VAR* _var
        for k in coeffs:
            coeff = <SCIP_Real>coeffs[k]
            v = <Var>k.var
            _var = <SCIP_VAR*>v._var
            self._addCoefLinear(scip_cons, _var, coeff)
        self._addCons(scip_cons)
        self._releaseCons(scip_cons)

        return pythonizeCons(scip_cons, name)

    def _addLinCons(self, lincons, **kwargs):
        """Add object of class LinCons."""
        assert isinstance(lincons, LinCons)
        kwargs['lhs'], kwargs['rhs'] = lincons.lb, lincons.ub
        terms = lincons.expr.terms
        assert lincons.expr.degree() <= 1
        assert terms[()] == 0.0
        coeffs = {t[0]:c for t, c in terms.items() if c != 0.0}

        return self.addCons(coeffs, **kwargs)

    def _addQuadCons(self, quadcons, **kwargs):
        """Add object of class LinCons."""
        assert isinstance(quadcons, LinCons) # TODO
        kwargs['lhs'] = -SCIPinfinity(self._scip) if quadcons.lb is None else quadcons.lb
        kwargs['rhs'] =  SCIPinfinity(self._scip) if quadcons.ub is None else quadcons.ub
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2
        assert terms[()] == 0.0

        name = str_conversion("quadcons") # TODO

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsQuadratic(
            self._scip, &scip_cons, name,
            0, NULL, NULL,        # linear
            0, NULL, NULL, NULL,  # quadratc
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable']))

        cdef Var var1
        cdef Var var2
        cdef SCIP_VAR* _var1
        cdef SCIP_VAR* _var2
        for v, c in terms.items():
            if len(v) == 0: # constant
                assert c == 0.0
            elif len(v) == 1: # linear
                var1 = <Var>v[0].var
                _var1 = <SCIP_VAR*>var1._var
                PY_SCIP_CALL(SCIPaddLinearVarQuadratic(self._scip, scip_cons, _var1, c))
            else: # quadratic
                assert len(v) == 2, 'term: %s' % v
                var1 = <Var>v[0].var
                _var1 = <SCIP_VAR*>var1._var
                var2 = <Var>v[1].var
                _var2 = <SCIP_VAR*>var2._var
                PY_SCIP_CALL(SCIPaddBilinTermQuadratic(self._scip, scip_cons, _var1, _var2, c))

        self._addCons(scip_cons)
        cons = Cons()
        cons._cons = scip_cons
        return cons

    def addConsCoeff(self, cons, var, coeff):
        """Add coefficient to the linear constraint (if non-zero).

        Keyword arguments:
        cons -- the constraint
        coeff -- the coefficient
        """
        cdef Cons c
        cdef Var v
        c = cons.cons
        v = var.var
        self._addCoefLinear(c._cons, v._var, coeff)

    def addConsSOS1(self, vars, weights=None, name="SOS1cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an SOS1 constraint.

        Keyword arguments:
        vars -- list of variables to be included
        weights -- list of weights (default None)
        name -- the name of the constraint (default 'SOS1cons')
        initial -- should the LP relaxation of constraint be in the initial LP? (default True)
        separate -- should the constraint be separated during LP processing? (default True)
        enforce -- should the constraint be enforced during node processing? (default True)
        check -- should the constraint be checked for feasibility? (default True)
        propagate -- should the constraint be propagated during node processing? (default True)
        local -- is the constraint only valid locally? (default False)
        dynamic -- is the constraint subject to aging? (default False)
        removable -- hould the relaxation be removed from the LP due to aging or cleanup? (default False)
        stickingatnode -- should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (default False)
        """
        cdef SCIP_CONS* scip_cons
        cdef Var v
        cdef SCIP_VAR* _var
        cdef int _nvars

        self._createConsSOS1(&scip_cons, name, 0, NULL, NULL,
                                initial, separate, enforce, check, propagate,
                                local, dynamic, removable, stickingatnode)

        if weights is None:
            for k in vars:
                v = <Var>k.var
                _var = <SCIP_VAR*>v._var
                self._appendVarSOS1(scip_cons, _var)
        else:
            nvars = len(vars)
            for k in range(nvars):
                v = <Var>vars[k].var
                _var = <SCIP_VAR*>v._var
                weight = weights[k]
                self._addVarSOS1(scip_cons, _var, weight)

        self._addCons(scip_cons)
        return pythonizeCons(scip_cons, name)

    def addConsSOS2(self, vars, weights=None, name="SOS2cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an SOS2 constraint.

        Keyword arguments:
        vars -- list of variables to be included
        weights -- list of weights (default None)
        name -- the name of the constraint (default 'SOS2cons')
        initial -- should the LP relaxation of constraint be in the initial LP? (default True)
        separate -- should the constraint be separated during LP processing? (default True)
        enforce -- should the constraint be enforced during node processing? (default True)
        check -- should the constraint be checked for feasibility? (default True)
        propagate -- should the constraint be propagated during node processing? (default True)
        local -- is the constraint only valid locally? (default False)
        dynamic -- is the constraint subject to aging? (default False)
        removable -- hould the relaxation be removed from the LP due to aging or cleanup? (default False)
        stickingatnode -- should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (default False)
        """
        cdef SCIP_CONS* scip_cons
        cdef Var v
        cdef SCIP_VAR* _var
        cdef int _nvars

        self._createConsSOS2(&scip_cons, name, 0, NULL, NULL,
                                initial, separate, enforce, check, propagate,
                                local, dynamic, removable, stickingatnode)

        if weights is None:
            for k in vars:
                v = <Var>k.var
                _var = <SCIP_VAR*>v._var
                self._appendVarSOS2(scip_cons, _var)
        else:
            nvars = len(vars)
            for k in range(nvars):
                v = <Var>vars[k].var
                _var = <SCIP_VAR*>v._var
                weight = weights[k]
                self._addVarSOS2(scip_cons, _var, weight)

        self._addCons(scip_cons)
        return pythonizeCons(scip_cons, name)


    def addVarSOS1(self, cons, var, weight):
        """Add variable to SOS1 constraint.

        Keyword arguments:
        cons -- the SOS1 constraint
        vars -- the variable
        weight -- the weight
        """
        cdef Cons c
        cdef Var v
        c = cons.cons
        v = var.var
        self._addVarSOS1(c._cons, v._var, weight)

    def appendVarSOS1(self, cons, var):
        """Append variable to SOS1 constraint.

        Keyword arguments:
        cons -- the SOS1 constraint
        vars -- the variable
        """
        cdef Cons c
        cdef Var v
        c = cons.cons
        v = var.var
        self._appendVarSOS1(c._cons, v._var)

    def addVarSOS2(self, cons, var, weight):
        """Add variable to SOS2 constraint.

        Keyword arguments:
        cons -- the SOS2 constraint
        vars -- the variable
        weight -- the weight
        """
        cdef Cons c
        cdef Var v
        c = cons.cons
        v = var.var
        self._addVarSOS2(c._cons, v._var, weight)

    def appendVarSOS2(self, cons, var):
        """Append variable to SOS2 constraint.

        Keyword arguments:
        cons -- the SOS2 constraint
        vars -- the variable
        """
        cdef Cons c
        cdef Var v
        c = cons.cons
        v = var.var
        self._appendVarSOS2(c._cons, v._var)

    def getTransformedCons(self, cons):
        """Retrieve transformed constraint.

        Keyword arguments:
        cons -- the constraint
        """
        cdef Cons c
        cdef Cons ctrans
        c = cons.cons
        transcons = Constraint("t-"+cons.name)
        ctrans = transcons.cons

        PY_SCIP_CALL(SCIPtransformCons(self._scip, c._cons, &ctrans._cons))
        return transcons

    def getConss(self):
        """Retrieve all constraints."""
        cdef SCIP_CONS** _conss
        cdef SCIP_CONS* _cons
        cdef Cons c
        cdef int _nconss
        conss = []

        _conss = SCIPgetConss(self._scip)
        _nconss = SCIPgetNConss(self._scip)

        for i in range(_nconss):
            _cons = _conss[i]
            conss.append(pythonizeCons(_cons, SCIPconsGetName(_cons).decode("utf-8")))

        return conss

    def getDualsolLinear(self, cons):
        """Retrieve the dual solution to a linear constraint.

        Keyword arguments:
        cons -- the linear constraint
        """
        cdef Cons c
        c = cons.cons
        return SCIPgetDualsolLinear(self._scip, c._cons)

    def getDualfarkasLinear(self, cons):
        """Retrieve the dual farkas value to a linear constraint.

        Keyword arguments:
        cons -- the linear constraint
        """
        cdef Cons c
        c = cons.cons
        return SCIPgetDualfarkasLinear(self._scip, c._cons)

    def optimize(self):
        """Optimize the problem."""
        PY_SCIP_CALL(SCIPsolve(self._scip))
        self._bestSol = SCIPgetBestSol(self._scip)

    def includePricer(self, Pricer pricer, name, desc, priority=1, delay=True):
        """Include a pricer.

        Keyword arguments:
        pricer -- the pricer
        name -- the name
        desc -- the description
        priority -- priority of the variable pricer
        delay -- should the pricer be delayed until no other pricers or already
                 existing problem variables with negative reduced costs are found?
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePricer(self._scip, n, d,
                                            priority, delay,
                                            PyPricerCopy, PyPricerFree, PyPricerInit, PyPricerExit, PyPricerInitsol, PyPricerExitsol, PyPricerRedcost, PyPricerFarkas,
                                            <SCIP_PRICERDATA*>pricer))
        cdef SCIP_PRICER* scip_pricer
        scip_pricer = SCIPfindPricer(self._scip, n)
        PY_SCIP_CALL(SCIPactivatePricer(self._scip, scip_pricer))
        pricer.model = self


    def includeConshdlr(self, Conshdlr conshdlr, name, desc, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                        maxprerounds, delaysepa, delayprop, needscons, proptiming=SCIP_PROPTIMING_AFTERLPNODE, presoltiming=SCIP_PRESOLTIMING_FAST):
        """Include a constraint handler

        Keyword arguments:
        name -- name of constraint handler
        desc -- description of constraint handler
        sepapriority -- priority of the constraint handler for separation
        enfopriority -- priority of the constraint handler for constraint enforcing
        chckpriority -- priority of the constraint handler for checking feasibility (and propagation)
        sepafreq -- frequency for separating cuts; zero means to separate only in the root node
        propfreq -- frequency for propagating domains; zero means only preprocessing propagation
        eagerfreq -- frequency for using all instead of only the useful constraints in separation,
                     propagation and enforcement, -1 for no eager evaluations, 0 for first only
        maxprerounds -- maximal number of presolving rounds the constraint handler participates in (-1: no limit)
        delaysepa -- should separation method be delayed, if other separators found cuts?
        delayprop -- should propagation method be delayed, if other propagators found reductions?
        needscons -- should the constraint handler be skipped, if no constraints are available?
        proptiming -- positions in the node solving loop where propagation method of constraint handlers should be executed
        presoltiming -- timing mask of the constraint handler's presolving method
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeConshdlr(self._scip, n, d, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                                              maxprerounds, delaysepa, delayprop, needscons, proptiming, presoltiming,
                                              PyConshdlrCopy, PyConsFree, PyConsInit, PyConsExit, PyConsInitpre, PyConsExitpre,
                                              PyConsInitsol, PyConsExitsol, PyConsDelete, PyConsTrans, PyConsInitlp, PyConsSepalp, PyConsSepasol,
                                              PyConsEnfolp, PyConsEnfops, PyConsCheck, PyConsProp, PyConsPresol, PyConsResprop, PyConsLock,
                                              PyConsActive, PyConsDeactive, PyConsEnable, PyConsDisable, PyConsDelvars, PyConsPrint, PyConsCopy,
                                              PyConsParse, PyConsGetvars, PyConsGetnvars, PyConsGetdivebdchgs,
                                              <SCIP_CONSHDLRDATA*>conshdlr))
        conshdlr.model = self
        conshdlr.name = name

    def createCons(self, Conshdlr conshdlr, name, initial=True, separate=True, enforce=True, check=True, propagate=True,
                   local=False, modifiable=False, dynamic=False, removable=False, stickingatnode=False):

        n = str_conversion(name)
        cdef SCIP_CONSHDLR* _conshdlr
        _conshdlr = SCIPfindConshdlr(self._scip, str_conversion(conshdlr.name))
        constraint = Constraint(name)
        cdef SCIP_CONS* _cons
        _cons = <SCIP_CONS*>constraint._constraint
        PY_SCIP_CALL(SCIPcreateCons(self._scip, &_cons, n, _conshdlr, NULL,
                                initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))
        return constraint

    def includePresol(self, Presol presol, name, desc, priority, maxrounds, timing=SCIP_PRESOLTIMING_FAST):
        """Include a presolver

        Keyword arguments:
        name         -- name of presolver
        desc         -- description of presolver
        priority     -- priority of the presolver (>= 0: before, < 0: after constraint handlers)
        maxrounds    -- maximal number of presolving rounds the presolver participates in (-1: no limit)
        timing       -- timing mask of the presolver
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePresol(self._scip, n, d, priority, maxrounds, timing, PyPresolCopy, PyPresolFree, PyPresolInit,
                                            PyPresolExit, PyPresolInitpre, PyPresolExitpre, PyPresolExec, <SCIP_PRESOLDATA*>presol))
        presol.model = self

    def includeSepa(self, Sepa sepa, name, desc, priority, freq, maxbounddist, usessubscip=False, delay=False):
        """Include a separator

        Keyword arguments:
        name         -- name of separator
        desc         -- description of separator
        priority     -- priority of separator (>= 0: before, < 0: after constraint handlers)
        freq         -- frequency for calling separator
        maxbounddist -- maximal relative distance from current node's dual bound to primal bound compared
                        to best node's dual bound for applying separation
        usessubscip  -- does the separator use a secondary SCIP instance?
        delay        -- should separator be delayed, if other separators found cuts?
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeSepa(self._scip, n, d, priority, freq, maxbounddist, usessubscip, delay, PySepaCopy, PySepaFree,
                                          PySepaInit, PySepaExit, PySepaInitsol, PySepaExitsol, PySepaExeclp, PySepaExecsol, <SCIP_SEPADATA*>sepa))
        sepa.model = self

    def includeProp(self, Prop prop, name, desc, presolpriority, presolmaxrounds,
                    proptiming, presoltiming=SCIP_PRESOLTIMING_FAST, priority=1, freq=1, delay=True):
        """Include a propagator.

        Keyword arguments:
        prop -- the propagator
        name -- the name
        desc -- the description
        priority -- priority of the propagator
        freq -- frequency for calling propagator
        delay -- should propagator be delayed if other propagators have found reductions?
        presolpriority -- presolving priority of the propagator (>= 0: before, < 0: after constraint handlers)
        presolmaxrounds --maximal number of presolving rounds the propagator participates in (-1: no limit)
        proptiming -- positions in the node solving loop where propagation method of constraint handlers should be executed
        presoltiming -- timing mask of the constraint handler's presolving method
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeProp(self._scip, n, d,
                                          priority, freq, delay,
                                          proptiming, presolpriority, presolmaxrounds,
                                          presoltiming, PyPropCopy, PyPropFree, PyPropInit, PyPropExit,
                                          PyPropInitpre, PyPropExitpre, PyPropInitsol, PyPropExitsol,
                                          PyPropPresol, PyPropExec, PyPropResProp,
                                          <SCIP_PROPDATA*> prop))
        prop.model = self

    def includeHeur(self, Heur heur, name, desc, dispchar, priority=10000, freq=1, freqofs=0,
                    maxdepth=-1, timingmask=SCIP_HEURTIMING_BEFORENODE, usessubscip=False):
        """Include a primal heuristic.

        Keyword arguments:
        heur -- the heuristic
        name -- the name of the heuristic
        desc -- the description
        dispchar -- display character of primal heuristic
        priority -- priority of the heuristic
        freq -- frequency offset for calling heuristic
        freqofs -- frequency offset for calling heuristic
        maxdepth -- maximal depth level to call heuristic at (-1: no limit)
        timingmask -- positions in the node solving loop where heuristic should be executed; see definition of SCIP_HeurTiming for possible values
        usessubscip -- does the heuristic use a secondary SCIP instance?
        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        dis = ord(str_conversion(dispchar))
        PY_SCIP_CALL(SCIPincludeHeur(self._scip, nam, des, dis,
                                          priority, freq, freqofs,
                                          maxdepth, timingmask, usessubscip,
                                          PyHeurCopy, PyHeurFree, PyHeurInit, PyHeurExit,
                                          PyHeurInitsol, PyHeurExitsol, PyHeurExec,
                                          <SCIP_HEURDATA*> heur))
        heur.model = self
        heur.name = name

    def createSol(self, Heur heur):
        """Create a new primal solution.

        Keyword arguments:
        solution -- the new solution
        heur -- the heuristic that found the solution
        """
        n = str_conversion(heur.name)
        cdef SCIP_HEUR* _heur
        _heur = SCIPfindHeur(self._scip, n)
        solution = Solution()
        PY_SCIP_CALL(SCIPcreateSol(self._scip, &solution._solution, _heur))
        return solution


    def setSolVal(self, Solution solution, variable, val):
        """Set a variable in a solution.

        Keyword arguments:
        solution -- the solution to be modified
        variable -- the variable in the solution
        val -- the value of the variable in the solution
        """
        cdef SCIP_SOL* _sol
        cdef SCIP_VAR* _var
        cdef Var var
        var = <Var>variable.var
        _var = <SCIP_VAR*>var._var
        _sol = <SCIP_SOL*>solution._solution
        PY_SCIP_CALL(SCIPsetSolVal(self._scip, _sol, _var, val))

    def trySol(self, Solution solution, printreason=True, checkbounds=True, checkintegrality=True, checklprows=True):
        """Try to add a solution to the storage.

        Keyword arguments:
        solution -- the solution to store
        printreason -- should all reasons of violations be printed?
        checkbounds -- should the bounds of the variables be checked?
        checkintegrality -- has integrality to be checked?
        checklprows -- have current LP rows (both local and global) to be checked?
        """
        cdef SCIP_Bool stored
        PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution._solution, printreason, checkbounds, checkintegrality, checklprows, &stored))
        return stored

    def includeBranchrule(self, Branchrule branchrule, name, desc, priority, maxdepth, maxbounddist):
        """Include a branching rule.

        Keyword arguments:
        branchrule -- the branching rule
        name -- name of branching rule
        desc --description of branching rule
        priority --priority of the branching rule
        maxdepth -- maximal depth level, up to which this branching rule should be used (or -1)
        maxbounddist -- maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
        """

        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBranchrule(self._scip, nam, des,
                                          maxdepth, maxdepth, maxbounddist,
                                          PyBranchruleCopy, PyBranchruleFree, PyBranchruleInit, PyBranchruleExit,
                                          PyBranchruleInitsol, PyBranchruleExitsol, PyBranchruleExeclp, PyBranchruleExecext,
                                          PyBranchruleExecps, <SCIP_BRANCHRULEDATA*> branchrule))
        branchrule.model = self

    # Solution functions

    def getSols(self):
        """Retrieve list of all feasible primal solutions stored in the solution storage."""
        cdef SCIP_SOL** _sols
        cdef SCIP_SOL* _sol
        _sols = SCIPgetSols(self._scip)
        nsols = SCIPgetNSols(self._scip)
        sols = []

        for i in range(nsols):
            _sol = _sols[i]
            solution = Solution()
            solution._solution = _sol
            sols.append(solution)

        return sols

    def getBestSol(self):
        """Retrieve currently best known feasible primal solution."""
        solution = Solution()
        solution._solution = SCIPgetBestSol(self._scip)
        return solution

    def getSolObjVal(self, Solution solution, original=True):
        """Retrieve the objective value of the solution.

        Keyword arguments:
        solution -- the solution
        original -- retrieve the solution of the original problem? (default True)
        """
        cdef SCIP_SOL* _solution
        _solution = <SCIP_SOL*>solution._solution
        if original:
            objval = SCIPgetSolOrigObj(self._scip, _solution)
        else:
            objval = SCIPgetSolTransObj(self._scip, _solution)
        return objval

    def getObjVal(self, original=True):
        """Retrieve the objective value of value of best solution"""
        if original:
            objval = SCIPgetSolOrigObj(self._scip, self._bestSol)
        else:
            objval = SCIPgetSolTransObj(self._scip, self._bestSol)
        return objval

    # Get best dual bound
    def getDualbound(self):
        """Retrieve the best dual bound."""
        return SCIPgetDualbound(self._scip)

    def getVal(self, var, Solution solution=None):
        """Retrieve the value of the variable in the specified solution. If no solution is specified,
        the best known solution is used.

        Keyword arguments:
        var -- the variable
        solution -- the solution (default None)
        """
        cdef SCIP_SOL* _sol
        if solution is None:
            _sol = self._bestSol
        else:
            _sol = <SCIP_SOL*>solution._solution
        cdef SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <SCIP_VAR*>v._var
        return SCIPgetSolVal(self._scip, _sol, _var)

    def writeName(self, var):
        """Write the name of the variable to the std out."""
        cdef SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <SCIP_VAR*>v._var
        self._writeVarName(_var)

    def getStatus(self):
        """Retrieve solution status."""
        cdef SCIP_STATUS stat = SCIPgetStatus(self._scip)
        if stat == SCIP_STATUS_OPTIMAL:
            return "optimal"
        elif stat == SCIP_STATUS_TIMELIMIT:
            return "timelimit"
        elif stat == SCIP_STATUS_INFEASIBLE:
            return "infeasible"
        elif stat == SCIP_STATUS_UNBOUNDED:
            return "unbounded"
        else:
            return "unknown"

    def getObjectiveSense(self):
        """Retrieve objective sense."""
        cdef SCIP_OBJSENSE sense = SCIPgetObjsense(self._scip)
        if sense == SCIP_OBJSENSE_MAXIMIZE:
            return "maximize"
        elif sense == SCIP_OBJSENSE_MINIMIZE:
            return "minimize"
        else:
            return "unknown"

    # Statistic Methods

    def printStatistics(self):
        """Print statistics."""
        PY_SCIP_CALL(SCIPprintStatistics(self._scip, NULL))

    # Verbosity Methods

    def hideOutput(self, quiet = True):
        """Hide the output.

        Keyword arguments:
        quiet -- hide output? (default True)
        """
        SCIPsetMessagehdlrQuiet(self._scip, quiet)

    # Parameter Methods

    def setBoolParam(self, name, value):
        """Set a boolean-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, value))

    def setIntParam(self, name, value):
        """Set an int-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, value))

    def setLongintParam(self, name, value):
        """Set a long-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, value))

    def setRealParam(self, name, value):
        """Set a real-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, value))

    def setCharParam(self, name, value):
        """Set a char-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, value))

    def setStringParam(self, name, value):
        """Set a string-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, value))

    def readParams(self, file):
        """Read an external parameter file.

        Keyword arguments:
        file -- the file to be read
        """
        absfile = bytes(abspath(file), 'utf-8')
        PY_SCIP_CALL(SCIPreadParams(self._scip, absfile))

    def readProblem(self, file, extension = None):
        """Read a problem instance from an external file.

        Keyword arguments:
        file -- the file to be read
        extension -- specifies extensions (default None)
        """
        absfile = bytes(abspath(file), 'utf-8')
        if extension is None:
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = bytes(extension, 'utf-8')
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, extension))
