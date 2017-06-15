import weakref
from os.path import abspath
import sys

from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free

include "expr.pxi"
include "lp.pxi"

include "branchrule.pxi"
include "conshdlr.pxi"
include "heuristic.pxi"
include "presol.pxi"
include "pricer.pxi"
include "propagator.pxi"
include "sepa.pxi"

# for external user functions use def; for functions used only inside the interface (starting with _) use cdef
# todo: check whether this is currently done like this

if sys.version_info >= (3, 0):
    str_conversion = lambda x:bytes(x,'utf-8')
else:
    str_conversion = lambda x:x

# Mapping the SCIP_RESULT enum to a python class
# This is required to return SCIP_RESULT in the python code
# In __init__.py this is imported as SCIP_RESULT to keep the
# original naming scheme using capital letters
cdef class PY_SCIP_RESULT:
    DIDNOTRUN   = SCIP_DIDNOTRUN
    DELAYED     = SCIP_DELAYED
    DIDNOTFIND  = SCIP_DIDNOTFIND
    FEASIBLE    = SCIP_FEASIBLE
    INFEASIBLE  = SCIP_INFEASIBLE
    UNBOUNDED   = SCIP_UNBOUNDED
    CUTOFF      = SCIP_CUTOFF
    SEPARATED   = SCIP_SEPARATED
    NEWROUND    = SCIP_NEWROUND
    REDUCEDDOM  = SCIP_REDUCEDDOM
    CONSADDED   = SCIP_CONSADDED
    CONSCHANGED = SCIP_CONSCHANGED
    BRANCHED    = SCIP_BRANCHED
    SOLVELP     = SCIP_SOLVELP
    FOUNDSOL    = SCIP_FOUNDSOL
    SUSPENDED   = SCIP_SUSPENDED
    SUCCESS     = SCIP_SUCCESS

cdef class PY_SCIP_PARAMSETTING:
    DEFAULT     = SCIP_PARAMSETTING_DEFAULT
    AGGRESSIVE  = SCIP_PARAMSETTING_AGGRESSIVE
    FAST        = SCIP_PARAMSETTING_FAST
    OFF         = SCIP_PARAMSETTING_OFF

cdef class PY_SCIP_PARAMEMPHASIS:
    DEFAULT      = SCIP_PARAMEMPHASIS_DEFAULT
    CPSOLVER     = SCIP_PARAMEMPHASIS_CPSOLVER
    EASYCIP      = SCIP_PARAMEMPHASIS_EASYCIP
    FEASIBILITY  = SCIP_PARAMEMPHASIS_FEASIBILITY
    HARDLP       = SCIP_PARAMEMPHASIS_HARDLP
    OPTIMALITY   = SCIP_PARAMEMPHASIS_OPTIMALITY
    COUNTER      = SCIP_PARAMEMPHASIS_COUNTER
    #PHASEFEAS    = SCIP_PARAMEMPHASIS_PHASEFEAS
    #PHASEIMPROVE = SCIP_PARAMEMPHASIS_PHASEIMPROVE
    #PHASEPROOF   = SCIP_PARAMEMPHASIS_PHASEPROOF

cdef class PY_SCIP_STATUS:
    UNKNOWN        = SCIP_STATUS_UNKNOWN
    USERINTERRUPT  = SCIP_STATUS_USERINTERRUPT
    NODELIMIT      = SCIP_STATUS_NODELIMIT
    TOTALNODELIMIT = SCIP_STATUS_TOTALNODELIMIT
    STALLNODELIMIT = SCIP_STATUS_STALLNODELIMIT
    TIMELIMIT      = SCIP_STATUS_TIMELIMIT
    MEMLIMIT       = SCIP_STATUS_MEMLIMIT
    GAPLIMIT       = SCIP_STATUS_GAPLIMIT
    SOLLIMIT       = SCIP_STATUS_SOLLIMIT
    BESTSOLLIMIT   = SCIP_STATUS_BESTSOLLIMIT
    RESTARTLIMIT   = SCIP_STATUS_RESTARTLIMIT
    OPTIMAL        = SCIP_STATUS_OPTIMAL
    INFEASIBLE     = SCIP_STATUS_INFEASIBLE
    UNBOUNDED      = SCIP_STATUS_UNBOUNDED
    INFORUNBD      = SCIP_STATUS_INFORUNBD

cdef class PY_SCIP_STAGE:
    STAGE_INIT         = SCIP_STAGE_INIT
    STAGE_PROBLEM      = SCIP_STAGE_PROBLEM
    STAGE_TRANSFORMING = SCIP_STAGE_TRANSFORMING
    STAGE_TRANSFORMED  = SCIP_STAGE_TRANSFORMED
    STAGE_INITPRESOLVE = SCIP_STAGE_INITPRESOLVE
    STAGE_PRESOLVING   = SCIP_STAGE_PRESOLVING
    STAGE_EXITPRESOLVE = SCIP_STAGE_EXITPRESOLVE
    STAGE_PRESOLVED    = SCIP_STAGE_PRESOLVED
    STAGE_INITSOLVE    = SCIP_STAGE_INITSOLVE
    STAGE_SOLVING      = SCIP_STAGE_SOLVING
    STAGE_SOLVED       = SCIP_STAGE_SOLVED
    STAGE_EXITSOLVE    = SCIP_STAGE_EXITSOLVE
    STAGE_FREETRANS    = SCIP_STAGE_FREETRANS
    STAGE_FREE         = SCIP_STAGE_FREE

cdef class PY_SCIP_PROPTIMING:
    BEFORELP     = SCIP_PROPTIMING_BEFORELP
    DURINGLPLOOP = SCIP_PROPTIMING_DURINGLPLOOP
    AFTERLPLOOP  = SCIP_PROPTIMING_AFTERLPLOOP
    AFTERLPNODE  = SCIP_PROPTIMING_AFTERLPNODE

cdef class PY_SCIP_PRESOLTIMING:
    NONE       = SCIP_PRESOLTIMING_NONE
    FAST       = SCIP_PRESOLTIMING_FAST
    MEDIUM     = SCIP_PRESOLTIMING_MEDIUM
    EXHAUSTIVE = SCIP_PRESOLTIMING_EXHAUSTIVE

cdef class PY_SCIP_HEURTIMING:
    BEFORENODE        = SCIP_HEURTIMING_BEFORENODE
    DURINGLPLOOP      = SCIP_HEURTIMING_DURINGLPLOOP
    AFTERLPLOOP       = SCIP_HEURTIMING_AFTERLPLOOP
    AFTERLPNODE       = SCIP_HEURTIMING_AFTERLPNODE
    AFTERPSEUDONODE   = SCIP_HEURTIMING_AFTERPSEUDONODE
    AFTERLPPLUNGE     = SCIP_HEURTIMING_AFTERLPPLUNGE
    AFTERPSEUDOPLUNGE = SCIP_HEURTIMING_AFTERPSEUDOPLUNGE
    DURINGPRICINGLOOP = SCIP_HEURTIMING_DURINGPRICINGLOOP
    BEFOREPRESOL      = SCIP_HEURTIMING_BEFOREPRESOL
    DURINGPRESOLLOOP  = SCIP_HEURTIMING_DURINGPRESOLLOOP
    AFTERPROPLOOP     = SCIP_HEURTIMING_AFTERPROPLOOP

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

cdef class Column:
    """Base class holding a pointer to corresponding SCIP_COL"""
    cdef SCIP_COL* col

    @staticmethod
    cdef create(SCIP_COL* scip_col):
        col = Column()
        col.col = scip_col
        return col

cdef class Row:
    """Base class holding a pointer to corresponding SCIP_ROW"""
    cdef SCIP_ROW* row

    @staticmethod
    cdef create(SCIP_ROW* scip_row):
        row = Row()
        row.row = scip_row
        return row

cdef class Solution:
    """Base class holding a pointer to corresponding SCIP_SOL"""
    cdef SCIP_SOL* sol

    @staticmethod
    cdef create(SCIP_SOL* scip_sol):
        sol = Solution()
        sol.sol = scip_sol
        return sol

cdef class Variable(Expr):
    """Is a linear expression and has SCIP_VAR*"""
    cdef SCIP_VAR* var

    @staticmethod
    cdef create(SCIP_VAR* scipvar):
        var = Variable()
        var.var = scipvar
        Expr.__init__(var, {Term(var) : 1.0})
        return var

    property name:
        def __get__(self):
            cname = bytes( SCIPvarGetName(self.var) )
            return cname.decode('utf-8')

    def ptr(self):
        return <size_t>(self.var)

    def __repr__(self):
        return self.name

    def vtype(self):
        """Return the variables type (BINARY, INTEGER or CONTINUOUS)"""
        vartype = SCIPvarGetType(self.var)
        if vartype == SCIP_VARTYPE_BINARY:
            return "BINARY"
        elif vartype == SCIP_VARTYPE_INTEGER:
            return "INTEGER"
        elif vartype == SCIP_VARTYPE_CONTINUOUS or vartype == SCIP_VARTYPE_IMPLINT:
            return "CONTINUOUS"

    def isOriginal(self):
        """Returns whether the variable belongs to the original problem"""
        return SCIPvarIsOriginal(self.var)

    def isInLP(self):
        """Returns whether the variable is a COLUMN variable that is member of the current LP"""
        return SCIPvarIsInLP(self.var)

    def getCol(self):
        """Returns column of COLUMN variable"""
        cdef SCIP_COL* scip_col
        scip_col = SCIPvarGetCol(self.var)
        return Column.create(scip_col)

    def getLbOriginal(self):
        """Returns original lower bound of variable"""
        return SCIPvarGetLbOriginal(self.var)

    def getUbOriginal(self):
        """Returns original upper bound of variable"""
        return SCIPvarGetUbOriginal(self.var)

    def getLbGlobal(self):
        """Returns global lower bound of variable"""
        return SCIPvarGetLbGlobal(self.var)

    def getUbGlobal(self):
        """Returns global upper bound of variable"""
        return SCIPvarGetUbGlobal(self.var)

    def getLbLocal(self):
        """Returns current lower bound of variable"""
        return SCIPvarGetLbLocal(self.var)

    def getUbLocal(self):
        """Returns current upper bound of variable"""
        return SCIPvarGetUbLocal(self.var)

    def getObj(self):
        """Returns current objective value of variable"""
        return SCIPvarGetObj(self.var)


cdef class Constraint:
    cdef SCIP_CONS* cons
    cdef public object data #storage for python user

    @staticmethod
    cdef create(SCIP_CONS* scipcons):
        if scipcons == NULL:
            raise Warning("cannot create Constraint with SCIP_CONS* == NULL")
        cons = Constraint()
        cons.cons = scipcons
        return cons

    property name:
        def __get__(self):
            cname = bytes( SCIPconsGetName(self.cons) )
            return cname.decode('utf-8')

    def __repr__(self):
        return self.name

    def isOriginal(self):
        """Returns whether the constraint belongs to the original problem"""
        return SCIPconsIsOriginal(self.cons)

    def isInitial(self):
        """Returns True if the relaxation of the constraint should be in the initial LP"""
        return SCIPconsIsInitial(self.cons)

    def isSeparated(self):
        """Returns True if constraint should be separated during LP processing"""
        return SCIPconsIsSeparated(self.cons)

    def isEnforced(self):
        """Returns True if constraint should be enforced during node processing"""
        return SCIPconsIsEnforced(self.cons)

    def isChecked(self):
        """Returns True if conestraint should be checked for feasibility"""
        return SCIPconsIsChecked(self.cons)

    def isPropagated(self):
        """Returns True if constraint should be propagated during node processing"""
        return SCIPconsIsPropagated(self.cons)

    def isLocal(self):
        """Returns True if constraint is only locally valid or not added to any (sub)problem"""
        return SCIPconsIsLocal(self.cons)

    def isModifiable(self):
        """Returns True if constraint is modifiable (subject to column generation)"""
        return SCIPconsIsModifiable(self.cons)

    def isDynamic(self):
        """Returns True if constraint is subject to aging"""
        return SCIPconsIsDynamic(self.cons)

    def isRemovable(self):
        """Returns True if constraint's relaxation should be removed from the LP due to aging or cleanup"""
        return SCIPconsIsRemovable(self.cons)

    def isStickingAtNode(self):
        """Returns True if constraint is only locally valid or not added to any (sub)problem"""
        return SCIPconsIsStickingAtNode(self.cons)

# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
cdef class Model:
    cdef SCIP* _scip
    # store best solution to get the solution values easier
    cdef Solution _bestSol
    # can be used to store problem data
    cdef public object data
    # make Model weak referentiable
    cdef object __weakref__

    def __init__(self, problemName='model', defaultPlugins=True):
        """
        Keyword arguments:
        problemName -- the name of the problem (default 'model')
        defaultPlugins -- use default plugins? (default True)
        """
        self.create()
        self._bestSol = None
        if defaultPlugins:
            self.includeDefaultPlugins()
        self.createProbBasic(problemName)

    def __dealloc__(self):
        # call C function directly, because we can no longer call this object's methods, according to
        # http://docs.cython.org/src/reference/extension_types.html#finalization-dealloc
        PY_SCIP_CALL( SCIPfree(&self._scip) )

    def create(self):
        """Create a new SCIP instance"""
        PY_SCIP_CALL(SCIPcreate(&self._scip))

    def includeDefaultPlugins(self):
        """Includes all default plug-ins into SCIP"""
        PY_SCIP_CALL(SCIPincludeDefaultPlugins(self._scip))

    def createProbBasic(self, problemName='model'):
        """Create new problem iinstance with given name"""
        n = str_conversion(problemName)
        PY_SCIP_CALL(SCIPcreateProbBasic(self._scip, n))

    def freeProb(self):
        """Frees problem and solution process data"""
        PY_SCIP_CALL(SCIPfreeProb(self._scip))

    def freeTransform(self):
        """Frees all solution process data including presolving and transformed problem, only original problem is kept"""
        PY_SCIP_CALL(SCIPfreeTransform(self._scip))

    def printVersion(self):
        """Print version, copyright information and compile mode"""
        SCIPprintVersion(self._scip, NULL)

    def getTotalTime(self):
        """Returns the current total SCIP time in seconds, i.e. the total time since the SCIP instance has been created"""
        return SCIPgetTotalTime(self._scip)

    def getSolvingTime(self):
        """Returns the current solving time in seconds"""
        return SCIPgetSolvingTime(self._scip)

    def getReadingTime(self):
        """Returns the current reading time in seconds"""
        return SCIPgetReadingTime(self._scip)

    def getPresolvingTime(self):
        """Returns the curernt presolving time in seconds"""
        return SCIPgetPresolvingTime(self._scip)

    def getNNodes(self):
        """Retrieve the total number of processed nodes."""
        return SCIPgetNNodes(self._scip)

    def getGap(self):
        """Retrieve the gap, i.e. |(primalbound - dualbound)/min(|primalbound|,|dualbound|)|."""
        return SCIPgetGap(self._scip)

    def infinity(self):
        """Returns SCIP's infinity value"""
        return SCIPinfinity(self._scip)

    def epsilon(self):
        """Return epsilon for e.g. equality checks"""
        return SCIPepsilon(self._scip)

    def feastol(self):
        """Return feasibility tolerance"""
        return SCIPfeastol(self._scip)


    # Objective function

    def setMinimize(self):
        """Set the objective sense to minimization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        """Set the objective sense to maximization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    def setObjlimit(self, objlimit):
        """Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Keyword arguments:
        objlimit -- limit on the objective function
        """
        PY_SCIP_CALL(SCIPsetObjlimit(self._scip, objlimit))

    def setObjective(self, coeffs, sense = 'minimize', clear = 'true'):
        """Establish the objective function as a linear expression.

        Keyword arguments:
        coeffs -- the coefficients
        sense -- the objective sense (default 'minimize')
        clear -- set all other variables objective coefficient to zero (default 'true')
        """
        cdef SCIP_VAR** _vars
        cdef int _nvars
        assert isinstance(coeffs, Expr)

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            raise ValueError("Constant offsets in objective are not supported!")

        if clear:
            # clear existing objective function
            _vars = SCIPgetOrigVars(self._scip)
            _nvars = SCIPgetNOrigVars(self._scip)
            for i in range(_nvars):
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, _vars[i], 0.0))

        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, var.var, coef))

        if sense == "minimize":
            self.setMinimize()
        elif sense == "maximize":
            self.setMaximize()
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

    def getObjective(self):
        """Return objective function as Expr"""
        variables = self.getVars()
        objective = Expr()
        for var in variables:
            coeff = var.getObj()
            if coeff != 0:
                objective += coeff * var
        objective.normalize()
        return objective

    # Setting parameters
    def setPresolve(self, setting):
        """Set presolving parameter settings.

        Keyword arguments:
        setting -- the parameter settings
        """
        PY_SCIP_CALL(SCIPsetPresolving(self._scip, setting, True))

    def setSeparating(self, setting):
        """Set separating parameter settings.

        Keyword arguments:
        setting -- the parameter settings
        """
        PY_SCIP_CALL(SCIPsetSeparating(self._scip, setting, True))

    def setHeuristics(self, setting):
        """Set heuristics parameter settings.

        Keyword arguments:
        setting -- the parameter settings
        """
        PY_SCIP_CALL(SCIPsetHeuristics(self._scip, setting, True))

    def disablePropagation(self, onlyroot=False):
        """Disables propagation in SCIP to avoid modifying the original problem during transformation.

        Keyword arguments:
        onlyroot -- use propagation when root processing is finished
        """
        self.setIntParam("propagating/maxroundsroot", 0)
        if not onlyroot:
            self.setIntParam("propagating/maxrounds", 0)

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
        cname = str_conversion(name)
        if ub is None:
            ub = SCIPinfinity(self._scip)
        cdef SCIP_VAR* scip_var
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS))
        elif vtype in ['B', 'BINARY']:
            lb = 0.0
            ub = 1.0
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_BINARY))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_INTEGER))
        else:
            raise Warning("unrecognized variable type")

        if pricedVar:
            PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, 1.0))
        else:
            PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

        pyVar = Variable.create(scip_var)
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &scip_var))
        return pyVar

    def releaseVar(self, Variable var):
        """Release the variable.

        Keyword arguments:
        var -- the variable
        """
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &var.var))

    def getTransformedVar(self, Variable var):
        """Retrieve the transformed variable.

        Keyword arguments:
        var -- the variable
        """
        cdef SCIP_VAR* _tvar
        PY_SCIP_CALL(SCIPtransformVar(self._scip, var.var, &_tvar))
        return Variable.create(_tvar)

    def addVarLocks(self, Variable var, nlocksdown, nlocksup):
        """adds given values to lock numbers of variable for rounding

        Keyword arguments:
        var -- the variable to adjust the locks for
        nlocksdown -- modification number of down locks
        nlocksup -- modification number of up locks
        """
        PY_SCIP_CALL(SCIPaddVarLocks(self._scip, var.var, nlocksdown, nlocksup))

    def chgVarLb(self, Variable var, lb=None):
        """Changes the lower bound of the specified variable.

        Keyword arguments:
        var -- the variable
        lb -- the lower bound (default None)
        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLb(self._scip, var.var, lb))

    def chgVarUb(self, Variable var, ub=None):
        """Changes the upper bound of the specified variable.

        Keyword arguments:
        var -- the variable
        ub -- the upper bound (default None)
        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUb(self._scip, var.var, ub))

    def chgVarType(self, Variable var, vtype):
        cdef SCIP_Bool infeasible
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.var, SCIP_VARTYPE_CONTINUOUS, &infeasible))
        elif vtype in ['B', 'BINARY']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.var, SCIP_VARTYPE_BINARY, &infeasible))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.var, SCIP_VARTYPE_INTEGER, &infeasible))
        else:
            raise Warning("unrecognized variable type")
        if infeasible:
            print('could not change variable type of variable %s' % var)

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

        return [Variable.create(_vars[i]) for i in range(_nvars)]

    # Constraint functions
    def addCons(self, cons, name="cons", initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """Add a linear or quadratic constraint.

        Keyword arguments:
        cons -- list of coefficients
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
        assert isinstance(cons, ExprCons)
        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode)
        kwargs['lhs'] = -SCIPinfinity(self._scip) if cons.lhs is None else cons.lhs
        kwargs['rhs'] =  SCIPinfinity(self._scip) if cons.rhs is None else cons.rhs

        deg = cons.expr.degree()
        if deg <= 1:
            return self._addLinCons(cons, **kwargs)
        elif deg <= 2:
            return self._addQuadCons(cons, **kwargs)
        else:
            return self._addNonlinearCons(cons, **kwargs)

    def _addLinCons(self, ExprCons lincons, **kwargs):
        """Add object of class ExprCons."""
        assert isinstance(lincons, ExprCons)

        assert lincons.expr.degree() <= 1
        terms = lincons.expr.terms

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsLinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']), 0, NULL, NULL,
            kwargs['lhs'], kwargs['rhs'], kwargs['initial'],
            kwargs['separate'], kwargs['enforce'], kwargs['check'],
            kwargs['propagate'], kwargs['local'], kwargs['modifiable'],
            kwargs['dynamic'], kwargs['removable'], kwargs['stickingatnode']))

        for key, coeff in terms.items():
            var = <Variable>key[0]
            PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, scip_cons, var.var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return PyCons

    def _addQuadCons(self, ExprCons quadcons, **kwargs):
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsQuadratic(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL,        # linear
            0, NULL, NULL, NULL,  # quadratc
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable']))

        for v, c in terms.items():
            if len(v) == 1: # linear
                var = <Variable>v[0]
                PY_SCIP_CALL(SCIPaddLinearVarQuadratic(self._scip, scip_cons, var.var, c))
            else: # quadratic
                assert len(v) == 2, 'term: %s' % v
                var1, var2 = <Variable>v[0], <Variable>v[1]
                PY_SCIP_CALL(SCIPaddBilinTermQuadratic(self._scip, scip_cons, var1.var, var2.var, c))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        return PyCons

    def _addNonlinearCons(self, ExprCons cons, **kwargs):
        """Add object of class ExprCons."""
        cdef SCIP_EXPR* expr
        cdef SCIP_EXPR** varexprs
        cdef SCIP_EXPRDATA_MONOMIAL** monomials
        cdef int* idxs
        cdef SCIP_EXPRTREE* exprtree
        cdef SCIP_VAR** vars
        cdef SCIP_CONS* scip_cons

        terms = cons.expr.terms

        # collect variables
        variables = {var.ptr():var for term in terms for var in term}
        variables = list(variables.values())
        varindex = {var.ptr():idx for (idx,var) in enumerate(variables)}

        # create variable expressions
        varexprs = <SCIP_EXPR**> malloc(len(varindex) * sizeof(SCIP_EXPR*))
        for idx in varindex.values():
            PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &expr, SCIP_EXPR_VARIDX, <int>idx) )
            varexprs[idx] = expr

        # create monomials for terms
        monomials = <SCIP_EXPRDATA_MONOMIAL**> malloc(len(terms) * sizeof(SCIP_EXPRDATA_MONOMIAL*))
        for i, (term, coef) in enumerate(terms.items()):
            idxs = <int*> malloc(len(term) * sizeof(int))
            for j, var in enumerate(term):
                idxs[j] = varindex[var.ptr()]
            PY_SCIP_CALL( SCIPexprCreateMonomial(SCIPblkmem(self._scip), &monomials[i], <SCIP_Real>coef, <int>len(term), idxs, NULL) );
            free(idxs)

        # create polynomial from monomials
        PY_SCIP_CALL( SCIPexprCreatePolynomial(SCIPblkmem(self._scip), &expr,
                                               <int>len(varindex), varexprs,
                                               <int>len(terms), monomials, 0.0, <SCIP_Bool>True) );

        # create expression tree
        PY_SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(self._scip), &exprtree, expr, <int>len(variables), 0, NULL) );
        vars = <SCIP_VAR**> malloc(len(variables) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(variables): # same as varindex
            vars[idx] = (<Variable>var).var
        PY_SCIP_CALL( SCIPexprtreeSetVars(exprtree, <int>len(variables), vars) );

        # create nonlinear constraint for exprtree
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL, # linear
            1, &exprtree, NULL, # nonlinear
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable'],
            kwargs['stickingatnode']) )
        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        PY_SCIP_CALL( SCIPexprtreeFree(&exprtree) )
        free(vars)
        free(monomials)
        free(varexprs)
        return PyCons

    def addConsCoeff(self, Constraint cons, Variable var, coeff):
        """Add coefficient to the linear constraint (if non-zero).

        Keyword arguments:
        cons -- the constraint
        coeff -- the coefficient
        """
        PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, cons.cons, var.var, coeff))

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
        cdef int _nvars

        PY_SCIP_CALL(SCIPcreateConsSOS1(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, scip_cons, var.var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, scip_cons, var.var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        return Constraint.create(scip_cons)

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
        cdef int _nvars

        PY_SCIP_CALL(SCIPcreateConsSOS2(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, scip_cons, var.var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, scip_cons, var.var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        return Constraint.create(scip_cons)

    def addConsCardinality(self, consvars, cardval, indvars=None, weights=None, name="CardinalityCons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add a cardinality constraint that allows at most 'cardval' many nonzero variables.

        Keyword arguments:
        consvars -- list of variables to be included
        cardval -- nonnegative integer
        indvars -- indicator variables indicating which variables may be treated as nonzero in cardinality constraint, or None
                   if new indicator variables should be introduced automatically
        weights -- weights determining the variable order, or None if variables should be ordered in the same way they were added to the constraint
        name -- the name of the constraint (default 'CardinalityCons')
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
        cdef SCIP_VAR* indvar

        PY_SCIP_CALL(SCIPcreateConsCardinality(self._scip, &scip_cons, str_conversion(name), 0, NULL, cardval, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        # circumvent an annoying bug in SCIP 4.0.0 that does not allow uninitialized weights
        if weights is None:
            weights = list(range(1, len(consvars) + 1))

        for i, v in enumerate(consvars):
            var = <Variable>v
            if indvars:
                indvar = (<Variable>indvars[i]).var
            else:
                indvar = NULL
            if weights is None:
                PY_SCIP_CALL(SCIPappendVarCardinality(self._scip, scip_cons, var.var, indvar))
            else:
                PY_SCIP_CALL(SCIPaddVarCardinality(self._scip, scip_cons, var.var, indvar, <SCIP_Real>weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons


    def addConsIndicator(self, cons, binvar=None, name="CardinalityCons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an indicator constraint for the linear inequality 'cons'.

        The 'binvar' argument models the redundancy of the linear constraint. A solution for which
        'binvar' is 1 must satisfy the constraint.

        Keyword arguments:
        cons -- a linear inequality of the form "<="
        binvar -- binary indicator variable, or None if it should be created
        name -- the name of the constraint (default 'CardinalityCons')
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
        assert isinstance(cons, ExprCons)
        cdef SCIP_CONS* scip_cons
        cdef SCIP_VAR* _binVar
        if cons.lhs is not None and cons.rhs is not None:
            raise ValueError("expected inequality that has either only a left or right hand side")

        if cons.expr.degree() > 1:
            raise ValueError("expected linear inequality, expression has degree %d" % cons.expr.degree)

        assert cons.expr.degree() <= 1

        if cons.rhs is not None:
            rhs =  cons.rhs
            negate = False
        else:
            rhs = -cons.lhs
            negate = True

        _binVar = (<Variable>binvar).var if binvar is not None else NULL

        PY_SCIP_CALL(SCIPcreateConsIndicator(self._scip, &scip_cons, str_conversion(name), _binVar, 0, NULL, NULL, rhs,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))
        terms = cons.expr.terms

        for key, coeff in terms.items():
            var = <Variable>key[0]
            if negate:
                coeff = -coeff
            PY_SCIP_CALL(SCIPaddVarIndicator(self._scip, scip_cons, var.var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addPyCons(self, Constraint cons):
        """Adds a customly created cons.

        Keyword arguments:
        cons -- the Python constraint
        """
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons.cons))
        Py_INCREF(cons)

    def addVarSOS1(self, Constraint cons, Variable var, weight):
        """Add variable to SOS1 constraint.

        Keyword arguments:
        cons -- the SOS1 constraint
        vars -- the variable
        weight -- the weight
        """
        PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, cons.cons, var.var, weight))

    def appendVarSOS1(self, Constraint cons, Variable var):
        """Append variable to SOS1 constraint.

        Keyword arguments:
        cons -- the SOS1 constraint
        vars -- the variable
        """
        PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, cons.cons, var.var))

    def addVarSOS2(self, Constraint cons, Variable var, weight):
        """Add variable to SOS2 constraint.

        Keyword arguments:
        cons -- the SOS2 constraint
        vars -- the variable
        weight -- the weight
        """
        PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, cons.cons, var.var, weight))

    def appendVarSOS2(self, Constraint cons, Variable var):
        """Append variable to SOS2 constraint.

        Keyword arguments:
        cons -- the SOS2 constraint
        vars -- the variable
        """
        PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, cons.cons, var.var))

    def chgRhs(self, Constraint cons, rhs):
        """Change right hand side value of a constraint.

        Keyword arguments:
        cons -- linear or quadratic constraint
        rhs -- new right hand side
        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgRhsLinear(self._scip, cons.cons, rhs))
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPchgRhsQuadratic(self._scip, cons.cons, rhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def chgLhs(self, Constraint cons, lhs):
        """Change left hand side value of a constraint.

        Keyword arguments:
        cons -- linear or quadratic constraint
        lhs -- new left hand side
        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgLhsLinear(self._scip, cons.cons, lhs))
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPchgLhsQuadratic(self._scip, cons.cons, lhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getTransformedCons(self, Constraint cons):
        """Retrieve transformed constraint.

        Keyword arguments:
        cons -- the constraint
        """
        cdef SCIP_CONS* transcons
        PY_SCIP_CALL(SCIPgetTransformedCons(self._scip, cons.cons, &transcons))
        return Constraint.create(transcons)

    def getConss(self):
        """Retrieve all constraints."""
        cdef SCIP_CONS** _conss
        cdef SCIP_CONS* _cons
        cdef int _nconss
        conss = []

        _conss = SCIPgetConss(self._scip)
        _nconss = SCIPgetNConss(self._scip)
        return [Constraint.create(_conss[i]) for i in range(_nconss)]

    def delCons(self, Constraint cons):
        """Delete constraint from the model

        Keyword arguments:
        cons -- constraint to be deleted
        """
        PY_SCIP_CALL(SCIPdelCons(self._scip, cons.cons))

    def delConsLocal(self, Constraint cons):
        """Delete constraint from the current node and it's children

        Keyword arguments:
        cons -- constraint to be deleted
        """
        PY_SCIP_CALL(SCIPdelConsLocal(self._scip, cons.cons))

    def getDualsolLinear(self, Constraint cons):
        """Retrieve the dual solution to a linear constraint.

        Keyword arguments:
        cons -- the linear constraint
        """
        # TODO this should ideally be handled on the SCIP side
        dual = None
        try:
            if cons.isOriginal():
                transcons = <Constraint>self.getTransformedCons(cons)
                dual = SCIPgetDualsolLinear(self._scip, transcons.cons)
            else:
                dual = SCIPgetDualsolLinear(self._scip, cons.cons)
            if self.getObjectiveSense() == "maximize":
                dual = -dual
        except:
            raise Warning("no dual solution available for constraint " + cons.name)
        return dual

    def getDualfarkasLinear(self, Constraint cons):
        """Retrieve the dual farkas value to a linear constraint.

        Keyword arguments:
        cons -- the linear constraint
        """
        # TODO this should ideally be handled on the SCIP side
        if cons.isOriginal():
            transcons = <Constraint>self.getTransformedCons(cons)
            return SCIPgetDualfarkasLinear(self._scip, transcons.cons)
        else:
            return SCIPgetDualfarkasLinear(self._scip, cons.cons)

    def getVarRedcost(self, Variable var):
        """Retrieve the reduced cost of a variable.

        Keyword arguments:
        var -- variable to get the reduced cost of
        """
        redcost = None
        try:
            redcost = SCIPgetVarRedcost(self._scip, var.var)
            if self.getObjectiveSense() == "maximize":
                redcost = -redcost
        except:
            raise Warning("no reduced cost available for variable " + var.name)
        return redcost

    def optimize(self):
        """Optimize the problem."""
        PY_SCIP_CALL(SCIPsolve(self._scip))
        self._bestSol = Solution.create(SCIPgetBestSol(self._scip))

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
        pricer.model = <Model>weakref.proxy(self)
        Py_INCREF(pricer)

    def includeConshdlr(self, Conshdlr conshdlr, name, desc, sepapriority=0,
                        enfopriority=0, chckpriority=0, sepafreq=-1, propfreq=-1,
                        eagerfreq=100, maxprerounds=-1, delaysepa=False,
                        delayprop=False, needscons=True,
                        proptiming=PY_SCIP_PROPTIMING.BEFORELP,
                        presoltiming=PY_SCIP_PRESOLTIMING.MEDIUM):
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
                                              PyConsEnfolp, PyConsEnforelax, PyConsEnfops, PyConsCheck, PyConsProp, PyConsPresol, PyConsResprop, PyConsLock,
                                              PyConsActive, PyConsDeactive, PyConsEnable, PyConsDisable, PyConsDelvars, PyConsPrint, PyConsCopy,
                                              PyConsParse, PyConsGetvars, PyConsGetnvars, PyConsGetdivebdchgs,
                                              <SCIP_CONSHDLRDATA*>conshdlr))
        conshdlr.model = <Model>weakref.proxy(self)
        conshdlr.name = name
        Py_INCREF(conshdlr)

    def createCons(self, Conshdlr conshdlr, name, initial=True, separate=True, enforce=True, check=True, propagate=True,
                   local=False, modifiable=False, dynamic=False, removable=False, stickingatnode=False):

        n = str_conversion(name)
        cdef SCIP_CONSHDLR* scip_conshdlr
        scip_conshdlr = SCIPfindConshdlr(self._scip, str_conversion(conshdlr.name))
        constraint = Constraint()
        PY_SCIP_CALL(SCIPcreateCons(self._scip, &(constraint.cons), n, scip_conshdlr, <SCIP_CONSDATA*>constraint,
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
        presol.model = <Model>weakref.proxy(self)
        Py_INCREF(presol)

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
        sepa.model = <Model>weakref.proxy(self)
        Py_INCREF(sepa)

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
        prop.model = <Model>weakref.proxy(self)
        Py_INCREF(prop)

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
        heur.model = <Model>weakref.proxy(self)
        heur.name = name
        Py_INCREF(heur)

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
        PY_SCIP_CALL(SCIPcreateSol(self._scip, &solution.sol, _heur))
        return solution

    def setSolVal(self, Solution solution, Variable var, val):
        """Set a variable in a solution.

        Keyword arguments:
        solution -- the solution to be modified
        var -- the variable in the solution
        val -- the value of the variable in the solution
        """
        cdef SCIP_SOL* _sol
        _sol = <SCIP_SOL*>solution.sol
        PY_SCIP_CALL(SCIPsetSolVal(self._scip, _sol, var.var, val))

    def trySol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True):
        """Try to add a solution to the storage.

        Keyword arguments:
        solution -- the solution to store
        printreason -- should all reasons of violations be printed?
        completely -- should all violation be checked?
        checkbounds -- should the bounds of the variables be checked?
        checkintegrality -- has integrality to be checked?
        checklprows -- have current LP rows (both local and global) to be checked?
        """
        cdef SCIP_Bool stored
        PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
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
        branchrule.model = <Model>weakref.proxy(self)
        Py_INCREF(branchrule)

    # Solution functions

    def getSols(self):
        """Retrieve list of all feasible primal solutions stored in the solution storage."""
        cdef SCIP_SOL** _sols
        cdef SCIP_SOL* _sol
        _sols = SCIPgetSols(self._scip)
        nsols = SCIPgetNSols(self._scip)
        sols = []

        for i in range(nsols):
            sols.append(Solution.create(_sols[i]))

        return sols

    def getBestSol(self):
        """Retrieve currently best known feasible primal solution."""
        self._bestSol = Solution.create(SCIPgetBestSol(self._scip))
        return self._bestSol

    def printBestSol(self):
        """Prints the best feasible primal solution."""
        PY_SCIP_CALL(SCIPprintBestSol(self._scip, NULL, False));

    def getSolObjVal(self, Solution sol, original=True):
        """Retrieve the objective value of the solution.

        Keyword arguments:
        sol -- the solution
        original -- objective value in original or transformed space (default True)
        """
        if sol == None:
            sol = Solution.create(NULL)
        if original:
            objval = SCIPgetSolOrigObj(self._scip, sol.sol)
        else:
            objval = SCIPgetSolTransObj(self._scip, sol.sol)
        return objval

    def getObjVal(self, original=True):
        """Retrieve the objective value of value of best solution.
        Can only be called after solving is completed.

        Keyword arguments:
        original -- objective value in original or transformed space (default True)
        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        return self.getSolObjVal(self._bestSol, original)

    def getSolVal(self, Solution sol, Variable var):
        """Retrieve value of given variable in the given solution or in
        the LP/pseudo solution if sol == None

        Keyword arguments:
        sol -- the solution
        var -- the variable to query the value of
        """
        if sol == None:
            sol = Solution.create(NULL)
        return SCIPgetSolVal(self._scip, sol.sol, var.var)

    def getVal(self, Variable var):
        """Retrieve the value of the best known solution.
        Can only be called after solving is completed.

        Keyword arguments:
        var -- the variable to query the value of
        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        return self.getSolVal(self._bestSol, var)

    def getPrimalbound(self):
        """Retrieve the best primal bound."""
        return SCIPgetPrimalbound(self._scip)

    def getDualbound(self):
        """Retrieve the best dual bound."""
        return SCIPgetDualbound(self._scip)

    def getDualboundRoot(self):
        """Retrieve the best root dual bound."""
        return SCIPgetDualboundRoot(self._scip)

    def writeName(self, Variable var):
        """Write the name of the variable to the std out."""
        PY_SCIP_CALL(SCIPwriteVarName(self._scip, NULL, var.var, False))

    def getStage(self):
        """Return current SCIP stage"""
        return SCIPgetStage(self._scip)

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
        absfile = str_conversion(abspath(file))
        PY_SCIP_CALL(SCIPreadParams(self._scip, absfile))

    def setEmphasis(self, paraemphasis, quiet = True):
        """Set emphasis settings

        Keyword arguments:
        paraemphasis -- emphasis to set
        quiet -- hide output? (default True)
        """
        PY_SCIP_CALL(SCIPsetEmphasis(self._scip, paraemphasis, quiet))

    def readProblem(self, file, extension = None):
        """Read a problem instance from an external file.

        Keyword arguments:
        file -- the file to be read
        extension -- specifies extensions (default None)
        """
        absfile = str_conversion(abspath(file))
        if extension is None:
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = str_conversion(extension)
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, extension))

    # Counting functions

    def count(self):
        """Counts the number of feasible points of problem."""
        PY_SCIP_CALL(SCIPcount(self._scip))

    def getNCountedSols(self):
        """Get number of feasible solution."""
        cdef SCIP_Bool valid
        cdef SCIP_Longint nsols

        nsols = SCIPgetNCountedSols(self._scip, &valid)
        if not valid:
            print('total number of solutions found is not valid!')
        return nsols

    def setParamsCountsols(self):
        """sets SCIP parameters such that a valid counting process is possible."""
        PY_SCIP_CALL(SCIPsetParamsCountsols(self._scip))

# debugging memory management
def is_memory_freed():
    return BMSgetMemoryUsed() == 0

def print_memory_in_use():
    BMScheckEmptyMemory()
