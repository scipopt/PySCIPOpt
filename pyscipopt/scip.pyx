# Copyright (C) 2012-2013 Robert Schwarz
#   see file 'LICENSE' for details.

from os.path import abspath
import sys

cimport pyscipopt.scip as scip
from pyscipopt.linexpr import LinExpr, LinCons

include "reader.pxi"
include "pricer.pxi"
include "conshdlr.pxi"
include "presol.pxi"
include "sepa.pxi"
include "propagator.pxi"
include "heuristic.pxi"
include "nodesel.pxi"
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
cdef class scip_result:
    didnotrun   =   1
    delayed     =   2
    didnotfind  =   3
    feasible    =   4
    infeasible  =   5
    unbounded   =   6
    cutoff      =   7
    separated   =   8
    newround    =   9
    reducedom   =  10
    consadded   =  11
    consshanged =  12
    branched    =  13
    solvelp     =  14
    foundsol    =  15
    suspended   =  16
    success     =  17


cdef class scip_paramsetting:
    default     = 0
    agressive   = 1
    fast        = 2
    off         = 3

cdef class scip_status:
    unknown        =  0
    userinterrupt  =  1
    nodelimit      =  2
    totalnodelimit =  3
    stallnodelimit =  4
    timelimit      =  5
    memlimit       =  6
    gaplimit       =  7
    sollimit       =  8
    bestsollimit   =  9
    restartlimit   = 10
    optimal        = 11
    infeasible     = 12
    unbounded      = 13
    inforunbd      = 14

def PY_SCIP_CALL(scip.SCIP_RETCODE rc):
    if rc == scip.SCIP_OKAY:
        pass
    elif rc == scip.SCIP_ERROR:
        raise Exception('SCIP: unspecified error!')
    elif rc == scip.SCIP_NOMEMORY:
        raise MemoryError('SCIP: insufficient memory error!')
    elif rc == scip.SCIP_READERROR:
        raise IOError('SCIP: read error!')
    elif rc == scip.SCIP_WRITEERROR:
        raise IOError('SCIP: write error!')
    elif rc == scip.SCIP_NOFILE:
        raise IOError('SCIP: file not found error!')
    elif rc == scip.SCIP_FILECREATEERROR:
        raise IOError('SCIP: cannot create file!')
    elif rc == scip.SCIP_LPERROR:
        raise Exception('SCIP: error in LP solver!')
    elif rc == scip.SCIP_NOPROBLEM:
        raise Exception('SCIP: no problem exists!')
    elif rc == scip.SCIP_INVALIDCALL:
        raise Exception('SCIP: method cannot be called at this time'
                            + ' in solution process!')
    elif rc == scip.SCIP_INVALIDDATA:
        raise Exception('SCIP: error in input data!')
    elif rc == scip.SCIP_INVALIDRESULT:
        raise Exception('SCIP: method returned an invalid result code!')
    elif rc == scip.SCIP_PLUGINNOTFOUND:
        raise Exception('SCIP: a required plugin was not found !')
    elif rc == scip.SCIP_PARAMETERUNKNOWN:
        raise KeyError('SCIP: the parameter with the given name was not found!')
    elif rc == scip.SCIP_PARAMETERWRONGTYPE:
        raise LookupError('SCIP: the parameter is not of the expected type!')
    elif rc == scip.SCIP_PARAMETERWRONGVAL:
        raise ValueError('SCIP: the value is invalid for the given parameter!')
    elif rc == scip.SCIP_KEYALREADYEXISTING:
        raise KeyError('SCIP: the given key is already existing in table!')
    elif rc == scip.SCIP_MAXDEPTHLEVEL:
        raise Exception('SCIP: maximal branching depth level exceeded!')
    else:
        raise Exception('SCIP: unknown return code!')
    return rc

cdef class Solution:
    cdef scip.SCIP_SOL* _solution


cdef class Var:
    """Base class holding a pointer to corresponding SCIP_VAR"""
    cdef scip.SCIP_VAR* _var


class Variable(LinExpr):
    """Is a linear expression and has SCIP_VAR*"""
    def __init__(self, name=None):
        self.var = Var()
        self.name = name
        LinExpr.__init__(self, {(self,) : 1.0})

    def __hash__(self):
        return hash(id(self))

    def __lt__(self, other):
        return id(self) < id(other)

    def __gt__(self, other):
        return id(self) > id(other)

    def __repr__(self):
        return self.name

cdef pythonizeVar(scip.SCIP_VAR* scip_var, name):
    var = Variable(name)
    cdef Var v
    v = var.var
    v._var = scip_var
    return var

cdef class Cons:
    cdef scip.SCIP_CONS* _cons

class Constraint:
    def __init__(self, name=None):
        self.cons = Cons()
        self.name = name

cdef pythonizeCons(scip.SCIP_CONS* scip_cons, name):
    cons = Constraint(name)
    cdef Cons c
    c = cons.cons
    c._cons = scip_cons
    return cons

# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
cdef class Model:
    cdef scip.SCIP* _scip
    # store best solution to get the solution values easier
    cdef scip.SCIP_SOL* _bestSol
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
        return scip.SCIPcreate(&self._scip)

    @scipErrorHandler
    def includeDefaultPlugins(self):
        return scip.SCIPincludeDefaultPlugins(self._scip)

    @scipErrorHandler
    def createProbBasic(self, problemName='model'):
        n = str_conversion(problemName)
        return scip.SCIPcreateProbBasic(self._scip, n)

    @scipErrorHandler
    def free(self):
        return scip.SCIPfree(&self._scip)

    @scipErrorHandler
    def freeProb(self):
        return scip.SCIPfreeProb(self._scip)

    @scipErrorHandler
    def freeTransform(self):
        return scip.SCIPfreeTransform(self._scip)

    def printVersion(self):
        scip.SCIPprintVersion(self._scip, NULL)

    #@scipErrorHandler       We'll be able to use decorators when we
    #                        interface the relevant classes (SCIP_VAR, ...)
    cdef _createVarBasic(self, scip.SCIP_VAR** scip_var, name,
                        lb, ub, obj, scip.SCIP_VARTYPE varType):
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, scip_var,
                           n, lb, ub, obj, varType))

    cdef _addVar(self, scip.SCIP_VAR* scip_var):
        PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

    cdef _addPricedVar(self, scip.SCIP_VAR* scip_var):
        PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, 1.0))

    cdef _createConsLinear(self, scip.SCIP_CONS** cons, name, nvars,
                                SCIP_VAR** vars, SCIP_Real* vals, lhs, rhs,
                                initial=True, separate=True, enforce=True, check=True,
                                propagate=True, local=False, modifiable=False, dynamic=False,
                                removable=False, stickingatnode=False):
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPcreateConsLinear(self._scip, cons,
                                                    n, nvars, vars, vals,
                                                    lhs, rhs, initial, separate, enforce,
                                                    check, propagate, local, modifiable,
                                                    dynamic, removable, stickingatnode) )

    cdef _createConsSOS1(self, scip.SCIP_CONS** cons, name, nvars,
                              SCIP_VAR** vars, SCIP_Real* weights,
                              initial=True, separate=True, enforce=True, check=True,
                              propagate=True, local=False, dynamic=False, removable=False,
                              stickingatnode=False):
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPcreateConsSOS1(self._scip, cons,
                                                    n, nvars, vars, weights,
                                                    initial, separate, enforce,
                                                    check, propagate, local, dynamic, removable,
                                                    stickingatnode) )

    cdef _createConsSOS2(self, scip.SCIP_CONS** cons, name, nvars,
                              SCIP_VAR** vars, SCIP_Real* weights,
                              initial=True, separate=True, enforce=True, check=True,
                              propagate=True, local=False, dynamic=False, removable=False,
                              stickingatnode=False):
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPcreateConsSOS2(self._scip, cons,
                                                    n, nvars, vars, weights,
                                                    initial, separate, enforce,
                                                    check, propagate, local, dynamic, removable,
                                                    stickingatnode) )

    cdef _addCoefLinear(self, scip.SCIP_CONS* cons, SCIP_VAR* var, val):
        PY_SCIP_CALL(scip.SCIPaddCoefLinear(self._scip, cons, var, val))

    cdef _addCons(self, scip.SCIP_CONS* cons):
        PY_SCIP_CALL(scip.SCIPaddCons(self._scip, cons))

    cdef _addVarSOS1(self, scip.SCIP_CONS* cons, SCIP_VAR* var, weight):
        PY_SCIP_CALL(scip.SCIPaddVarSOS1(self._scip, cons, var, weight))

    cdef _appendVarSOS1(self, scip.SCIP_CONS* cons, SCIP_VAR* var):
        PY_SCIP_CALL(scip.SCIPappendVarSOS1(self._scip, cons, var))

    cdef _addVarSOS2(self, scip.SCIP_CONS* cons, SCIP_VAR* var, weight):
        PY_SCIP_CALL(scip.SCIPaddVarSOS2(self._scip, cons, var, weight))

    cdef _appendVarSOS2(self, scip.SCIP_CONS* cons, SCIP_VAR* var):
        PY_SCIP_CALL(scip.SCIPappendVarSOS2(self._scip, cons, var))

    cdef _writeVarName(self, scip.SCIP_VAR* var):
        PY_SCIP_CALL(scip.SCIPwriteVarName(self._scip, NULL, var, False))

    cdef _releaseVar(self, scip.SCIP_VAR* var):
        PY_SCIP_CALL(scip.SCIPreleaseVar(self._scip, &var))

    cdef _releaseCons(self, scip.SCIP_CONS* cons):
        PY_SCIP_CALL(scip.SCIPreleaseCons(self._scip, &cons))


    # Objective function

    def setMinimize(self):
        """Set the objective sense to maximization."""
        PY_SCIP_CALL(scip.SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        """Set the objective sense to minimization."""
        PY_SCIP_CALL(scip.SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    def setObjlimit(self, objlimit):
        """Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Keyword arguments:
        objlimit -- limit on the objective function
        """
        PY_SCIP_CALL(scip.SCIPsetObjlimit(self._scip, objlimit))

    def setObjective(self, coeffs, sense = 'minimize'):
        """Establish the objective function, either as a variable dictionary or as a linear expression.

        Keyword arguments:
        coeffs -- the coefficients
        sense -- the objective sense (default 'minimize')
        """
        cdef scip.SCIP_Real coeff
        cdef Var v
        cdef scip.SCIP_VAR* _var
        if isinstance(coeffs, LinExpr):
            # transform linear expression into variable dictionary
            terms = coeffs.terms
            coeffs = {t[0]:c for t, c in terms.items() if c != 0.0}
        elif coeffs == 0:
            coeffs = {}
        for k in coeffs:
            coeff = <scip.SCIP_Real>coeffs[k]
            v = <Var>k.var
            _var = <scip.SCIP_VAR*>v._var
            PY_SCIP_CALL(scip.SCIPchgVarObj(self._scip, _var, coeff))
        if sense == 'maximize':
            self.setMaximize();
        else:
            self.setMinimize();

    # Setting parameters
    def setPresolve(self, setting):
        """Set presolving parameter settings.

        Keyword arguments:
        setting -- the parameter settings
        """
        PY_SCIP_CALL(scip.SCIPsetPresolving(self._scip, setting, True))

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
        PY_SCIP_CALL(scip.SCIPwriteOrigProblem(self._scip, fn, ext, False))
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
            ub = scip.SCIPinfinity(self._scip)
        cdef scip.SCIP_VAR* scip_var
        if vtype in ['C', 'CONTINUOUS']:
            self._createVarBasic(&scip_var, name, lb, ub, obj, scip.SCIP_VARTYPE_CONTINUOUS)
        elif vtype in ['B', 'BINARY']:
            lb = 0.0
            ub = 1.0
            self._createVarBasic(&scip_var, name, lb, ub, obj, scip.SCIP_VARTYPE_BINARY)
        elif vtype in ['I', 'INTEGER']:
            self._createVarBasic(&scip_var, name, lb, ub, obj, scip.SCIP_VARTYPE_INTEGER)

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
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        self._releaseVar(_var)

    def getTransformedVar(self, var):
        """Retrieve the transformed variable.

        Keyword arguments:
        var -- the variable
        """
        cdef scip.SCIP_VAR* _var
        cdef scip.SCIP_VAR* _tvar
        cdef Var v
        cdef Var tv
        transvar = Variable() # TODO: set proper name?
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        tv = <Var>var.var
        _tvar = <scip.SCIP_VAR*>tv._var
        PY_SCIP_CALL(
            scip.SCIPtransformVar(self._scip, _var, &_tvar))
        return transvar

    def chgVarLb(self, var, lb=None):
        """Changes the lower bound of the specified variable.

        Keyword arguments:
        var -- the variable
        lb -- the lower bound (default None)
        """
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var

        if lb is None:
           lb = -scip.SCIPinfinity(self._scip)

        PY_SCIP_CALL(scip.SCIPchgVarLb(self._scip, _var, lb))

    def chgVarUb(self, var, ub=None):
        """Changes the upper bound of the specified variable.

        Keyword arguments:
        var -- the variable
        ub -- the upper bound (default None)
        """
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var

        if ub is None:
           ub = scip.SCIPinfinity(self._scip)

        PY_SCIP_CALL(scip.SCIPchgVarLb(self._scip, _var, ub))

    def getVars(self):
        """Retrieve all variables."""
        cdef scip.SCIP_VAR** _vars
        cdef scip.SCIP_VAR* _var
        cdef Var v
        cdef const char* _name
        cdef int _nvars
        vars = []

        _vars = SCIPgetVars(self._scip)
        _nvars = SCIPgetNVars(self._scip)

        for i in range(_nvars):
            _var = _vars[i];
            _name = SCIPvarGetName(_var)
            var = Variable(_name)
            v = var.var
            v._var = _var
            vars.append(var)

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
            lhs = -scip.SCIPinfinity(self._scip)
        if rhs is None:
            rhs = scip.SCIPinfinity(self._scip)
        cdef scip.SCIP_CONS* scip_cons
        self._createConsLinear(&scip_cons, name, 0, NULL, NULL, lhs, rhs,
                                initial, separate, enforce, check, propagate,
                                local, modifiable, dynamic, removable, stickingatnode)
        cdef scip.SCIP_Real coeff
        cdef Var v
        cdef scip.SCIP_VAR* _var
        for k in coeffs:
            coeff = <scip.SCIP_Real>coeffs[k]
            v = <Var>k.var
            _var = <scip.SCIP_VAR*>v._var
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
        kwargs['lhs'] = -scip.SCIPinfinity(self._scip) if quadcons.lb is None else quadcons.lb
        kwargs['rhs'] =  scip.SCIPinfinity(self._scip) if quadcons.ub is None else quadcons.ub
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2
        assert terms[()] == 0.0

        name = str_conversion("quadcons") # TODO

        cdef scip.SCIP_CONS* scip_cons
        PY_SCIP_CALL(scip.SCIPcreateConsQuadratic(
            self._scip, &scip_cons, name,
            0, NULL, NULL,        # linear
            0, NULL, NULL, NULL,  # quadratc
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable']))

        cdef Var var1
        cdef Var var2
        cdef scip.SCIP_VAR* _var1
        cdef scip.SCIP_VAR* _var2
        for v, c in terms.items():
            if len(v) == 0: # constant
                assert c == 0.0
            elif len(v) == 1: # linear
                var1 = <Var>v[0].var
                _var1 = <scip.SCIP_VAR*>var1._var
                PY_SCIP_CALL(SCIPaddLinearVarQuadratic(self._scip, scip_cons, _var1, c))
            else: # quadratic
                assert len(v) == 2, 'term: %s' % v
                var1 = <Var>v[0].var
                _var1 = <scip.SCIP_VAR*>var1._var
                var2 = <Var>v[1].var
                _var2 = <scip.SCIP_VAR*>var2._var
                PY_SCIP_CALL(SCIPaddBilinTermQuadratic(self._scip, scip_cons, _var1, _var2, c))

        self._addCons(scip_cons)
        cons = Cons()
        cons._cons = scip_cons
        return cons

    def addConsCoeff(self, Cons cons, var, coeff):
        """Add coefficient to the linear constraint (if non-zero).

        Keyword arguments:
        cons -- the constraint
        coeff -- the coefficient
        """
        cdef scip.SCIP_CONS* _cons
        cdef scip.SCIP_VAR* _var
        cdef Var v
        _cons = <scip.SCIP_CONS*>cons._cons
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        PY_SCIP_CALL(scip.SCIPaddCoefLinear(self._scip, _cons, _var, coeff))

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
        cdef scip.SCIP_CONS* scip_cons
        cdef Var v
        cdef scip.SCIP_VAR* _var
        cdef int _nvars

        self._createConsSOS1(&scip_cons, name, 0, NULL, NULL,
                                initial, separate, enforce, check, propagate,
                                local, dynamic, removable, stickingatnode)

        if weights is None:
            for k in vars:
                v = <Var>k.var
                _var = <scip.SCIP_VAR*>v._var
                self._appendVarSOS1(scip_cons, _var)
        else:
            nvars = len(vars)
            for k in range(nvars):
                v = <Var>vars[k].var
                _var = <scip.SCIP_VAR*>v._var
                weight = weights[k]
                self._addVarSOS1(scip_cons, _var, weight)

        cdef Cons c
        self._addCons(scip_cons)
        cons = Constraint(name)
        c = cons.cons
        c._cons = scip_cons

        return cons

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
        cdef scip.SCIP_CONS* scip_cons
        cdef Var v
        cdef scip.SCIP_VAR* _var
        cdef int _nvars

        self._createConsSOS2(&scip_cons, name, 0, NULL, NULL,
                                initial, separate, enforce, check, propagate,
                                local, dynamic, removable, stickingatnode)

        if weights is None:
            for k in vars:
                v = <Var>k.var
                _var = <scip.SCIP_VAR*>v._var
                self._appendVarSOS2(scip_cons, _var)
        else:
            nvars = len(vars)
            for k in range(nvars):
                v = <Var>vars[k].var
                _var = <scip.SCIP_VAR*>v._var
                weight = weights[k]
                self._addVarSOS2(scip_cons, _var, weight)

        cdef Cons c
        self._addCons(scip_cons)
        cons = Constraint(name)
        c = cons.cons
        c._cons = scip_cons

        return cons

    def addVarSOS1(self, Cons cons, var, weight):
        """Add variable to SOS1 constraint.

        Keyword arguments:
        cons -- the SOS1 constraint
        vars -- the variable
        weight -- the weight
        """
        cdef scip.SCIP_CONS* _cons
        cdef scip.SCIP_VAR* _var
        cdef Var v
        _cons = <scip.SCIP_CONS*>cons._cons
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        PY_SCIP_CALL(scip.SCIPaddVarSOS1(self._scip, _cons, _var, weight))

    def appendVarSOS1(self, Cons cons, var):
        """Append variable to SOS1 constraint.

        Keyword arguments:
        cons -- the SOS1 constraint
        vars -- the variable
        """
        cdef scip.SCIP_CONS* _cons
        cdef scip.SCIP_VAR* _var
        cdef Var v
        _cons = <scip.SCIP_CONS*>cons._cons
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        PY_SCIP_CALL(scip.SCIPappendVarSOS1(self._scip, _cons, _var))

    def addVarSOS2(self, Cons cons, var, weight):
        """Add variable to SOS2 constraint.

        Keyword arguments:
        cons -- the SOS2 constraint
        vars -- the variable
        weight -- the weight
        """
        cdef scip.SCIP_CONS* _cons
        cdef scip.SCIP_VAR* _var
        cdef Var v
        _cons = <scip.SCIP_CONS*>cons._cons
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        PY_SCIP_CALL(scip.SCIPaddVarSOS2(self._scip, _cons, _var, weight))

    def appendVarSOS2(self, Cons cons, var):
        """Append variable to SOS2 constraint.

        Keyword arguments:
        cons -- the SOS2 constraint
        vars -- the variable
        """
        cdef scip.SCIP_CONS* _cons
        cdef scip.SCIP_VAR* _var
        cdef Var v
        _cons = <scip.SCIP_CONS*>cons._cons
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        PY_SCIP_CALL(scip.SCIPappendVarSOS2(self._scip, _cons, _var))

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

        PY_SCIP_CALL(scip.SCIPtransformCons(self._scip, c._cons, &ctrans._cons))
        return transcons

    def getConss(self):
        """Retrieve all constraints."""
        cdef scip.SCIP_CONS** _conss
        cdef scip.SCIP_CONS* _cons
        cdef Cons c
        cdef const char* _name
        cdef int _nconss
        conss = []

        _conss = SCIPgetConss(self._scip)
        _nconss = SCIPgetNConss(self._scip)

        for i in range(_nconss):
            _cons = _conss[i];
            _name = SCIPconsGetName(_cons)
            cons = Constraint(_name)
            c = cons.cons
            c._cons = _cons
            conss.append(cons)

        return conss

    def getDualsolLinear(self, cons):
        """Retrieve the dual solution to a linear constraint.

        Keyword arguments:
        cons -- the linear constraint
        """
        cdef Cons c
        c = cons.cons
        return scip.SCIPgetDualsolLinear(self._scip, c._cons)

    def getDualfarkasLinear(self, cons):
        """Retrieve the dual farkas value to a linear constraint.

        Keyword arguments:
        cons -- the linear constraint
        """
        cdef Cons c
        c = cons.cons
        return scip.SCIPgetDualfarkasLinear(self._scip, c._cons)

    def optimize(self):
        """Optimize the problem."""
        PY_SCIP_CALL(scip.SCIPsolve(self._scip))
        self._bestSol = scip.SCIPgetBestSol(self._scip)

    def infinity(self):
        """Retrieve 'infinity' value."""
        inf = scip.SCIPinfinity(self._scip)
        return inf

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
        PY_SCIP_CALL(scip.SCIPincludePricer(self._scip, n, d,
                                            priority, delay,
                                            PyPricerCopy, PyPricerFree, PyPricerInit, PyPricerExit, PyPricerInitsol, PyPricerExitsol, PyPricerRedcost, PyPricerFarkas,
                                            <SCIP_PRICERDATA*>pricer))
        cdef SCIP_PRICER* scip_pricer
        scip_pricer = scip.SCIPfindPricer(self._scip, n)
        PY_SCIP_CALL(scip.SCIPactivatePricer(self._scip, scip_pricer))
        pricer.model = self

    def includeReader(self, Reader reader, name, desc, extension):
        """"Include a reader.

        Keyword arguments:
        reader -- the reader
        name -- the name
        desc -- the description
        extension -- file extension that reader processes
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        ext = str_conversion(extension)
        PY_SCIP_CALL(scip.SCIPincludeReader(self._scip, n, d, ext, PyReaderCopy, PyReaderFree, PyReaderRead, PyReaderWrite, <SCIP_READERDATA*>reader))
        reader.model = self

    def includeConshdlr(self, Conshdlr conshdlr, name, desc, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                        maxprerounds, delaysepa, delayprop, needscons, proptiming, presoltiming):
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
        PY_SCIP_CALL(scip.SCIPincludeConshdlr(self._scip, n, d, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                                              maxprerounds, delaysepa, delayprop, needscons, proptiming, presoltiming,
                                              PyConshdlrCopy, PyConsFree, PyConsInit, PyConsExit, PyConsInitpre, PyConsExitpre,
                                              PyConsInitsol, PyConsExitsol, PyConsDelete, PyConsTrans, PyConsInitlp, PyConsSepalp, PyConsSepasol,
                                              PyConsEnfolp, PyConsEnfops, PyConsCheck, PyConsProp, PyConsPresol, PyConsResprop, PyConsLock,
                                              PyConsActive, PyConsDeactive, PyConsEnable, PyConsDisable, PyConsDelvars, PyConsPrint, PyConsCopy,
                                              PyConsParse, PyConsGetvars, PyConsGetnvars, PyConsGetdivebdchgs,
                                              <SCIP_CONSHDLRDATA*>conshdlr))
        conshdlr.model = self

    def includePresol(self, Presol presol, name, desc, priority, maxrounds, timing):
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
        PY_SCIP_CALL(scip.SCIPincludePresol(self._scip, n, d, priority, maxrounds, timing,
           PyPresolCopy, PyPresolFree, PyPresolInit, PyPresolExit, PyPresolInitpre, PyPresolExitpre, PyPresolExec, <SCIP_PRESOLDATA*>presol))
        presol.model = self

    def includeSepa(self, Sepa sepa, name, desc, priority, freq, maxbounddist, usessubscip, delay):
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
        PY_SCIP_CALL(scip.SCIPincludeSepa(self._scip, n, d, priority, freq, maxbounddist, usessubscip, delay,
PySepaCopy, PySepaFree, PySepaInit, PySepaExit, PySepaInitsol, PySepaExitsol, PySepaExeclp, PySepaExecsol, <SCIP_SEPADATA*>sepa))
        sepa.model = self

    def includeProp(self, Prop prop, name, desc, presolpriority, presolmaxrounds,
                    proptiming, presoltiming, priority=1, freq=1, delay=True):
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
        PY_SCIP_CALL(scip.SCIPincludeProp(self._scip, n, d,
                                          priority, freq, delay,
                                          proptiming, presolpriority, presolmaxrounds,
                                          presoltiming, PyPropCopy, PyPropFree, PyPropInit, PyPropExit,
                                          PyPropInitpre, PyPropExitpre, PyPropInitsol, PyPropExitsol,
                                          PyPropPresol, PyPropExec, PyPropResProp,
                                          <SCIP_PROPDATA*> prop))
        prop.model = self

    def includeHeur(self, Heur heur, name, desc, dispchar, priority, freqofs,
                    maxdepth, timingmask, usessubscip, freq=1):
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
        dis = str_conversion(dispchar)
        PY_SCIP_CALL(scip.SCIPincludeHeur(self._scip, nam, des, dis,
                                          priority, freq, freqofs,
                                          maxdepth, timingmask, usessubscip,
                                          PyHeurCopy, PyHeurFree, PyHeurInit, PyHeurExit,
                                          PyHeurInitsol, PyHeurExitsol, PyHeurExec,
                                          <SCIP_HEURDATA*> heur))
        heur.model = self


    def includeNodesel(self, Nodesel nodesel, name, desc, stdpriority, memsavepriority):
        """Include a node selector.

        Keyword arguments:
        nodesel -- the node selector
        name -- the name of the node selector
        desc -- the description
        stdpriority -- priority of the node selector in standard mode
        memsavepriority -- priority of the node selector in memory saving mode
        """

        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(scip.SCIPincludeNodesel(self._scip, nam, des,
                                             stdpriority, memsavepriority,
                                             PyNodeselCopy, PyNodeselFree, PyNodeselInit, PyNodeselExit,
                                             PyNodeselInitsol, PyNodeselExitsol, PyNodeselSelect, PyNodeselComp,
                                             <SCIP_NODESELDATA*> nodesel))
        nodesel.model = self

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
        PY_SCIP_CALL(scip.SCIPincludeBranchrule(self._scip, nam, des,
                                          maxdepth, maxdepth, maxbounddist,
                                          PyBranchruleCopy, PyBranchruleFree, PyBranchruleInit, PyBranchruleExit,
                                          PyBranchruleInitsol, PyBranchruleExitsol, PyBranchruleExeclp, PyBranchruleExecext,
                                          PyBranchruleExecps, <SCIP_BRANCHRULEDATA*> branchrule))
        branchrule.model = self

    # Solution functions

    def getSols(self):
        """Retrieve list of all feasible primal solutions stored in the solution storage."""
        cdef scip.SCIP_SOL** _sols
        cdef scip.SCIP_SOL* _sol
        _sols = scip.SCIPgetSols(self._scip)
        nsols = scip.SCIPgetNSols(self._scip)
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
        solution._solution = scip.SCIPgetBestSol(self._scip)
        return solution

    def getSolObjVal(self, Solution solution, original=True):
        """Retrieve the objective value of the solution.

        Keyword arguments:
        solution -- the solution
        original -- retrieve the solution of the original problem? (default True)
        """
        cdef scip.SCIP_SOL* _solution
        _solution = <scip.SCIP_SOL*>solution._solution
        if original:
            objval = scip.SCIPgetSolOrigObj(self._scip, _solution)
        else:
            objval = scip.SCIPgetSolTransObj(self._scip, _solution)
        return objval

    def getObjVal(self, original=True):
        """Retrieve the objective value of value of best solution"""
        if original:
            objval = scip.SCIPgetSolOrigObj(self._scip, self._bestSol)
        else:
            objval = scip.SCIPgetSolTransObj(self._scip, self._bestSol)
        return objval

    # Get best dual bound
    def getDualbound(self):
        """Retrieve the best dual bound."""
        return scip.SCIPgetDualbound(self._scip)

    def getVal(self, var, Solution solution=None):
        """Retrieve the value of the variable in the specified solution. If no solution is specified,
        the best known solution is used.

        Keyword arguments:
        var -- the variable
        solution -- the solution (default None)
        """
        cdef scip.SCIP_SOL* _sol
        if solution is None:
            _sol = self._bestSol
        else:
            _sol = <scip.SCIP_SOL*>solution._solution
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        return scip.SCIPgetSolVal(self._scip, _sol, _var)

    def writeName(self, var):
        """Write the name of the variable to the std out."""
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        self._writeVarName(_var)

    def getStatus(self):
        """Retrieve solution status."""
        cdef scip.SCIP_STATUS stat = scip.SCIPgetStatus(self._scip)
        if stat == scip.SCIP_STATUS_OPTIMAL:
            return "optimal"
        elif stat == scip.SCIP_STATUS_TIMELIMIT:
            return "timelimit"
        elif stat == scip.SCIP_STATUS_INFEASIBLE:
            return "infeasible"
        elif stat == scip.SCIP_STATUS_UNBOUNDED:
            return "unbounded"
        else:
            return "unknown"

    def getObjectiveSense(self):
        """Retrieve objective sense."""
        cdef scip.SCIP_OBJSENSE sense = scip.SCIPgetObjsense(self._scip)
        if sense == scip.SCIP_OBJSENSE_MAXIMIZE:
            return "maximize"
        elif sense == scip.SCIP_OBJSENSE_MINIMIZE:
            return "minimize"
        else:
            return "unknown"

    # Statistic Methods

    def printStatistics(self):
        """Print statistics."""
        PY_SCIP_CALL(scip.SCIPprintStatistics(self._scip, NULL))

    # Verbosity Methods

    def hideOutput(self, quiet = True):
        """Hide the output.

        Keyword arguments:
        quiet -- hide output? (default True)
        """
        scip.SCIPsetMessagehdlrQuiet(self._scip, quiet)

    # Parameter Methods

    def setBoolParam(self, name, value):
        """Set a boolean-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetBoolParam(self._scip, n, value))

    def setIntParam(self, name, value):
        """Set an int-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetIntParam(self._scip, n, value))

    def setLongintParam(self, name, value):
        """Set a long-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetLongintParam(self._scip, n, value))

    def setRealParam(self, name, value):
        """Set a real-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetRealParam(self._scip, n, value))

    def setCharParam(self, name, value):
        """Set a char-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetCharParam(self._scip, n, value))

    def setStringParam(self, name, value):
        """Set a string-valued parameter.

        Keyword arguments:
        name -- the name of the parameter
        value -- the value of the parameter
        """
        n = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetStringParam(self._scip, n, value))

    def readParams(self, file):
        """Read an external parameter file.

        Keyword arguments:
        file -- the file to be read
        """
        absfile = bytes(abspath(file), 'utf-8')
        PY_SCIP_CALL(scip.SCIPreadParams(self._scip, absfile))

    def readProblem(self, file, extension = None):
        """Read a problem instance from an external file.

        Keyword arguments:
        file -- the file to be read
        extension -- specifies extensions (default None)
        """
        absfile = bytes(abspath(file), 'utf-8')
        if extension is None:
            PY_SCIP_CALL(scip.SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = bytes(extension, 'utf-8')
            PY_SCIP_CALL(scip.SCIPreadProb(self._scip, absfile, extension))
