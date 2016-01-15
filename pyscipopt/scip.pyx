# Copyright (C) 2012-2013 Robert Schwarz
#   see file 'LICENSE' for details.

include "pricer.pyx"

from os.path import abspath
import sys

cimport pyscipopt.scip as scip
from pyscipopt.linexpr import LinExpr, LinCons

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
    '''Base class holding a pointer to corresponding SCIP_VAR'''
    cdef scip.SCIP_VAR* _var


class Variable(LinExpr):
    '''Is a linear expression and has SCIP_VAR*'''

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


cdef class Cons:
    cdef scip.SCIP_CONS* _cons

class Constraint:

    def __init__(self, name=None):
        self.cons = Cons()
        self.name = name



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
        name1 = str_conversion(problemName)
        return scip.SCIPcreateProbBasic(self._scip, name1)

    @scipErrorHandler
    def free(self):
        return scip.SCIPfree(&self._scip)

    @scipErrorHandler
    def freeProb(self):
        return scip.SCIPfreeProb(self._scip)

    @scipErrorHandler
    def freeTransform(self):
        return scip.SCIPfreeTransform(self._scip)

    #@scipErrorHandler       We'll be able to use decorators when we
    #                        interface the relevant classes (SCIP_VAR, ...)
    cdef _createVarBasic(self, scip.SCIP_VAR** scip_var, name,
                        lb, ub, obj, scip.SCIP_VARTYPE varType):
        name1 = str_conversion(name)
        PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, scip_var,
                           name1, lb, ub, obj, varType))

    cdef _addVar(self, scip.SCIP_VAR* scip_var):
        PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

    cdef _addPricedVar(self, scip.SCIP_VAR* scip_var):
        PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, 1.0))

    cdef _createConsLinear(self, scip.SCIP_CONS** cons, name, nvars,
                                SCIP_VAR** vars, SCIP_Real* vals, lhs, rhs,
                                initial=True, separate=True, enforce=True, check=True,
                                propagate=True, local=False, modifiable=False, dynamic=False,
                                removable=False, stickingatnode=False):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPcreateConsLinear(self._scip, cons,
                                                    name1, nvars, vars, vals,
                                                    lhs, rhs, initial, separate, enforce,
                                                    check, propagate, local, modifiable,
                                                    dynamic, removable, stickingatnode) )

    cdef _addCoefLinear(self, scip.SCIP_CONS* cons, SCIP_VAR* var, val):
        PY_SCIP_CALL(scip.SCIPaddCoefLinear(self._scip, cons, var, val))

    cdef _addCons(self, scip.SCIP_CONS* cons):
        PY_SCIP_CALL(scip.SCIPaddCons(self._scip, cons))

    cdef _writeVarName(self, scip.SCIP_VAR* var):
        PY_SCIP_CALL(scip.SCIPwriteVarName(self._scip, NULL, var, False))

    cdef _releaseVar(self, scip.SCIP_VAR* var):
        PY_SCIP_CALL(scip.SCIPreleaseVar(self._scip, &var))


    # Setting the objective sense
    def setMinimize(self):
        PY_SCIP_CALL(scip.SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        PY_SCIP_CALL(scip.SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    # set objective function, either as variable dictionary or as linear expression
    def setObjective(self, coeffs, sense = 'minimize'):
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
        PY_SCIP_CALL(scip.SCIPsetPresolving(self._scip, setting, True))

    # Write original problem to file
    def writeProblem(self, filename='origprob.cip'):
        if filename.find('.') < 0:
            filename = filename + '.cip'
            ext = str_conversion('cip')
        else:
            ext = str_conversion(filename.split('.')[1])
        fn = str_conversion(filename)
        PY_SCIP_CALL(scip.SCIPwriteOrigProblem(self._scip, fn, ext, False))
        print('wrote original problem to file ' + filename)

    # Variable Functions
    # Create a new variable
    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, pricedVar = False):
        if ub is None:
            ub = scip.SCIPinfinity(self._scip)
        cdef scip.SCIP_VAR* scip_var
        cdef Var v
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
        var = Variable(name)
        v = var.var
        v._var = scip_var

        self._releaseVar(scip_var)
        return var

    # Release the variable
    def releaseVar(self, var):
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        self._releaseVar(_var)

    # Retrieving the pointer for the transformed variable
    def getTransformedVar(self, var):
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

    # Get variables
    def getVars(self):
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
    # Adding a linear constraint. By default the lhs is set to 0.0.
    # If the lhs is to be unbounded, then you set lhs to None.
    # By default the rhs is unbounded.
    def addCons(self, coeffs, lhs=0.0, rhs=None, name="cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, modifiable=False, dynamic=False,
                removable=False, stickingatnode=False):
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
        cdef Cons c
        self._addCons(scip_cons)
        cons = Constraint(name)
        c = cons.cons
        c._cons = scip_cons
     #   cons = Cons(name)
     #   cons._cons = scip_cons

        return cons

    def _addLinCons(self, lincons, **kwargs):
        '''add object of class LinCons'''
        assert isinstance(lincons, LinCons)
        kwargs['lhs'], kwargs['rhs'] = lincons.lb, lincons.ub
        terms = lincons.expr.terms
        assert lincons.expr.degree() <= 1
        assert terms[()] == 0.0
        coeffs = {t[0]:c for t, c in terms.items() if c != 0.0}

        return self.addCons(coeffs, **kwargs)

    def _addQuadCons(self, quadcons, **kwargs):
        '''add object of class LinCons'''
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
        cdef scip.SCIP_CONS* _cons
        cdef scip.SCIP_VAR* _var
        cdef Var v
        _cons = <scip.SCIP_CONS*>cons._cons
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        PY_SCIP_CALL(scip.SCIPaddCoefLinear(self._scip, _cons, _var, coeff))


    # Retrieving the pointer for the transformed constraint
    def getTransformedCons(self, Cons cons):
        transcons = Cons()
        PY_SCIP_CALL(scip.SCIPtransformCons(self._scip, cons._cons, &transcons._cons))
        return transcons

    # Get constraints
    def getConss(self):
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

    # Retrieving the dual solution for a linear constraint
    def getDualsolLinear(self, cons):
        cdef Cons c
        c = cons.cons
        return scip.SCIPgetDualsolLinear(self._scip, c._cons)

    # Retrieving the dual farkas value for a linear constraint
    def getDualfarkasLinear(self, Cons cons):
        return scip.SCIPgetDualfarkasLinear(self._scip, cons._cons)


    # Problem solving functions
    # todo: define optimize() as a copy of solve() for Gurobi compatibility
    def optimize(self):
        PY_SCIP_CALL(scip.SCIPsolve(self._scip))
        self.freeTransform()
        self._bestSol = scip.SCIPgetBestSol(self._scip)

    # Numerical methods
    def infinity(self):
         inf = scip.SCIPinfinity(self._scip)
         return inf

    # Pricer functions
    def includePricer(self, Pricer pricer, name, desc):
        PY_SCIP_CALL(scip.SCIPincludePricerBasic(self._scip, &(pricer._pricer), name, desc, 1, True, scipPricerRedcost, scipPricerFarkas, pricer._pricerdata))
        PY_SCIP_CALL(scip.SCIPactivatePricer(self._scip, pricer._pricer))
        PY_SCIP_CALL(scip.SCIPsetPricerInit(self._scip, pricer._pricer, scipPricerInit))



    # Solution functions
    # Retrieve the current best solution
    def getBestSol(self):
        solution = Solution()
        solution._solution = scip.SCIPgetBestSol(self._scip)
        return solution

    # Get problem objective value
    def getSolObjVal(self, Solution solution, original=True):
        cdef scip.SCIP_SOL* _solution
        _solution = <scip.SCIP_SOL*>solution._solution
        if original:
            objval = scip.SCIPgetSolOrigObj(self._scip, _solution)
        else:
            objval = scip.SCIPgetSolTransObj(self._scip, _solution)
        return objval

    # Get objective value of best solution
    def getObjVal(self, original=True):
        if original:
            objval = scip.SCIPgetSolOrigObj(self._scip, self._bestSol)
        else:
            objval = scip.SCIPgetSolTransObj(self._scip, self._bestSol)
        return objval

    # Get best dual bound
    def getDualbound(self):
        return scip.SCIPgetDualbound(self._scip)

    # Retrieve the value of the variable in the final solution
    def getVal(self, var, Solution solution=None):
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

    # Write the names of the variable to the std out.
    def writeName(self, var):
        cdef scip.SCIP_VAR* _var
        cdef Var v
        v = <Var>var.var
        _var = <scip.SCIP_VAR*>v._var
        self._writeVarName(_var)

    # return solution status
    def getStatus(self):
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

    # return objective sense
    def getObjectiveSense(self):
        cdef scip.SCIP_OBJSENSE sense = scip.SCIPgetObjsense(self._scip)
        if sense == scip.SCIP_OBJSENSE_MAXIMIZE:
            return "maximize"
        elif sense == scip.SCIP_OBJSENSE_MINIMIZE:
            return "minimize"
        else:
            return "unknown"

    # Statistic Methods
    def printStatistics(self):
        PY_SCIP_CALL(scip.SCIPprintStatistics(self._scip, NULL))

    # Verbosity Methods
    def hideOutput(self, quiet = True):
        scip.SCIPsetMessagehdlrQuiet(self._scip, quiet)

    # Parameter Methods
    def setBoolParam(self, name, value):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetBoolParam(self._scip, name1, value))

    def setIntParam(self, name, value):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetIntParam(self._scip, name1, value))

    def setLongintParam(self, name, value):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetLongintParam(self._scip, name1, value))

    def setRealParam(self, name, value):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetRealParam(self._scip, name1, value))

    def setCharParam(self, name, value):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetCharParam(self._scip, name1, value))

    def setStringParam(self, name, value):
        name1 = str_conversion(name)
        PY_SCIP_CALL(scip.SCIPsetStringParam(self._scip, name1, value))

    def readParams(self, file):
        absfile = bytes(abspath(file), 'utf-8')
        PY_SCIP_CALL(scip.SCIPreadParams(self._scip, absfile))

    def readProblem(self, file, extension = None):
        absfile = bytes(abspath(file), 'utf-8')
        if extension is None:
            PY_SCIP_CALL(scip.SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = bytes(extension, 'utf-8')
            PY_SCIP_CALL(scip.SCIPreadProb(self._scip, absfile, extension))
