##@file scip.pxi
#@brief holding functions in python that reference the SCIP public functions included in scip.pxd
from os.path import abspath
from os.path import splitext
import os
import sys
import warnings
import locale

cimport cython
from cpython cimport Py_INCREF, Py_DECREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
from libc.stdio cimport stdout, stderr, fdopen, fputs, fflush, fclose
from posix.stdio cimport fileno

from collections.abc import Iterable
from itertools import repeat
from dataclasses import dataclass
from typing import Union

import numpy as np

include "expr.pxi"
include "lp.pxi"
include "benders.pxi"
include "benderscut.pxi"
include "branchrule.pxi"
include "conshdlr.pxi"
include "cutsel.pxi"
include "event.pxi"
include "heuristic.pxi"
include "iisfinder.pxi"
include "presol.pxi"
include "pricer.pxi"
include "propagator.pxi"
include "sepa.pxi"
include "reader.pxi"
include "relax.pxi"
include "nodesel.pxi"
include "matrix.pxi"

# recommended SCIP version; major version is required
MAJOR = 10
MINOR = 0
PATCH = 0

# for external user functions use def; for functions used only inside the interface (starting with _) use cdef
# todo: check whether this is currently done like this

if sys.version_info >= (3, 0):
    str_conversion = lambda x:bytes(x,'utf-8')
else:
    str_conversion = lambda x:x

_SCIP_BOUNDTYPE_TO_STRING = {SCIP_BOUNDTYPE_UPPER: '<=',
                             SCIP_BOUNDTYPE_LOWER: '>='}

cdef extern from "scip/config.h":
    """
    #ifdef WITH_DEBUG_SOLUTION
    #define PYSCIPOPT_WITH_DEBUG_SOLUTION 1
    #else
    #define PYSCIPOPT_WITH_DEBUG_SOLUTION 0
    #endif
    #ifdef SCIP_WITH_EXACTSOLVE
    #define PYSCIPOPT_WITH_EXACTSOLVE 1
    #else
    #define PYSCIPOPT_WITH_EXACTSOLVE 0
    #endif
    """
    bint PYSCIPOPT_WITH_DEBUG_SOLUTION
    bint PYSCIPOPT_WITH_EXACTSOLVE

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

cdef class PY_SCIP_LPPARAM:
    FROMSCRATCH    = SCIP_LPPAR_FROMSCRATCH
    FASTMIP        = SCIP_LPPAR_FASTMIP
    SCALING        = SCIP_LPPAR_SCALING
    PRESOLVING     = SCIP_LPPAR_PRESOLVING
    PRICING        = SCIP_LPPAR_PRICING
    LPINFO         = SCIP_LPPAR_LPINFO
    FEASTOL        = SCIP_LPPAR_FEASTOL
    DUALFEASTOL    = SCIP_LPPAR_DUALFEASTOL
    BARRIERCONVTOL = SCIP_LPPAR_BARRIERCONVTOL
    OBJLIM         = SCIP_LPPAR_OBJLIM
    LPITLIM        = SCIP_LPPAR_LPITLIM
    LPTILIM        = SCIP_LPPAR_LPTILIM
    MARKOWITZ      = SCIP_LPPAR_MARKOWITZ
    ROWREPSWITCH   = SCIP_LPPAR_ROWREPSWITCH
    THREADS        = SCIP_LPPAR_THREADS
    CONDITIONLIMIT = SCIP_LPPAR_CONDITIONLIMIT
    TIMING         = SCIP_LPPAR_TIMING
    RANDOMSEED     = SCIP_LPPAR_RANDOMSEED
    POLISHING      = SCIP_LPPAR_POLISHING
    REFACTOR       = SCIP_LPPAR_REFACTOR

cdef class PY_SCIP_BASESTAT:
    LOWER = SCIP_BASESTAT_LOWER
    BASIC = SCIP_BASESTAT_BASIC
    UPPER = SCIP_BASESTAT_UPPER
    ZERO  = SCIP_BASESTAT_ZERO

cdef class PY_SCIP_PARAMEMPHASIS:
    DEFAULT      = SCIP_PARAMEMPHASIS_DEFAULT
    CPSOLVER     = SCIP_PARAMEMPHASIS_CPSOLVER
    EASYCIP      = SCIP_PARAMEMPHASIS_EASYCIP
    FEASIBILITY  = SCIP_PARAMEMPHASIS_FEASIBILITY
    HARDLP       = SCIP_PARAMEMPHASIS_HARDLP
    OPTIMALITY   = SCIP_PARAMEMPHASIS_OPTIMALITY
    COUNTER      = SCIP_PARAMEMPHASIS_COUNTER
    PHASEFEAS    = SCIP_PARAMEMPHASIS_PHASEFEAS
    PHASEIMPROVE = SCIP_PARAMEMPHASIS_PHASEIMPROVE
    PHASEPROOF   = SCIP_PARAMEMPHASIS_PHASEPROOF
    NUMERICS     = SCIP_PARAMEMPHASIS_NUMERICS
    BENCHMARK    = SCIP_PARAMEMPHASIS_BENCHMARK

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
    PRIMALLIMIT    = SCIP_STATUS_PRIMALLIMIT
    DUALLIMIT      = SCIP_STATUS_DUALLIMIT
    OPTIMAL        = SCIP_STATUS_OPTIMAL
    INFEASIBLE     = SCIP_STATUS_INFEASIBLE
    UNBOUNDED      = SCIP_STATUS_UNBOUNDED
    INFORUNBD      = SCIP_STATUS_INFORUNBD

StageNames = {}
cdef class PY_SCIP_STAGE:
    INIT         = SCIP_STAGE_INIT
    PROBLEM      = SCIP_STAGE_PROBLEM
    TRANSFORMING = SCIP_STAGE_TRANSFORMING
    TRANSFORMED  = SCIP_STAGE_TRANSFORMED
    INITPRESOLVE = SCIP_STAGE_INITPRESOLVE
    PRESOLVING   = SCIP_STAGE_PRESOLVING
    EXITPRESOLVE = SCIP_STAGE_EXITPRESOLVE
    PRESOLVED    = SCIP_STAGE_PRESOLVED
    INITSOLVE    = SCIP_STAGE_INITSOLVE
    SOLVING      = SCIP_STAGE_SOLVING
    SOLVED       = SCIP_STAGE_SOLVED
    EXITSOLVE    = SCIP_STAGE_EXITSOLVE
    FREETRANS    = SCIP_STAGE_FREETRANS
    FREE         = SCIP_STAGE_FREE

cdef class PY_SCIP_NODETYPE:
    FOCUSNODE   = SCIP_NODETYPE_FOCUSNODE
    PROBINGNODE = SCIP_NODETYPE_PROBINGNODE
    SIBLING     = SCIP_NODETYPE_SIBLING
    CHILD       = SCIP_NODETYPE_CHILD
    LEAF        = SCIP_NODETYPE_LEAF
    DEADEND     = SCIP_NODETYPE_DEADEND
    JUNCTION    = SCIP_NODETYPE_JUNCTION
    PSEUDOFORK  = SCIP_NODETYPE_PSEUDOFORK
    FORK        = SCIP_NODETYPE_FORK
    SUBROOT     = SCIP_NODETYPE_SUBROOT
    REFOCUSNODE = SCIP_NODETYPE_REFOCUSNODE

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

EventNames = {}
cdef class PY_SCIP_EVENTTYPE:
    DISABLED          = SCIP_EVENTTYPE_DISABLED
    VARADDED          = SCIP_EVENTTYPE_VARADDED
    VARDELETED        = SCIP_EVENTTYPE_VARDELETED
    VARFIXED          = SCIP_EVENTTYPE_VARFIXED
    VARUNLOCKED       = SCIP_EVENTTYPE_VARUNLOCKED
    OBJCHANGED        = SCIP_EVENTTYPE_OBJCHANGED
    GLBCHANGED        = SCIP_EVENTTYPE_GLBCHANGED
    GUBCHANGED        = SCIP_EVENTTYPE_GUBCHANGED
    LBTIGHTENED       = SCIP_EVENTTYPE_LBTIGHTENED
    LBRELAXED         = SCIP_EVENTTYPE_LBRELAXED
    UBTIGHTENED       = SCIP_EVENTTYPE_UBTIGHTENED
    UBRELAXED         = SCIP_EVENTTYPE_UBRELAXED
    GHOLEADDED        = SCIP_EVENTTYPE_GHOLEADDED
    GHOLEREMOVED      = SCIP_EVENTTYPE_GHOLEREMOVED
    LHOLEADDED        = SCIP_EVENTTYPE_LHOLEADDED
    LHOLEREMOVED      = SCIP_EVENTTYPE_LHOLEREMOVED
    IMPLADDED         = SCIP_EVENTTYPE_IMPLADDED
    PRESOLVEROUND     = SCIP_EVENTTYPE_PRESOLVEROUND
    NODEFOCUSED       = SCIP_EVENTTYPE_NODEFOCUSED
    NODEFEASIBLE      = SCIP_EVENTTYPE_NODEFEASIBLE
    NODEINFEASIBLE    = SCIP_EVENTTYPE_NODEINFEASIBLE
    NODEBRANCHED      = SCIP_EVENTTYPE_NODEBRANCHED
    NODEDELETE        = SCIP_EVENTTYPE_NODEDELETE
    DUALBOUNDIMPROVED = SCIP_EVENTTYPE_DUALBOUNDIMPROVED
    FIRSTLPSOLVED     = SCIP_EVENTTYPE_FIRSTLPSOLVED
    LPSOLVED          = SCIP_EVENTTYPE_LPSOLVED
    LPEVENT           = SCIP_EVENTTYPE_LPEVENT
    POORSOLFOUND      = SCIP_EVENTTYPE_POORSOLFOUND
    BESTSOLFOUND      = SCIP_EVENTTYPE_BESTSOLFOUND
    ROWADDEDSEPA      = SCIP_EVENTTYPE_ROWADDEDSEPA
    ROWDELETEDSEPA    = SCIP_EVENTTYPE_ROWDELETEDSEPA
    ROWADDEDLP        = SCIP_EVENTTYPE_ROWADDEDLP
    ROWDELETEDLP      = SCIP_EVENTTYPE_ROWDELETEDLP
    ROWCOEFCHANGED    = SCIP_EVENTTYPE_ROWCOEFCHANGED
    ROWCONSTCHANGED   = SCIP_EVENTTYPE_ROWCONSTCHANGED
    ROWSIDECHANGED    = SCIP_EVENTTYPE_ROWSIDECHANGED
    SYNC              = SCIP_EVENTTYPE_SYNC
    GBDCHANGED        = SCIP_EVENTTYPE_GBDCHANGED
    LBCHANGED         = SCIP_EVENTTYPE_LBCHANGED
    UBCHANGED         = SCIP_EVENTTYPE_UBCHANGED
    BOUNDTIGHTENED    = SCIP_EVENTTYPE_BOUNDTIGHTENED
    BOUNDRELAXED      = SCIP_EVENTTYPE_BOUNDRELAXED
    BOUNDCHANGED      = SCIP_EVENTTYPE_BOUNDCHANGED
    GHOLECHANGED      = SCIP_EVENTTYPE_GHOLECHANGED
    LHOLECHANGED      = SCIP_EVENTTYPE_LHOLECHANGED
    HOLECHANGED       = SCIP_EVENTTYPE_HOLECHANGED
    DOMCHANGED        = SCIP_EVENTTYPE_DOMCHANGED
    VARCHANGED        = SCIP_EVENTTYPE_VARCHANGED
    VAREVENT          = SCIP_EVENTTYPE_VAREVENT
    NODESOLVED        = SCIP_EVENTTYPE_NODESOLVED
    NODEEVENT         = SCIP_EVENTTYPE_NODEEVENT
    SOLFOUND          = SCIP_EVENTTYPE_SOLFOUND
    SOLEVENT          = SCIP_EVENTTYPE_SOLEVENT
    GAPUPDATED        = SCIP_EVENTTYPE_GAPUPDATED
    ROWCHANGED        = SCIP_EVENTTYPE_ROWCHANGED
    ROWEVENT          = SCIP_EVENTTYPE_ROWEVENT

cdef class PY_SCIP_LOCKTYPE:
    MODEL    = SCIP_LOCKTYPE_MODEL
    CONFLICT = SCIP_LOCKTYPE_CONFLICT

cdef class PY_SCIP_IMPLINTTYPE:
    NONE   = SCIP_IMPLINTTYPE_NONE
    WEAK   = SCIP_IMPLINTTYPE_WEAK
    STRONG = SCIP_IMPLINTTYPE_STRONG

cdef class PY_SCIP_LPSOLSTAT:
    NOTSOLVED    = SCIP_LPSOLSTAT_NOTSOLVED
    OPTIMAL      = SCIP_LPSOLSTAT_OPTIMAL
    INFEASIBLE   = SCIP_LPSOLSTAT_INFEASIBLE
    UNBOUNDEDRAY = SCIP_LPSOLSTAT_UNBOUNDEDRAY
    OBJLIMIT     = SCIP_LPSOLSTAT_OBJLIMIT
    ITERLIMIT    = SCIP_LPSOLSTAT_ITERLIMIT
    TIMELIMIT    = SCIP_LPSOLSTAT_TIMELIMIT
    ERROR        = SCIP_LPSOLSTAT_ERROR

cdef class PY_SCIP_BRANCHDIR:
    DOWNWARDS = SCIP_BRANCHDIR_DOWNWARDS
    UPWARDS   = SCIP_BRANCHDIR_UPWARDS
    FIXED     = SCIP_BRANCHDIR_FIXED
    AUTO      = SCIP_BRANCHDIR_AUTO

cdef class PY_SCIP_BENDERSENFOTYPE:
    LP     = SCIP_BENDERSENFOTYPE_LP
    RELAX  = SCIP_BENDERSENFOTYPE_RELAX
    PSEUDO = SCIP_BENDERSENFOTYPE_PSEUDO
    CHECK  = SCIP_BENDERSENFOTYPE_CHECK

cdef class PY_SCIP_ROWORIGINTYPE:
    UNSPEC = SCIP_ROWORIGINTYPE_UNSPEC
    CONS   = SCIP_ROWORIGINTYPE_CONS
    SEPA   = SCIP_ROWORIGINTYPE_SEPA
    REOPT  = SCIP_ROWORIGINTYPE_REOPT

cdef class PY_SCIP_SOLORIGIN:
    ORIGINAL  = SCIP_SOLORIGIN_ORIGINAL
    ZERO      = SCIP_SOLORIGIN_ZERO
    LPSOL     = SCIP_SOLORIGIN_LPSOL
    NLPSOL    = SCIP_SOLORIGIN_NLPSOL
    RELAXSOL  = SCIP_SOLORIGIN_RELAXSOL
    PSEUDOSOL = SCIP_SOLORIGIN_PSEUDOSOL
    PARTIAL   = SCIP_SOLORIGIN_PARTIAL
    UNKNOWN   = SCIP_SOLORIGIN_UNKNOWN

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

cdef class Event:
    """Base class holding a pointer to corresponding SCIP_EVENT."""

    @staticmethod
    cdef create(SCIP_EVENT* scip_event):
        """
        Main method for creating an Event class. Is used instead of __init__.

        Parameters
        ----------
        scip_event : SCIP_EVENT*
            A pointer to the SCIP_EVENT

        Returns
        -------
        event : Event
            The Python representative of the SCIP_EVENT

        """
        if scip_event == NULL:
            raise Warning("cannot create Event with SCIP_EVENT* == NULL")
        event = Event()
        event.event = scip_event
        return event

    def getType(self):
        """
        Gets type of event.

        Returns
        -------
        PY_SCIP_EVENTTYPE

        """
        return SCIPeventGetType(self.event)

    def getName(self):
        """
        Gets name of event.

        Returns
        -------
        str

        """
        if not EventNames:
            self._getEventNames()
        return EventNames[self.getType()]

    def _getEventNames(self):
        """Gets event names."""
        for name in dir(PY_SCIP_EVENTTYPE):
            attr = getattr(PY_SCIP_EVENTTYPE, name)
            if isinstance(attr, int):
                EventNames[attr] = name

    def __repr__(self):
        return str(self.getType())

    def __str__(self):
        return self.getName()

    def getNewBound(self):
        """
        Gets new bound for a bound change event.

        Returns
        -------
        float

        """
        return SCIPeventGetNewbound(self.event)

    def getOldBound(self):
        """
        Gets old bound for a bound change event.

        Returns
        -------
        float

        """
        return SCIPeventGetOldbound(self.event)

    def getVar(self):
        """
        Gets variable for a variable event (var added, var deleted, var fixed,
        objective value or domain change, domain hole added or removed).

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* var = SCIPeventGetVar(self.event)
        return Variable.create(var)

    def getNode(self):
        """
        Gets node for a node or LP event.

        Returns
        -------
        Node

        """
        cdef SCIP_NODE* node = SCIPeventGetNode(self.event)
        return Node.create(node)

    def getRow(self):
        """
        Gets row for a row event.

        Returns
        -------
        Row

        """
        cdef SCIP_ROW* row = SCIPeventGetRow(self.event)
        return Row.create(row)

    def __hash__(self):
        return hash(<size_t>self.event)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.event == (<Event>other).event)

cdef class Solution:
    """Base class holding a pointer to corresponding SCIP_SOL."""

    # We are raising an error here to avoid creating a solution without an associated model. See Issue #625
    def __init__(self, raise_error = True):
        if raise_error:
            raise ValueError("To create a solution you should use the createSol method of the Model class.")

    @staticmethod
    cdef create(SCIP* scip, SCIP_SOL* scip_sol):
        """
        Main method for creating a Solution class. Please use createSol method of the Model class
        when wanting to create a Solution as a user.

        Parameters
        ----------
        scip : SCIP*
            A pointer to the SCIP object
        scip_sol : SCIP_SOL*
            A pointer to the SCIP_SOL

        Returns
        -------
        sol : Solution
            The Python representative of the SCIP_SOL

        """
        if scip == NULL:
            raise Warning("cannot create Solution with SCIP* == NULL")
        sol = Solution(raise_error=False)
        sol.sol = scip_sol
        sol.scip = scip
        return sol

    def __getitem__(
        self,
        expr: Union[Expr, GenExpr, MatrixExpr],
    ) -> Union[float, np.ndarray]:
        if not isinstance(expr, (Expr, GenExpr, MatrixExpr)):
            raise TypeError(
                "Argument 'expr' has incorrect type, expected 'Expr', 'GenExpr', or "
                f"'MatrixExpr', got {type(expr).__name__!r}"
            )

        self._checkStage("SCIPgetSolVal")
        return expr._evaluate(self)

    def __setitem__(self, Variable var, value):
        PY_SCIP_CALL(SCIPsetSolVal(self.scip, self.sol, var.scip_var, value))

    def __repr__(self):
        cdef SCIP_VAR* scip_var
        cdef int i
        vals = {}
        self._checkStage("SCIPgetSolVal")
        for i in range(SCIPgetNOrigVars(self.scip)):
            scip_var = SCIPgetOrigVars(self.scip)[i]
            # extract name
            cname = bytes(SCIPvarGetName(scip_var))
            name = cname.decode('utf-8')
            vals[name] = SCIPgetSolVal(self.scip, self.sol, scip_var)
        return str(vals)

    def _checkStage(self, method):
        if method in ["SCIPgetSolVal", "getSolObjVal"]:
            stage_check = SCIPgetStage(self.scip) not in [SCIP_STAGE_INIT, SCIP_STAGE_FREE]
            if not stage_check or self.sol == NULL and SCIPgetStage(self.scip) != SCIP_STAGE_SOLVING:
                raise Warning(f"{method} can only be called with a valid solution or in stage SOLVING (current stage: {SCIPgetStage(self.scip)})")

    def getOrigin(self):
        """
        Returns origin of solution: where to retrieve uncached elements.

        Returns
        -------
        PY_SCIP_SOLORIGIN
        """
        return SCIPsolGetOrigin(self.sol)

    def retransform(self):
        """ retransforms solution to original problem space """
        PY_SCIP_CALL(SCIPretransformSol(self.scip, self.sol))

    def translate(self, BaseModel target):
        """
        translate solution to a target model solution

        Parameters
        ----------
        target : Model

        Returns
        -------
        targetSol: Solution
        """
        if self.getOrigin() != SCIP_SOLORIGIN_ORIGINAL:
            PY_SCIP_CALL(SCIPretransformSol(self.scip, self.sol))
        cdef Solution targetSol = Solution.create(target._scip, NULL)
        cdef SCIP_VAR** source_vars = SCIPgetOrigVars(self.scip)
        PY_SCIP_CALL(SCIPtranslateSubSol(target._scip, self.scip, self.sol, NULL, source_vars, &(targetSol.sol)))
        return targetSol

cdef class BoundChange:
    """Bound change."""

    @staticmethod
    cdef create(SCIP_BOUNDCHG* scip_boundchg):
        """
        Main method for creating a BoundChange class. Is used instead of __init__.

        Parameters
        ----------
        scip_boundchg : SCIP_BOUNDCHG*
            A pointer to the SCIP_BOUNDCHG

        Returns
        -------
        boundchg : BoundChange
            The Python representative of the SCIP_BOUNDCHG

        """
        if scip_boundchg == NULL:
            raise Warning("cannot create BoundChange with SCIP_BOUNDCHG* == NULL")
        boundchg = BoundChange()
        boundchg.scip_boundchg = scip_boundchg
        return boundchg

    def getNewBound(self):
        """
        Returns the new value of the bound in the bound change.

        Returns
        -------
        float

        """
        return SCIPboundchgGetNewbound(self.scip_boundchg)

    def getVar(self):
        """
        Returns the variable of the bound change.

        Returns
        -------
        Variable

        """
        return Variable.create(SCIPboundchgGetVar(self.scip_boundchg))

    def getBoundchgtype(self):
        """
        Returns the bound change type of the bound change.

        Returns
        -------
        int
            (0 = branching, 1 = consinfer, 2 = propinfer)

        """
        return SCIPboundchgGetBoundchgtype(self.scip_boundchg)

    def getBoundtype(self):
        """
        Returns the bound type of the bound change.

        Returns
        -------
        int
            (0 = lower, 1 = upper)

        """
        return SCIPboundchgGetBoundtype(self.scip_boundchg)

    def isRedundant(self):
        """
        Returns whether the bound change is redundant due to a more global bound that is at least as strong.

        Returns
        -------
        bool

        """
        return SCIPboundchgIsRedundant(self.scip_boundchg)

    def __repr__(self):
        return f"{self.getVar()} {_SCIP_BOUNDTYPE_TO_STRING[self.getBoundtype()]} {self.getNewBound()}"

cdef class DomainChanges:
    """Set of domain changes."""

    @staticmethod
    cdef create(SCIP_DOMCHG* scip_domchg):
        """
        Main method for creating a DomainChanges class. Is used instead of __init__.

        Parameters
        ----------
        scip_domchg : SCIP_DOMCHG*
            A pointer to the SCIP_DOMCHG

        Returns
        -------
        domchg : DomainChanges
            The Python representative of the SCIP_DOMCHG

        """
        if scip_domchg == NULL:
            raise Warning("cannot create DomainChanges with SCIP_DOMCHG* == NULL")
        domchg = DomainChanges()
        domchg.scip_domchg = scip_domchg
        return domchg

    def getBoundchgs(self):
        """
        Returns the bound changes in the domain change.

        Returns
        -------
        list of BoundChange

        """
        cdef int nboundchgs = SCIPdomchgGetNBoundchgs(self.scip_domchg)
        cdef int i
        return [BoundChange.create(SCIPdomchgGetBoundchg(self.scip_domchg, i))
                for i in range(nboundchgs)]

cdef class Node:
    """Base class holding a pointer to corresponding SCIP_NODE"""

    @staticmethod
    cdef create(SCIP_NODE* scipnode):
        """
        Main method for creating a Node class. Is used instead of __init__.

        Parameters
        ----------
        scipnode : SCIP_NODE*
            A pointer to the SCIP_NODE

        Returns
        -------
        node : Node
            The Python representative of the SCIP_NODE

        """
        if scipnode == NULL:
            return None
        node = Node()
        node.scip_node = scipnode
        return node

    def getParent(self):
        """
        Retrieve parent node (or None if the node has no parent node).

        Returns
        -------
        Node

        """
        return Node.create(SCIPnodeGetParent(self.scip_node))

    def getNumber(self):
        """
        Retrieve number of node.

        Returns
        -------
        int

        """
        return SCIPnodeGetNumber(self.scip_node)

    def getDepth(self):
        """
        Retrieve depth of node.

        Returns
        -------
        int

        """
        return SCIPnodeGetDepth(self.scip_node)

    def getType(self):
        """
        Retrieve type of node.

        Returns
        -------
        PY_SCIP_NODETYPE

        """
        return SCIPnodeGetType(self.scip_node)

    def getLowerbound(self):
        """
        Retrieve lower bound of node.

        Returns
        -------
        float

        """
        return SCIPnodeGetLowerbound(self.scip_node)

    def getEstimate(self):
        """
        Retrieve the estimated value of the best feasible solution in subtree of the node.

        Returns
        -------
        float

        """
        return SCIPnodeGetEstimate(self.scip_node)

    def getAddedConss(self):
        """
        Retrieve all constraints added at this node.

        Returns
        -------
        list of Constraint

        """
        cdef int addedconsssize = SCIPnodeGetNAddedConss(self.scip_node)
        if addedconsssize == 0:
            return []
        cdef SCIP_CONS** addedconss = <SCIP_CONS**> malloc(addedconsssize * sizeof(SCIP_CONS*))
        cdef int nconss
        cdef int i
        SCIPnodeGetAddedConss(self.scip_node, addedconss, &nconss, addedconsssize)
        assert nconss == addedconsssize
        constraints = [Constraint.create(addedconss[i]) for i in range(nconss)]
        free(addedconss)
        return constraints

    def getNAddedConss(self):
        """
        Retrieve number of added constraints at this node.

        Returns
        -------
        int

        """
        return SCIPnodeGetNAddedConss(self.scip_node)

    def isActive(self):
        """
        Is the node in the path to the current node?

        Returns
        -------
        bool

        """
        return SCIPnodeIsActive(self.scip_node)

    def isPropagatedAgain(self):
        """
        Is the node marked to be propagated again?

        Returns
        -------
        bool

        """
        return SCIPnodeIsPropagatedAgain(self.scip_node)

    def getNParentBranchings(self):
        """
        Retrieve the number of variable branchings that were performed in the parent node to create this node.

        Returns
        -------
        int

        """
        cdef SCIP_VAR* dummy_branchvars
        cdef SCIP_Real dummy_branchbounds
        cdef SCIP_BOUNDTYPE dummy_boundtypes
        cdef int nbranchvars
        # This is a hack: the SCIP interface has no function to directly get the
        # number of parent branchings, i.e., SCIPnodeGetNParentBranchings() does
        # not exist.
        SCIPnodeGetParentBranchings(self.scip_node, &dummy_branchvars,
                                    &dummy_branchbounds, &dummy_boundtypes,
                                    &nbranchvars, 0)
        return nbranchvars

    def getParentBranchings(self):
        """
        Retrieve the set of variable branchings that were performed in the parent node to create this node.

        Returns
        -------
        list of Variable
        list of float
        list of int

        """
        cdef int nbranchvars = self.getNParentBranchings()
        if nbranchvars == 0:
            return None
        cdef SCIP_VAR** branchvars = <SCIP_VAR**> malloc(nbranchvars * sizeof(SCIP_VAR*))
        cdef SCIP_Real* branchbounds = <SCIP_Real*> malloc(nbranchvars * sizeof(SCIP_Real))
        cdef SCIP_BOUNDTYPE* boundtypes = <SCIP_BOUNDTYPE*> malloc(nbranchvars * sizeof(SCIP_BOUNDTYPE))
        cdef int i
        SCIPnodeGetParentBranchings(self.scip_node, branchvars, branchbounds,
                                    boundtypes, &nbranchvars, nbranchvars)

        py_variables = [Variable.create(branchvars[i]) for i in range(nbranchvars)]
        py_branchbounds = [branchbounds[i] for i in range(nbranchvars)]
        py_boundtypes = [boundtypes[i] for i in range(nbranchvars)]
        free(boundtypes)
        free(branchbounds)
        free(branchvars)
        return py_variables, py_branchbounds, py_boundtypes

    def getNDomchg(self):
        """
        Retrieve the number of bound changes due to branching, constraint propagation, and propagation.

        Returns
        -------
        nbranchings : int
        nconsprop : int
        nprop : int

        """
        cdef int nbranchings
        cdef int nconsprop
        cdef int nprop
        SCIPnodeGetNDomchg(self.scip_node, &nbranchings, &nconsprop, &nprop)
        return nbranchings, nconsprop, nprop

    def getDomchg(self):
        """
        Retrieve domain changes for this node.

        Returns
        -------
        DomainChanges

        """
        cdef SCIP_DOMCHG* domchg = SCIPnodeGetDomchg(self.scip_node)
        if domchg == NULL:
            return None
        return DomainChanges.create(domchg)

    def __hash__(self):
        return hash(<size_t>self.scip_node)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_node == (<Node>other).scip_node)

cdef class Variable(Expr):
    """Is a linear expression and has SCIP_VAR*"""

    @staticmethod
    cdef create(SCIP_VAR* scipvar):
        """
        Main method for creating a Variable class. Is used instead of __init__.

        Parameters
        ----------
        scipvar : SCIP_VAR*
            A pointer to the SCIP_VAR

        Returns
        -------
        var : Variable
            The Python representative of the SCIP_VAR

        """
        if scipvar == NULL:
            raise Warning("cannot create Variable with SCIP_VAR* == NULL")
        var = Variable()
        var.scip_var = scipvar
        Expr.__init__(var, {Term(var) : 1.0})
        return var

    @property
    def name(self):
        if self.scip_var == NULL:
            return ""
        return bytes(SCIPvarGetName(self.scip_var)).decode("utf-8")

    cdef Variable copy(self, bint copy=True):
        return self

    def ptr(self):
        return <size_t>(self.scip_var)

    def __repr__(self):
        return self.name

    def vtype(self):
        """
        Retrieve the variables type (BINARY, INTEGER, CONTINUOUS, or IMPLINT)

        Returns
        -------
        str
            "BINARY", "INTEGER", "CONTINUOUS", or "IMPLINT"

        """
        vartype = SCIPvarGetType(self.scip_var)
        if vartype == SCIP_VARTYPE_BINARY:
            return "BINARY"
        elif vartype == SCIP_VARTYPE_INTEGER:
            return "INTEGER"
        elif vartype == SCIP_VARTYPE_CONTINUOUS:
            return "CONTINUOUS"
        elif vartype == SCIP_DEPRECATED_VARTYPE_IMPLINT:
            return "IMPLINT"
    
    def isBinary(self):
        """
        Returns whether variable is of BINARY type.

        Returns
        -------
        bool
        """
        return SCIPvarIsBinary(self.scip_var)
    
    def isIntegral(self):
        """
        Returns whether variable is of INTEGER type.

        Returns
        -------
        bool
        """
        return SCIPvarIsIntegral(self.scip_var)
    
    def isImpliedIntegral(self):
        """
        Returns whether variable is implied integral (weakly or strongly).

        Returns
        -------
        bool
        """
        return SCIPvarIsImpliedIntegral(self.scip_var)
    
    def isNonImpliedIntegral(self):
        """
        Returns TRUE if the variable is integral, but not implied integral..

        Returns
        -------
        bool
        """
        return SCIPvarIsNonimpliedIntegral(self.scip_var)

    def getImplType(self):
        """
        Returns the implied integral type of the variable

        Returns
        -------
        PY_SCIP_IMPLINTTYPE
        """
        return SCIPvarGetImplType(self.scip_var)

    def getStatus(self):
        """
        Retrieve the variable status (ORIGINAL, LOOSE, COLUMN, FIXED, AGGREGATED, MULTAGGR, NEGATED)

        Returns
        -------
        str
            "ORIGINAL", "LOOSE", "COLUMN", "FIXED", "AGGREGATED", "MULTAGGR", "NEGATED"

        """
        varstatus = SCIPvarGetStatus(self.scip_var)
        if varstatus == SCIP_VARSTATUS_ORIGINAL:
            return "ORIGINAL"
        elif varstatus == SCIP_VARSTATUS_LOOSE:
            return "LOOSE"
        elif varstatus == SCIP_VARSTATUS_COLUMN:
            return "COLUMN"
        elif varstatus == SCIP_VARSTATUS_FIXED:
            return "FIXED"
        elif varstatus == SCIP_VARSTATUS_AGGREGATED:
            return "AGGREGATED"
        elif varstatus == SCIP_VARSTATUS_MULTAGGR:
            return "MULTAGGR"
        elif varstatus == SCIP_VARSTATUS_NEGATED:
            return "NEGATED"

    def isOriginal(self):
        """
        Retrieve whether the variable belongs to the original problem

        Returns
        -------
        bool

        """
        return SCIPvarIsOriginal(self.scip_var)

    def isActive(self):
        """
        Retrieve whether the variable is active

        Returns
        -------
        bool

        """
        return SCIPvarIsActive(self.scip_var)

    def isInLP(self):
        """
        Retrieve whether the variable is a COLUMN variable that is member of the current LP.

        Returns
        -------
        bool

        """
        return SCIPvarIsInLP(self.scip_var)

    def getIndex(self):
        """
        Retrieve the unique index of the variable.

        Returns
        -------
        int

        """
        return SCIPvarGetIndex(self.scip_var)

    def getCol(self):
        """
        Retrieve column of COLUMN variable.

        Returns
        -------
        Column

        """
        cdef SCIP_COL* scip_col
        scip_col = SCIPvarGetCol(self.scip_var)
        return Column.create(scip_col)

    def getLbOriginal(self):
        """
        Retrieve original lower bound of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetLbOriginal(self.scip_var)

    def getUbOriginal(self):
        """
        Retrieve original upper bound of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetUbOriginal(self.scip_var)

    def getLbGlobal(self):
        """
        Retrieve global lower bound of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetLbGlobal(self.scip_var)

    def getUbGlobal(self):
        """
        Retrieve global upper bound of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetUbGlobal(self.scip_var)

    def getLbLocal(self):
        """
        Retrieve current lower bound of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetLbLocal(self.scip_var)

    def getUbLocal(self):
        """
        Retrieve current upper bound of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetUbLocal(self.scip_var)

    def getObj(self):
        """
        Retrieve current objective value of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetObj(self.scip_var)

    def getLPSol(self):
        """
        Retrieve the current LP solution value of variable.

        Returns
        -------
        float

        """
        return SCIPvarGetLPSol(self.scip_var)

    def getAvgSol(self):
        """
        Get the weighted average solution of variable in all feasible primal solutions found.

        Returns
        -------
        float

        """
        return SCIPvarGetAvgSol(self.scip_var)

    def markRelaxationOnly(self):
        """
        marks that this variable has only been introduced to define a relaxation

        The variable must not have a coefficient in the objective and must be deletable.
        If it is not marked deletable, it will be marked as deletable, which is only possible before
        the variable is added to a problem.

        Returns
        -------
        None

        """
        SCIPvarMarkRelaxationOnly(self.scip_var)
    
    def isRelaxationOnly(self):
        """
        returns whether a variable has been introduced to define a relaxation

        These variables are only valid for the current SCIP solve round, they are not contained in any (checked)
        constraints, but may be used in cutting planes, for example. Relaxation-only variables are not copied 
        by SCIPcopyVars and cuts that contain these variables are not added as linear constraints when 
        restarting or transferring information from a copied SCIP to a SCIP. Also conflicts with relaxation-only
        variables are not generated at the moment. Relaxation-only variables do not appear in the objective.

        Returns
        -------
        bool

        """
        return SCIPvarIsRelaxationOnly(self.scip_var)
    
    def isDeletable(self):
        """
        Returns whether variable is allowed to be deleted completely from the problem.

        Returns
        -------
        bool

        """
        return SCIPvarIsDeletable(self.scip_var)

    def isActive(self):
        """
        Returns whether variable is an active (neither fixed nor aggregated) variable.

        Returns
        -------
        boolean
        """
        return SCIPvarIsActive(self.scip_var)

    def getNLocksDown(self):
        """
        Returns the number of locks for rounding down.

        Returns
        -------
        int

        """
        return SCIPvarGetNLocksDown(self.scip_var)
    
    def getNLocksUp(self):
        """
        Returns the number of locks for rounding up.

        Returns
        -------
        int

        """
        return SCIPvarGetNLocksUp(self.scip_var)

    def getNLocksDownType(self, locktype):
        """
        Returns the number of locks for rounding down of a certain type.

        Parameters
        ----------
        locktype : SCIP_LOCKTYPE
            type of variable locks

        Returns
        -------
        int

        """
        return SCIPvarGetNLocksDownType(self.scip_var, locktype)

    def getNLocksUpType(self, locktype):
        """
        Returns the number of locks for rounding up of a certain type.

        Parameters
        ----------
        locktype : SCIP_LOCKTYPE
            type of variable locks

        Returns
        -------
        int

        """
        return SCIPvarGetNLocksUpType(self.scip_var, locktype)

    def varMayRound(self, direction="down"):
        """
        Checks whether it is possible to round variable up / down and stay feasible for the relaxation.

        Parameters
        ----------
        direction : str
            "up" or "down"

        Returns
        -------
        bool

        """
        if direction not in ("down", "up"):
            raise Warning(f"Unrecognized direction for rounding: {direction}")
        cdef SCIP_Bool mayround
        if direction == "down":
            mayround = SCIPvarMayRoundDown(self.scip_var)
        else:
            mayround = SCIPvarMayRoundUp(self.scip_var)
        return mayround

    def getNBranchings(self, branchdir):
        """
        returns the number of times a bound of the variable was changed in given direction due to branching

        Parameters
        ----------
        branchdir : PY_SCIP_BRANCHDIR
            branching direction (downwards, or upwards)

        Returns
        -------
        int
        """
        return SCIPvarGetNBranchings(self.scip_var, branchdir)

    def getNBranchingsCurrentRun(self, branchdir):
        """
        returns the number of times a bound of the variable was changed in given direction due to branching in the
        current run

        Parameters
        ----------
        branchdir : PY_SCIP_BRANCHDIR
            branching direction (downwards, or upwards)

        Returns
        -------
        int
        """
        return SCIPvarGetNBranchingsCurrentRun(self.scip_var, branchdir)

class MatrixVariable(MatrixExpr):

    def vtype(self):
        """
        Retrieve the matrix variables type (BINARY, INTEGER, IMPLINT or CONTINUOUS)

        Returns
        -------
        np.ndarray
            A matrix containing "BINARY", "INTEGER", "CONTINUOUS", or "IMPLINT"

        """
        vtypes = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            vtypes[idx] = self[idx].vtype()
        return vtypes

    def isInLP(self):
        """
        Retrieve whether the matrix variable is a COLUMN variable that is member of the current LP.

        Returns
        -------
        np.ndarray
            An array of bools

        """
        in_lp = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self.shape):
            in_lp[idx] = self[idx].isInLP()
        return in_lp


    def getIndex(self):
        """
        Retrieve the unique index of the matrix variable.

        Returns
        -------
        np.ndarray
            An array of integers. No two should be the same
        """
        indices = np.empty(self.shape, dtype=int)
        for idx in np.ndindex(self.shape):
            indices[idx] = self[idx].getIndex()
        return indices

    def getCol(self):
        """
        Retrieve matrix of columns of COLUMN variables.

        Returns
        -------
        np.ndarray
            An array of Column objects
        """

        columns = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            columns[idx] = self[idx].getCol()
        return columns

    def getLbOriginal(self):
        """
        Retrieve original lower bound of matrix variable.

        Returns
        -------
        np.ndarray

        """
        lbs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            lbs[idx] = self[idx].getLbOriginal()
        return lbs

    def getUbOriginal(self):
        """
        Retrieve original upper bound of matrixvariable.

        Returns
        -------
        np.ndarray

        """
        ubs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            ubs[idx] = self[idx].getUbOriginal()
        return ubs

    def getLbGlobal(self):
        """
        Retrieve global lower bound of matrix variable.

        Returns
        -------
        np.ndarray

        """
        lbs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            lbs[idx] = self[idx].getLbGlobal()
        return lbs

    def getUbGlobal(self):
        """
        Retrieve global upper bound of matrix variable.

        Returns
        -------
        np.ndarray

        """
        ubs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            ubs[idx] = self[idx].getUbGlobal()
        return ubs

    def getLbLocal(self):
        """
        Retrieve current lower bound of matrix variable.

        Returns
        -------
        np.ndarray

        """
        lbs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            lbs[idx] = self[idx].getLbLocal()
        return lbs

    def getUbLocal(self):
        """
        Retrieve current upper bound of matrix variable.

        Returns
        -------
        np.ndarray

        """
        ubs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            ubs[idx] = self[idx].getUbLocal()
        return ubs

    def getObj(self):
        """
        Retrieve current objective value of matrix variable.

        Returns
        -------
        np.ndarray

        """
        objs = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            objs[idx] = self[idx].getObj()
        return objs

    def getLPSol(self):
        """
        Retrieve the current LP solution value of matrix variable.

        Returns
        -------
        np.ndarray

        """
        lpsols = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            lpsols[idx] = self[idx].getLPSol()
        return lpsols

    def getAvgSol(self):
        """
        Get the weighted average solution of matrix variable in all feasible primal solutions found.

        Returns
        -------
        np.ndarray

        """
        avgsols = np.empty(self.shape, dtype=float)
        for idx in np.ndindex(self.shape):
            avgsols[idx] = self[idx].getAvgSol()
        return avgsols

    def varMayRound(self, direction="down"):
        """
        Checks whether it is possible to round variable up / down and stay feasible for the relaxation.

        Parameters
        ----------
        direction : str
            "up" or "down"

        Returns
        -------
        np.ndarray
            An array of bools

        """
        mayround = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self.shape):
            mayround[idx] = self[idx].varMayRound(direction)
        return mayround

cdef class Constraint:
    """Base class holding a pointer to corresponding SCIP_CONS"""

    @staticmethod
    cdef create(SCIP_CONS* scipcons):
        """
        Main method for creating a Constraint class. Is used instead of __init__.

        Parameters
        ----------
        scipcons : SCIP_CONS*
            A pointer to the SCIP_CONS

        Returns
        -------
        cons : Constraint
            The Python representative of the SCIP_CONS

        """
        if scipcons == NULL:
            raise Warning("cannot create Constraint with SCIP_CONS* == NULL")
        cons = Constraint()
        cons.scip_cons = scipcons
        return cons

    @property
    def name(self):
        if self.scip_cons == NULL:
            return ""
        return bytes(SCIPconsGetName(self.scip_cons)).decode("utf-8")

    def ptr(self):
        return <size_t>(self.scip_cons)

    def __repr__(self):
        return self.name

    def isOriginal(self):
        """
        Retrieve whether the constraint belongs to the original problem.

        Returns
        -------
        bool

        """
        return SCIPconsIsOriginal(self.scip_cons)

    def isInitial(self):
        """
        Returns True if the relaxation of the constraint should be in the initial LP.

        Returns
        -------
        bool

        """
        return SCIPconsIsInitial(self.scip_cons)

    def isSeparated(self):
        """
        Returns True if constraint should be separated during LP processing.

        Returns
        -------
        bool

        """
        return SCIPconsIsSeparated(self.scip_cons)

    def isEnforced(self):
        """
        Returns True if constraint should be enforced during node processing.

        Returns
        -------
        bool

        """
        return SCIPconsIsEnforced(self.scip_cons)

    def isChecked(self):
        """
        Returns True if constraint should be checked for feasibility.

        Returns
        -------
        bool

        """
        return SCIPconsIsChecked(self.scip_cons)

    def isPropagated(self):
        """
        Returns True if constraint should be propagated during node processing.

        Returns
        -------
        bool

        """
        return SCIPconsIsPropagated(self.scip_cons)

    def isLocal(self):
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        bool

        """
        return SCIPconsIsLocal(self.scip_cons)

    def isModifiable(self):
        """
        Returns True if constraint is modifiable (subject to column generation).

        Returns
        -------
        bool

        """
        return SCIPconsIsModifiable(self.scip_cons)

    def isDynamic(self):
        """
        Returns True if constraint is subject to aging.

        Returns
        -------
        bool

        """
        return SCIPconsIsDynamic(self.scip_cons)

    def isRemovable(self):
        """
        Returns True if constraint's relaxation should be removed from the LP due to aging or cleanup.

        Returns
        -------
        bool

        """
        return SCIPconsIsRemovable(self.scip_cons)

    def isStickingAtNode(self):
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        bool

        """
        return SCIPconsIsStickingAtNode(self.scip_cons)

    def isActive(self):
        """
        Returns True iff constraint is active in the current node.

        Returns
        -------
        bool

        """
        return SCIPconsIsActive(self.scip_cons)

    def isLinear(self):
        """
        Returns True if constraint is linear

        Returns
        -------
        bool

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'linear'
    
    def isKnapsack(self):
        """
        Returns True if constraint is a knapsack constraint.
        This is a special case of a linear constraint.

        Returns
        -------
        bool

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'knapsack'

    def isCumulative(self):
        """
        Returns True if constraint is a cumulative constraint.
        Cumulative is typically used in scheduling applications.

        Returns
        -------
        bool

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'cumulative'

    def isLinearType(self):
        """
        Returns True if constraint can be represented as a linear constraint.

        Returns
        -------
        bool

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype in ('linear', 'knapsack', 'setppc', 'logicor', 'varbound')

    def isNonlinear(self):
        """
        Returns True if constraint is nonlinear.

        Returns
        -------
        bool

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'nonlinear'

    def getConshdlrName(self):
        """
        Return the constraint handler's name.

        Returns
        -------
        str

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype

    def __hash__(self):
        return hash(<size_t>self.scip_cons)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_cons == (<Constraint>other).scip_cons)

class MatrixConstraint(np.ndarray):

    def isInitial(self):
        """
        Returns True if the relaxation of the constraint should be in the initial LP.

        Returns
        -------
        np.ndarray

        """
        initial = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            initial[idx] = self[idx].isInitial()
        return initial

    def isSeparated(self):
        """
        Returns True if constraint should be separated during LP processing.

        Returns
        -------
        np.ndarray

        """
        separated = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            separated[idx] = self[idx].isSeparated()
        return separated

    def isEnforced(self):
        """
        Returns True if constraint should be enforced during node processing.

        Returns
        -------
        np.ndarray

        """
        enforced = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            enforced[idx] = self[idx].isEnforced()
        return enforced

    def isChecked(self):
        """
        Returns True if constraint should be checked for feasibility.

        Returns
        -------
        np.ndarray

        """
        checked = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            checked[idx] = self[idx].isCheced()
        return checked

    def isPropagated(self):
        """
        Returns True if constraint should be propagated during node processing.

        Returns
        -------
        np.ndarray

        """
        propagated = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            propagated[idx] = self[idx].isPropagated()
        return propagated

    def isLocal(self):
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        np.ndarray

        """
        local = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            local[idx] = self[idx].isLocal()
        return local

    def isModifiable(self):
        """
        Returns True if constraint is modifiable (subject to column generation).

        Returns
        -------
        np.ndarray

        """
        modifiable = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            modifiable[idx] = self[idx].isModifiable()
        return modifiable

    def isDynamic(self):
        """
        Returns True if constraint is subject to aging.

        Returns
        -------
        np.ndarray

        """
        dynamic = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            dynamic[idx] = self[idx].isDynamic()
        return dynamic

    def isRemovable(self):
        """
        Returns True if constraint's relaxation should be removed from the LP due to aging or cleanup.

        Returns
        -------
        np.ndarray

        """
        removable = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            removable[idx] = self[idx].isRemovable()
        return removable

    def isStickingAtNode(self):
        """
        Returns True if constraint is only locally valid or not added to any (sub)problem.

        Returns
        -------
        np.ndarray

        """
        stickingatnode = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            stickingatnode[idx] = self[idx].isStickingAtNode()
        return stickingatnode

    def isActive(self):
        """
        Returns True iff constraint is active in the current node.

        Returns
        -------
        np.ndarray

        """
        active = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            active[idx] = self[idx].isActive()
        return active

    def isLinear(self):
        """
        Returns True if constraint is linear

        Returns
        -------
        np.ndarray

        """
        islinear = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            islinear[idx] = self[idx].isLinear()
        return islinear

    def isNonlinear(self):
        """
        Returns True if constraint is nonlinear.

        Returns
        -------
        np.ndarray

        """
        isnonlinear = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            isnonlinear[idx] = self[idx].isNonlinear()
        return isnonlinear

    def getConshdlrName(self):
        """
        Return the constraint handler's name.

        Returns
        -------
        np.ndarray

        """
        name = np.empty(self.shape, dtype=bool)
        for idx in np.ndindex(self):
            name[idx] = self[idx].getConshdlrName()
        return name

cdef void relayMessage(SCIP_MESSAGEHDLR *messagehdlr, FILE *file, const char *msg) noexcept:
    if file is stdout:
        sys.stdout.write(msg.decode('UTF-8'))
    elif file is stderr:
        sys.stderr.write(msg.decode('UTF-8'))
    else:
        if msg is not NULL:
            fputs(msg, file)
        fflush(file)

cdef void relayErrorMessage(void *messagehdlr, FILE *file, const char *msg) noexcept:
    if file is NULL:
        sys.stderr.write(msg.decode('UTF-8'))
    else:
        if msg is not NULL:
            fputs(msg, file)
        fflush(file)

cdef class _VarArray:
    cdef SCIP_VAR** ptr
    cdef int size

    def __cinit__(self, object vars):
        self.ptr = NULL
        self.size = 0

        if isinstance(vars, Variable):
            vars = [vars]
        elif not isinstance(vars, (list, tuple, MatrixVariable)):
            raise TypeError(f"Expected Variable or list of Variable, got {type(vars)}.")

        self.size = len(vars)
        if self.size:
            self.ptr = <SCIP_VAR**> malloc(self.size * sizeof(SCIP_VAR*))
            for i, var in enumerate(vars):
                if not isinstance(var, Variable):
                    if (ndim := np.ndim(vars)) != 1:
                        raise ValueError(f"Expected a 1D array, but got a {ndim}D array.")
                    raise TypeError(f"Expected Variable, got {type(var)}.")
                self.ptr[i] = (<Variable>var).scip_var

    def __dealloc__(self):
        if self.ptr != NULL:
            free(self.ptr)

cdef class IIS:

    @staticmethod
    cdef create(SCIP_IIS* scip_iis):
        """
        Main method for creating an IIS class.

        Parameters
        ----------
        scip : SCIP_IIS*
            A pointer to the SCIP_IIS

        Returns
        -------
        sol : IIS
            The Python representative of the IIS

        """
        iis = IIS()
        iis._iis = scip_iis
        return iis
    
    def getTime(self):
        """
        Retrieve the solving time of the IIS.

        Returns
        -------
        float
        """
        return SCIPiisGetTime(self._iis)

    def isSubscipIrreducible(self):
        """
        Returns whether the IIS is irreducible.

        Returns
        -------
        bool
        """
        return SCIPiisIsSubscipIrreducible(self._iis)

    def isSubscipInfeasible(self):
        """
        Returns whether the IIS is infeasible.

        Returns
        -------
        bool
        """
        return SCIPiisIsSubscipInfeasible(self._iis)
    
    def setSubscipIrreducible(self, irreducible):
        """
        Sets the flag that states whether the IIS subscip is irreducible.

        Parameters
        ----------
        irreducible : bool
            the value to set the irreducible flag to
        """

        SCIPiisSetSubscipIrreducible(self._iis, irreducible)

    def setSubscipInfeasible(self, infeasible):
        """
        Sets the flag that states whether the IIS subscip is infeasible.

        Parameters
        ----------
        infeasible : bool
            the value to set the infeasible flag to
        """

        SCIPiisSetSubscipInfeasible(self._iis, infeasible)

    def getNNodes(self):
        """
        Gets number of nodes in the IIS solve.

        Returns
        -------
        int

        """
        return SCIPiisGetNNodes(self._iis)
    
    def getSubscip(self):
        """
        Get the subscip of an IIS.

        Returns
        -------
        Model
        """
        cdef SCIP* subscip

        subscip = SCIPiisGetSubscip(self._iis)
        model = BaseModel.create(subscip)
        return model

    def greedyMakeIrreducible(self):
       """
       Perform the greedy deletion algorithm with singleton batches to obtain an irreducible infeasible subsystem (IIS)
       """

       PY_SCIP_CALL(SCIPiisGreedyMakeIrreducible(self._iis))


# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
##
#@anchor Model
##
cdef class BaseModel:

    def __init__(self, problemName='model', defaultPlugins=True, BaseModel sourceModel=None, origcopy=False, globalcopy=True, enablepricing=True, createscip=True, threadsafe=False):
        """
        Main class holding a pointer to SCIP for managing most interactions

        Parameters
        ----------
        problemName : str, optional
            name of the problem (default 'model')
        defaultPlugins : bool, optional
            use default plugins? (default True)
        sourceModel : Model or None, optional
            create a copy of the given Model instance (default None)
        origcopy : bool, optional
            whether to call copy or copyOrig (default False)
        globalcopy : bool, optional
            whether to create a global or a local copy (default True)
        enablepricing : bool, optional
            whether to enable pricing in copy (default True)
        createscip : bool, optional
            initialize the Model object and creates a SCIP instance (default True)
        threadsafe : bool, optional
            False if data can be safely shared between the source and target problem (default False)

        """
        if self.getMajorVersion() < MAJOR:
            raise Exception("linked SCIP is not compatible to this version of PySCIPOpt - use at least version", MAJOR)
        if self.getMajorVersion() == MAJOR and self.getMinorVersion() < MINOR:
            warnings.warn(
                "linked SCIP {}.{} is not recommended for this version of PySCIPOpt - use version {}.{}.{}".format(
                    self.getMajorVersion(), self.getMinorVersion(), MAJOR, MINOR, PATCH))

        self._freescip = True
        self._modelvars = {}
        self._modelconss = {}
        self._generated_event_handlers_count = 0
        self._benders_subproblems = []  # Keep references to Benders subproblem Models
        self._plugins = []  # Keep references to plugins to break cycles in __dealloc__
        self._iis = NULL

        if not createscip:
            # if no SCIP instance should be created, then an empty Model object is created.
            self._scip = NULL
            self._bestSol = None
            self._freescip = False
        elif sourceModel is None:
            PY_SCIP_CALL(SCIPcreate(&self._scip))
            self._bestSol = None
            if defaultPlugins:
                self.includeDefaultPlugins()
            self._configureScip()
            self.createProbBasic(problemName)
        else:
            PY_SCIP_CALL(SCIPcreate(&self._scip))
            self._bestSol = <Solution> sourceModel._bestSol
            n = str_conversion(problemName)
            if origcopy:
                PY_SCIP_CALL(SCIPcopyOrig(sourceModel._scip, self._scip, NULL, NULL, n, enablepricing, threadsafe, True, self._valid))
            else:
                PY_SCIP_CALL(SCIPcopy(sourceModel._scip, self._scip, NULL, NULL, n, globalcopy, enablepricing, threadsafe, True, self._valid))

    def attachEventHandlerCallback(self,
        callback,
        events,
        name="eventhandler",
        description=""
        ):
        """
        Attach an event handler to the model using a callback function.

        Parameters
        ----------
        callback : callable
            The callback function to be called when an event occurs.
            The callback function should have the following signature:
            callback(model, event)
        events : list of SCIP_EVENTTYPE
            List of event types to attach the event handler to.
        name : str, optional
            Name of the event handler. If not provided, a unique default name will be generated.
        description : str, optional
            Description of the event handler. If not provided, an empty string will be used.
        """

        self._generated_event_handlers_count += 1
        model = self

        class EventHandler(Eventhdlr):
            def __init__(self, callback):
                super(EventHandler, self).__init__()
                self.callback = callback

            def eventinit(self):
                for event in events:
                    self.model.catchEvent(event, self)

            def eventexit(self):
                for event in events:
                    self.model.dropEvent(event, self)

            def eventexec(self, event):
                self.callback(model, event)

        event_handler = EventHandler(callback)

        if name == "eventhandler":
            name = f"eventhandler_{self._generated_event_handlers_count}"

        self.includeEventhdlr(event_handler, name, description)

    def __del__(self):
        """Free SCIP when the last external reference to the Model is dropped.

        Runs before the garbage collector clears the Model's attributes, so
        plugin callbacks can still reach a live ``self.model``. See
        ``Model.free`` for the full lifecycle policy.
        """
        self._free_scip_instance()

    cdef _configureScip(self):
        pass

    def __dealloc__(self):
        """Fallback SCIP cleanup if ``__del__`` did not already free the instance."""
        if self._scip is not NULL and self._freescip:
            SCIPfree(&self._scip)

    cdef _free_scip_instance(self):
        """Free the SCIP instance. Does not touch Python object references."""
        cdef SCIP_BENDERS** benders
        cdef int nbenders
        cdef int nsubproblems
        cdef int i, j

        if self._scip is not NULL and self._freescip and PY_SCIP_CALL:
            # Free Benders subproblems before freeing the main SCIP instance
            if self._benders_subproblems:
                benders = SCIPgetBenders(self._scip)
                nbenders = SCIPgetNActiveBenders(self._scip)

                for i in range(nbenders):
                    nsubproblems = SCIPbendersGetNSubproblems(benders[i])
                    for j in range(nsubproblems):
                        PY_SCIP_CALL(SCIPfreeBendersSubproblem(self._scip, benders[i], j))

            if SCIPgetStage(self._scip) == SCIP_STAGE_SOLVING and SCIPinProbing(self._scip):
                PY_SCIP_CALL(SCIPendProbing(self._scip))
                
            # Ignore SCIPfree retcode: cleanup must not turn into a new failure.
            SCIPfree(&self._scip)
            self._scip = NULL
            self._freescip = False

            # Invalidate all variable and constraint pointers after freeing SCIP. See issue #604.
            if self._modelvars:
                for var in self._modelvars.values():
                    (<Variable>var).scip_var = NULL
            if self._modelconss:
                for cons in self._modelconss.values():
                    (<Constraint>cons).scip_cons = NULL

    def __hash__(self):
        return hash(<size_t>self._scip)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self._scip == (<BaseModel>other)._scip)

    @staticmethod
    cdef create(SCIP* scip):
        """
        Creates a model and appropriately assigns the scip and bestsol parameters.

        Parameters
        ----------
        scip : SCIP*
            A pointer to a SCIP object

        Returns
        -------
        Model

        """
        if scip == NULL:
            raise Warning("cannot create Model with SCIP* == NULL")
        model = Model(createscip=False)
        model._scip = scip
        model._bestSol = Solution.create(scip, SCIPgetBestSol(scip))
        model._benders_subproblems = []  # Initialize Benders subproblems list
        return model

    cdef _getOrCreateCons(self, SCIP_CONS* scip_cons):
        """Get existing Constraint wrapper or create and track a new one."""
        cdef size_t ptr = <size_t>scip_cons
        if ptr in self._modelconss:
            return self._modelconss[ptr]
        pyCons = Constraint.create(scip_cons)
        self._modelconss[ptr] = pyCons
        return pyCons

    cdef _getOrCreateVar(self, SCIP_VAR* scip_var):
        """Get existing Variable wrapper or create and track a new one."""
        cdef size_t ptr = <size_t>scip_var
        if ptr in self._modelvars:
            return self._modelvars[ptr]
        pyVar = Variable.create(scip_var)
        self._modelvars[ptr] = pyVar
        return pyVar

    @property
    def _freescip(self):
        """
        Return whether the underlying Scip pointer gets deallocted when the current
        object is deleted.

        Returns
        -------
        bool

        """
        return self._freescip

    @_freescip.setter
    def _freescip(self, val):
        """
        Set whether the underlying Scip pointer gets deallocted when the current
        object is deleted.

        Parameters
        ----------
        val : bool

        """
        self._freescip = val

    @cython.always_allow_keywords(True)
    @staticmethod
    def from_ptr(capsule, take_ownership):
        """
        Create a Model from a given pointer.

        Parameters
        ----------
        capsule
            The PyCapsule containing the SCIP pointer under the name "scip"
        take_ownership : bool
            Whether the newly created Model assumes ownership of the
            underlying Scip pointer (see ``_freescip``)

        Returns
        -------
        Model

        """
        if not PyCapsule_IsValid(capsule, "scip"):
            raise ValueError("The given capsule does not contain a valid scip pointer")
        model = BaseModel.create(<SCIP*>PyCapsule_GetPointer(capsule, "scip"))
        model._freescip = take_ownership
        return model

    @cython.always_allow_keywords(True)
    def to_ptr(self, give_ownership):
        """
        Return the underlying Scip pointer to the current Model.

        Parameters
        ----------
        give_ownership : bool
            Whether the current Model gives away ownership of the
            underlying Scip pointer (see ``_freescip``)

        Returns
        -------
        capsule
            The underlying pointer to the current Model, wrapped in a
            PyCapsule under the name "scip".

        """
        capsule = PyCapsule_New(<void*>self._scip, "scip", NULL)
        if give_ownership:
            self._freescip = False
        return capsule

    def includeDefaultPlugins(self):
        """Includes all default plug-ins into SCIP."""
        PY_SCIP_CALL(SCIPincludeDefaultPlugins(self._scip))

    def createProbBasic(self, problemName='model'):
        """
        Create new problem instance with given name.

        Parameters
        ----------
        problemName : str, optional
            name of model or problem (Default value = 'model')

        """
        n = str_conversion(problemName)
        PY_SCIP_CALL(SCIPcreateProbBasic(self._scip, n))

    def freeProb(self):
        """Frees problem and solution process data."""
        PY_SCIP_CALL(SCIPfreeProb(self._scip))

    def free(self):
        """Explicitly free the SCIP instance and release plugin references.

        Every included plugin holds a strong reference to its Model via
        ``self.model``, and the Model holds strong references to every plugin
        via ``Model._plugins``. This cycle is intentional: it keeps both sides
        alive during SCIP teardown so callbacks such as ``eventexit`` and
        ``consfree`` can safely reach ``self.model``.

        The cycle is broken in one of two ways:

        * Implicitly, via ``Model.__del__`` when the last external reference to
          the Model is dropped. ``__del__`` runs before the garbage collector
          (GC) clears attributes, so callbacks still see a live ``self.model``.
          Timing is at the GC's discretion.
        * Explicitly, by calling this method. SCIP is freed immediately,
          teardown callbacks fire before ``free`` returns, and the plugin list
          is cleared. Use this when deterministic cleanup is required. After
          ``free`` the Model must not be used further.

        The solve itself is not affected by which path runs. ``free`` only
        controls when memory and resources are released, and when teardown
        callbacks execute.
        """
        self._free_scip_instance()

        # Break circular references with plugins
        if self._plugins:
            for plugin in self._plugins:
                plugin.model = None
            self._plugins = []

        # Clear Python-side caches
        self._modelvars = {}
        self._modelconss = {}
        self._benders_subproblems = []
        self._bestSol = None

    def freeTransform(self):
        """Frees all solution process data including presolving and
        transformed problem, only original problem is kept."""
        if self.getStage() not in [SCIP_STAGE_INIT,
                                 SCIP_STAGE_PROBLEM,
                                 SCIP_STAGE_TRANSFORMED,
                                 SCIP_STAGE_PRESOLVING,
                                 SCIP_STAGE_PRESOLVED,
                                 SCIP_STAGE_SOLVING,
                                 SCIP_STAGE_SOLVED]:
            raise Warning("method cannot be called in stage %i." % self.getStage())

        # Invalidate transformed variables. See issue #604.
        origvars = {ptr: var for ptr, var in self._modelvars.items() if var.isOriginal()}
        for ptr, var in self._modelvars.items():
            if ptr not in origvars:
                (<Variable>var).scip_var = NULL
        self._modelvars = origvars

        # Invalidate transformed constraints. See issue #604.
        origconss = {ptr: cons for ptr, cons in self._modelconss.items() if cons.isOriginal()}
        for ptr, cons in self._modelconss.items():
            if ptr not in origconss:
                (<Constraint>cons).scip_cons = NULL
        self._modelconss = origconss

        PY_SCIP_CALL(SCIPfreeTransform(self._scip))

    def version(self):
        """
        Retrieve SCIP version.

        Returns
        -------
        float

        """
        return SCIPversion()

    def getMajorVersion(self):
        """
        Retrieve SCIP major version.

        Returns
        -------
        int

        """
        return SCIPmajorVersion()

    def getMinorVersion(self):
        """
        Retrieve SCIP minor version.

        Returns
        -------
        int

        """
        return SCIPminorVersion()

    def getTechVersion(self):
        """
        Retrieve SCIP technical version.

        Returns
        -------
        int

        """
        return SCIPtechVersion()

    def printVersion(self):
        """Print version, copyright information and compile mode."""
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        SCIPprintVersion(self._scip, NULL)

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def printExternalCodeVersions(self):
        """Print external code versions, e.g. symmetry, non-linear solver, lp solver."""
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        SCIPprintExternalCodes(self._scip, NULL)

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def getProbName(self):
        """
        Retrieve problem name.

        Returns
        -------
        str

        """
        return bytes(SCIPgetProbName(self._scip)).decode('UTF-8')

    def getTotalTime(self):
        """
        Retrieve the current total SCIP time in seconds,
        i.e. the total time since the SCIP instance has been created.

        Returns
        -------
        float

        """
        return SCIPgetTotalTime(self._scip)

    def getSolvingTime(self):
        """
        Retrieve the current solving time in seconds.

        Returns
        -------
        float

        """
        return SCIPgetSolvingTime(self._scip)

    def getReadingTime(self):
        """
        Retrieve the current reading time in seconds.

        Returns
        -------
        float

        """
        return SCIPgetReadingTime(self._scip)

    def getPresolvingTime(self):
        """
        Returns the current presolving time in seconds.

        Returns
        -------
        float

        """
        return SCIPgetPresolvingTime(self._scip)

    def getDeterministicTime(self):
        """
        Computes a deterministic measure of time from statistics.

        Returns
        -------
        float

        """
        return SCIPgetDeterministicTime(self._scip)

    def getNLPIterations(self):
        """
        Returns the total number of LP iterations so far.

        Returns
        -------
        int

        """
        return SCIPgetNLPIterations(self._scip)

    def getNRuns(self):
        """
        Gets number of branch and bound runs performed, including the current run

        Returns
        -------
        int

        """
        return SCIPgetNRuns(self._scip)

    def getNReoptRuns(self):
        """
        Gets number of reoptimization runs performed, including the current run

        Returns
        -------
        int

        """
        return SCIPgetNReoptRuns(self._scip)

    def addNNodes(self, nnodes):
        """
        Add given number to the number of processed nodes in current run and in all runs, including the focus node

        Parameters
        ----------
        nnodes : int

        """
        SCIPaddNNodes(self._scip, nnodes)

    def getNNodes(self):
        """
        Gets number of processed nodes in current run, including the focus node.

        Returns
        -------
        int

        """
        return SCIPgetNNodes(self._scip)

    def getNTotalNodes(self):
        """
        Gets number of processed nodes in all runs, including the focus node.

        Returns
        -------
        int

        """
        return SCIPgetNTotalNodes(self._scip)

    def getNFeasibleLeaves(self):
        """
        Retrieve number of leaf nodes processed with feasible relaxation solution.

        Returns
        -------
        int

        """
        return SCIPgetNFeasibleLeaves(self._scip)

    def getNInfeasibleLeaves(self):
        """
        Gets number of infeasible leaf nodes processed.

        Returns
        -------
        int

        """
        return SCIPgetNInfeasibleLeaves(self._scip)

    def getNObjlimLeaves(self):
        """
        Gets number of processed leaf nodes that hit LP objective limit.

        Returns
        -------
        int

        """
        return SCIPgetNObjlimLeaves(self._scip)

    def getNLeaves(self):
        """
        Gets number of leaves in the tree.

        Returns
        -------
        int

        """
        return SCIPgetNLeaves(self._scip)

    def getNNodesLeft(self):
        """
        Gets number of nodes left in the tree (children + siblings + leaves)

        Returns
        -------
        int

        """
        return SCIPgetNNodesLeft(self._scip)

    def getNChildren(self):
        """
        Gets number of children of focus node.

        Returns
        -------
        int

        """
        return SCIPgetNChildren(self._scip)

    def getNSiblings(self):
        """
        Gets number of siblings of focus node.

        Returns
        -------
        int

        """
        return SCIPgetNSiblings(self._scip)

    def getFocusNode(self):
        """
        Gets focus node in the tree.
        If we are in probing/diving mode this method returns the node in the tree where the probing/diving mode was started.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetFocusNode(self._scip))

    def getFocusDepth(self):
        """
        Gets depth of the focus node, or -1 if no focus node exists.
        The focus node is the currently processed node in the branching tree, excluding the nodes of the probing path.

        Returns
        -------
        int
            The depth of the focus node, or -1 if no focus node exists.

        """
        return SCIPgetFocusDepth(self._scip)

    def getRootNode(self):
        """
        Gets the root node of the tree, or None if no root node exists.

        Returns
        -------
        Node or None
            The root node of the search tree, or None if no root node exists
            (e.g., after the problem has been solved to optimality).

        """
        return Node.create(SCIPgetRootNode(self._scip))

    def getCurrentNode(self):
        """
        Retrieve current node.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetCurrentNode(self._scip))

    def wasNodeLastBranchParent(self, Node node=None):
        """
        Checks whether the node was the parent in the most recent branching of the tree.

        Parameters
        ----------
        node : Node or None
            A node of this model, or None to check the focus node.

        Returns
        -------
        bool
            Whether the node was the last branching parent. False if no branching
            has occurred or no focus node exists when node is None.

        """
        cdef SCIP_NODE* scip_node = NULL
        if node is not None:
            scip_node = node.scip_node
        return bool(SCIPwasNodeLastBranchParent(self._scip, scip_node))

    def nodesSharePath(self, Node node1 not None, Node node2 not None):
        """
        Checks for two nodes whether they share the same root path, i.e., whether one is an ancestor of the other.
        If node1 and node2 are the same node, True is returned.

        Parameters
        ----------
        node1 : Node
            First node to compare.
        node2 : Node
            Second node to compare.

        Returns
        -------
        bool
            Whether the nodes share a root path.

        """
        return bool(SCIPnodesSharePath(node1.scip_node, node2.scip_node))

    def nodesGetCommonAncestor(self, Node node1 not None, Node node2 not None):
        """
        Finds the common ancestor node of two given nodes, or None if no common ancestor is found.
        If node1 and node2 are the same node, that node is returned.

        Parameters
        ----------
        node1 : Node
            First node to compare.
        node2 : Node
            Second node to compare.

        Returns
        -------
        Node or None
            The common ancestor closest to the given nodes, or None if no common ancestor is found
            (e.g., if the nodes belong to different trees).

        """
        return Node.create(SCIPnodesGetCommonAncestor(node1.scip_node, node2.scip_node))

    def getGap(self):
        """
        Retrieve the gap,
        i.e. abs((primalbound - dualbound)/min(abs(primalbound),abs(dualbound)))

        Returns
        -------
        float

        """
        return SCIPgetGap(self._scip)

    def getDepth(self):
        """
        Retrieve the depth of the current node.

        Returns
        -------
        int

        """
        return SCIPgetDepth(self._scip)

    def getMaxDepth(self):
        """
        Gets maximal depth of the branch-and-bound tree processed during solving (excluding probing nodes).

        Returns
        -------
        int

        """
        return SCIPgetMaxDepth(self._scip)

    def getMaxTotalDepth(self):
        """
        Gets maximal depth of all processed nodes over all branch and bound runs.

        Returns
        -------
        int

        """
        return SCIPgetMaxTotalDepth(self._scip)

    def getNBacktracks(self):
        """
        Gets total number of backtracks, i.e., number of times the new node was selected from the leaves queue.

        Returns
        -------
        int

        """
        return SCIPgetNBacktracks(self._scip)

    def getPlungeDepth(self):
        """
        Gets current plunging depth (successive selections of child/sibling nodes).

        Returns
        -------
        int

        """
        return SCIPgetPlungeDepth(self._scip)

    def getNNodeLPIterations(self):
        """
        Gets number of LP iterations used for solving node relaxations so far.

        Returns
        -------
        int

        """
        return SCIPgetNNodeLPIterations(self._scip)

    def getNStrongbranchLPIterations(self):
        """
        Gets number of LP iterations used for strong branching so far.

        Returns
        -------
        int

        """
        return SCIPgetNStrongbranchLPIterations(self._scip)

    def getPrimalDualIntegral(self):
        """
        Recomputes and returns the primal dual gap stored in the stats

        Returns
        ------
        float
        """
        return SCIPgetPrimalDualIntegral(self._scip)

    def cutoffNode(self, Node node):
        """
        marks node and whole subtree to be cut off from the branch and bound tree.

        Parameters
        ----------
        node : Node
        """
        PY_SCIP_CALL( SCIPcutoffNode(self._scip, node.scip_node) )

    def getCondition(self, exact=False):
        """
        Get the current LP's condition number.

        Parameters
        ----------
        exact : bool, optional
            whether to get an estimate or the exact value (Default value = False)

        Returns
        -------
        float

        """
        cdef SCIP_LPI* lpi
        PY_SCIP_CALL(SCIPgetLPI(self._scip, &lpi))
        cdef SCIP_Real quality = 0
        if exact:
            PY_SCIP_CALL(SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_EXACTCONDITION, &quality))
        else:
            PY_SCIP_CALL(SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_ESTIMCONDITION, &quality))

        return quality

    def enableReoptimization(self, enable=True):
        """
        Include specific heuristics and branching rules for reoptimization.

        Parameters
        ----------
        enable : bool, optional
            True to enable and False to disable

        """
        PY_SCIP_CALL(SCIPenableReoptimization(self._scip, enable))

    def isReoptEnabled(self):
        """
        Returns whether reoptimization is enabled.

        Returns
        -------
        bool

        """
        return SCIPisReoptEnabled(self._scip)

    def lpiGetIterations(self):
        """
        Get the iteration count of the last solved LP.

        Returns
        -------
        int

        """
        cdef SCIP_LPI* lpi
        cdef int iters = 0

        PY_SCIP_CALL(SCIPgetLPI(self._scip, &lpi))
        PY_SCIP_CALL(SCIPlpiGetIterations(lpi, &iters))

        return iters

    # Objective function
    def setMinimize(self):
        """Set the objective sense to minimization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        """Set the objective sense to maximization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    def setObjIntegral(self):
        """Informs SCIP that the objective value is always integral in every feasible solution.

        Notes
        -----
        This function should be used to inform SCIP that the objective function is integral,
        helping to improve the performance. This is useful when using column generation.
        If no column generation (pricing) is used, SCIP automatically detects whether the objective
        function is integral or can be scaled to be integral. However, in any case, the user has to
        make sure that no variable is added during the solving process that destroys this property.
        """
        PY_SCIP_CALL(SCIPsetObjIntegral(self._scip))

    def isObjIntegral(self):
        """
        Returns whether the objective function is integral.

        Returns
        -------
        bool

        """
        return SCIPisObjIntegral(self._scip)

    # Setting parameters
    def setPresolve(self, setting):
        """
        Set presolving parameter settings.


        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
        PY_SCIP_CALL(SCIPsetPresolving(self._scip, setting, True))

    def setProbName(self, name):
        """
        Set problem name.

        Parameters
        ----------
        name : str

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetProbName(self._scip, n))

    def setSeparating(self, setting):
        """
        Set separating parameter settings.

        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
        PY_SCIP_CALL(SCIPsetSeparating(self._scip, setting, True))

    def setHeuristics(self, setting):
        """
        Set heuristics parameter settings.

        Parameters
        ----------
        setting : SCIP_PARAMSETTING
            the parameter settings, e.g. SCIP_PARAMSETTING.OFF

        """
        PY_SCIP_CALL(SCIPsetHeuristics(self._scip, setting, True))

    def setHeurTiming(self, heurname, heurtiming):
        """
        Set the timing of a heuristic

        Parameters
        ----------
        heurname : string, name of the heuristic
        heurtiming : PY_SCIP_HEURTIMING
		   positions in the node solving loop where heuristic should be executed
        """
        cdef SCIP_HEUR* _heur
        n = str_conversion(heurname)
        _heur = SCIPfindHeur(self._scip, n)
        if _heur == NULL:
            raise ValueError("Could not find heuristic <%s>" % heurname)
        SCIPheurSetTimingmask(_heur, heurtiming)

    def getHeurTiming(self, heurname):
        """
        Get the timing of a heuristic

        Parameters
        ----------
        heurname : string, name of the heuristic

        Returns
        -------
        PY_SCIP_HEURTIMING
		   positions in the node solving loop where heuristic should be executed
        """
        cdef SCIP_HEUR* _heur
        n = str_conversion(heurname)
        _heur = SCIPfindHeur(self._scip, n)
        if _heur == NULL:
            raise ValueError("Could not find heuristic <%s>" % heurname)
        return SCIPheurGetTimingmask(_heur)

    def disablePropagation(self, onlyroot=False):
        """
        Disables propagation in SCIP to avoid modifying the original problem during transformation.

        Parameters
        ----------
        onlyroot : bool, optional
            use propagation when root processing is finished (Default value = False)

        """
        self.setIntParam("propagating/maxroundsroot", 0)
        if not onlyroot:
            self.setIntParam("propagating/maxrounds", 0)

    def printProblem(self, ext='.cip', trans=False, genericnames=False):
        """
        Write current model/problem to standard output.

        Parameters
        ----------
        ext   : str, optional
            the extension to be used (Default value = '.cip').
            Should have an extension corresponding to one of the readable file formats,
            described in https://www.scipopt.org/doc/html/group__FILEREADERS.php.
        trans : bool, optional
            indicates whether the transformed problem is written to file (Default value = False)
        genericnames : bool, optional
            indicates whether the problem should be written with generic variable
            and constraint names (Default value = False)
        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        if trans:
            PY_SCIP_CALL(SCIPwriteTransProblem(self._scip, NULL, str_conversion(ext)[1:], genericnames))
        else:
            PY_SCIP_CALL(SCIPwriteOrigProblem(self._scip, NULL, str_conversion(ext)[1:], genericnames))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeProblem(self, filename='model.cip', trans=False, genericnames=False, verbose=True):
        """
        Write current model/problem to a file.

        Parameters
        ----------
        filename : str, optional
            the name of the file to be used (Default value = 'model.cip').
            Should have an extension corresponding to one of the readable file formats,
            described in https://www.scipopt.org/doc/html/group__FILEREADERS.php.
        trans : bool, optional
            indicates whether the transformed problem is written to file (Default value = False)
        genericnames : bool, optional
            indicates whether the problem should be written with generic variable
            and constraint names (Default value = False)
        verbose : bool, optional
            indicates whether a success message should be printed

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        if filename:
            str_absfile = abspath(filename)
            absfile = str_conversion(str_absfile)
            fn, ext = splitext(absfile)
            if len(ext) == 0:
                ext = str_conversion('.cip')
            fn = fn + ext
            ext = ext[1:]

            if trans:
                PY_SCIP_CALL(SCIPwriteTransProblem(self._scip, fn, ext, genericnames))
            else:
                PY_SCIP_CALL(SCIPwriteOrigProblem(self._scip, fn, ext, genericnames))

            if verbose:
                print('wrote problem to file ' + str_absfile)
        else:
            if trans:
                PY_SCIP_CALL(SCIPwriteTransProblem(self._scip, NULL, str_conversion('.cip')[1:], genericnames))
            else:
                PY_SCIP_CALL(SCIPwriteOrigProblem(self._scip, NULL, str_conversion('.cip')[1:], genericnames))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def getTransformedVar(self, Variable var):
        """
        Retrieve the transformed variable.

        Parameters
        ----------
        var : Variable
            original variable to get the transformed of

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* _tvar
        PY_SCIP_CALL(SCIPgetTransformedVar(self._scip, var.scip_var, &_tvar))
        return self._getOrCreateVar(_tvar)

    def addVarLocks(self, Variable var, int nlocksdown, int nlocksup):
        """
        Adds given values to lock numbers of variable for rounding.

        Parameters
        ----------
        var : Variable
            variable to adjust the locks for
        nlocksdown : int
            new number of down locks
        nlocksup : int
            new number of up locks

        """
        PY_SCIP_CALL(SCIPaddVarLocks(self._scip, var.scip_var, nlocksdown, nlocksup))

    def addVarLocksType(self, Variable var, int locktype, int nlocksdown, int nlocksup):
        """
        adds given values to lock numbers of type locktype of variable for rounding

        Parameters
        ----------
        var : Variable
            variable to adjust the locks for
        locktype : SCIP_LOCKTYPE
            type of variable locks
        nlocksdown : int
            modification in number of down locks
        nlocksup : int
            modification in number of up locks

        """
        PY_SCIP_CALL(SCIPaddVarLocksType(self._scip, var.scip_var, locktype, nlocksdown, nlocksup))

    def delVar(self, Variable var):
        """
        Delete a variable.

        Parameters
        ----------
        var : Variable
            the variable which shall be deleted

        Returns
        -------
        deleted : bool
            Whether deleting was successfull

        """
        cdef SCIP_Bool deleted
        del self._modelvars[var.ptr()]
        PY_SCIP_CALL(SCIPdelVar(self._scip, var.scip_var, &deleted))
        # Invalidate pointer after deletion. See issue #604.
        var.scip_var = NULL
        return deleted

    def chgVarType(self, Variable var, vtype):
        """
        Changes the type of a variable.

        Parameters
        ----------
        var : Variable
            variable to change type of
        vtype : str
            new variable type. 'C' or "CONTINUOUS", 'I' or "INTEGER",
            'B' or "BINARY", and 'M' "IMPLINT".

        """
        cdef SCIP_Bool infeasible
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_CONTINUOUS, &infeasible))
        elif vtype in ['B', 'BINARY']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_BINARY, &infeasible))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_INTEGER, &infeasible))
        elif vtype in ['M', 'IMPLINT']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_DEPRECATED_VARTYPE_IMPLINT, &infeasible))
        else:
            raise Warning("unrecognized variable type")
        if infeasible:
            print('could not change variable type of variable %s' % var)

    def markDoNotAggrVar(self, Variable var):
        """
        Marks a variable preventing it from being aggregated.

        Parameters
        ----------
        var : Variable
            variable to mark
        """
        PY_SCIP_CALL(SCIPmarkDoNotAggrVar(self._scip, var.scip_var))

    def markDoNotMultaggrVar(self, Variable var):
        """
        Marks a variable preventing it from being multi-aggregated.

        Parameters
        ----------
        var : Variable
            variable to mark
        """
        PY_SCIP_CALL(SCIPmarkDoNotMultaggrVar(self._scip, var.scip_var))

    def getVars(self, transformed=False):
        """
        Retrieve all variables.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = False)

        Returns
        -------
        list of Variable

        """
        cdef SCIP_VAR** _vars
        cdef int nvars
        cdef int i

        vars = []
        if transformed:
            _vars = SCIPgetVars(self._scip)
            nvars = SCIPgetNVars(self._scip)
        else:
            _vars = SCIPgetOrigVars(self._scip)
            nvars = SCIPgetNOrigVars(self._scip)

        for i in range(nvars):
            vars.append(self._getOrCreateVar(_vars[i]))

        return vars

    def getNVars(self, transformed=True):
        """
        Retrieve number of variables in the problems.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = True)

        Returns
        -------
        int

        """
        if transformed:
            return SCIPgetNVars(self._scip)
        else:
            return SCIPgetNOrigVars(self._scip)

    def getNIntVars(self):
        """
        Gets number of integer active problem variables.

        Returns
        -------
        int

        """
        return SCIPgetNIntVars(self._scip)

    def getNBinVars(self):
        """
        Gets number of binary active problem variables.

        Returns
        -------
        int

        """
        return SCIPgetNBinVars(self._scip)

    def getNImplVars(self):
        """
        Gets number of implicit integer active problem variables.

        Returns
        -------
        int

        """
        return SCIPgetNImplVars(self._scip)

    def getNContVars(self):
        """
        Gets number of continuous active problem variables.

        Returns
        -------
        int

        """
        return SCIPgetNContVars(self._scip)

    def getVarPseudocostScore(self, Variable var, solVal):
        """
        gets the variable's pseudo cost score value for the given LP solution value

        Parameters
        ----------
        variable : Variable
            problem variable
        solVal : float
            difference of variable's new LP value - old LP value

        Returns
        -------
        float
        """
        return SCIPgetVarPseudocostScore(self._scip, var.scip_var, solVal)

    def getVarPseudocost(self, Variable var, branchdir):
        """
        gets the variable's pseudo cost value for the given direction

        Parameters
        ----------
        variable : Variable
            problem variable
        branchdir : PY_SCIP_BRANCHDIR
            branching direction (downwards, or upwards)

        Returns
        -------
        float
        """
        return SCIPgetVarPseudocost(self._scip, var.scip_var, branchdir)

    def relax(self):
        """Relaxes the integrality restrictions of the model."""
        if self.getStage() != SCIP_STAGE_PROBLEM:
            raise Warning("method can only be called in stage PROBLEM")

        for var in self.getVars():
            self.chgVarType(var, "C")

    # Node methods
    def getBestChild(self):
        """
        Gets the best child of the focus node w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetBestChild(self._scip))
    
    def getChildren(self):
        """
        Gets the children of the focus node.

        Returns
        -------
        list of Nodes

        """
        cdef SCIP_NODE** _children
        cdef int n_children
        cdef int i

        PY_SCIP_CALL(SCIPgetChildren(self._scip, &_children, &n_children))

        return [Node.create(_children[i]) for i in range(n_children)]

    def getSiblings(self):
        """
        Gets the siblings of the focus node.

        Returns
        -------
        list of Nodes

        """
        cdef SCIP_NODE** _siblings
        cdef int n_siblings
        cdef int i

        PY_SCIP_CALL(SCIPgetSiblings(self._scip, &_siblings, &n_siblings))

        return [Node.create(_siblings[i]) for i in range(n_siblings)]
    
    def getLeaves(self):
        """
        Gets the leaves of the tree along with number of leaves.

        Returns
        -------
        list of Nodes

        """
        cdef SCIP_NODE** _leaves
        cdef int n_leaves
        cdef int i

        PY_SCIP_CALL(SCIPgetLeaves(self._scip, &_leaves, &n_leaves))

        return [Node.create(_leaves[i]) for i in range(n_leaves)]
    
    def getBestSibling(self):
        """
        Gets the best sibling of the focus node w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetBestSibling(self._scip))

    def getPrioChild(self):
        """
        Gets the best child of the focus node w.r.t. the node selection priority
        assigned by the branching rule.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetPrioChild(self._scip))

    def getPrioSibling(self):
        """Gets the best sibling of the focus node w.r.t.
        the node selection priority assigned by the branching rule.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetPrioSibling(self._scip))

    def getBestLeaf(self):
        """Gets the best leaf from the node queue w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetBestLeaf(self._scip))

    def getBestNode(self):
        """Gets the best node from the tree (child, sibling, or leaf) w.r.t. the node selection strategy.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetBestNode(self._scip))

    def getBestboundNode(self):
        """Gets the node with smallest lower bound from the tree (child, sibling, or leaf).

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetBestboundNode(self._scip))

    def getOpenNodes(self):
        """
        Access to all data of open nodes (leaves, children, and siblings).

        Returns
        -------
        leaves : list of Node
            list of all open leaf nodes
        children : list of Node
            list of all open children nodes
        siblings : list of Node
            list of all open sibling nodes

        """
        cdef SCIP_NODE** _leaves
        cdef SCIP_NODE** _children
        cdef SCIP_NODE** _siblings
        cdef int _nleaves
        cdef int _nchildren
        cdef int _nsiblings
        cdef int i

        PY_SCIP_CALL(SCIPgetOpenNodesData(self._scip, &_leaves, &_children, &_siblings, &_nleaves, &_nchildren, &_nsiblings))

        leaves   = [Node.create(_leaves[i]) for i in range(_nleaves)]
        children = [Node.create(_children[i]) for i in range(_nchildren)]
        siblings = [Node.create(_siblings[i]) for i in range(_nsiblings)]

        return leaves, children, siblings

    def repropagateNode(self, Node node):
        """Marks the given node to be propagated again the next time a node of its subtree is processed."""
        PY_SCIP_CALL(SCIPrepropagateNode(self._scip, node.scip_node))

    def constructLP(self):
        """
        Makes sure that the LP of the current node is loaded and
        may be accessed through the LP information methods.


        Returns
        -------
        cutoff : bool
            Can the node be cutoff?

        """
        cdef SCIP_Bool cutoff
        PY_SCIP_CALL(SCIPconstructLP(self._scip, &cutoff))
        return cutoff

    def getNLPRows(self):
        """
        Retrieve the number of rows currently in the LP.

        Returns
        -------
        int

        """
        return SCIPgetNLPRows(self._scip)

    def getNLPCols(self):
        """
        Retrieve the number of columns currently in the LP.

        Returns
        -------
        int

        """
        return SCIPgetNLPCols(self._scip)

    def getLPBasisInd(self):
        """
        Gets all indices of basic columns and rows:
        index i >= 0 corresponds to column i, index i < 0 to row -i-1

        Returns
        -------
        list of int

        """
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef int* inds = <int *> malloc(nrows * sizeof(int))
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBasisInd(self._scip, inds))
        result = [inds[i] for i in range(nrows)]
        free(inds)

        return result

    def isLPSolBasic(self):
        """
        Returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis.

        Returns
        -------
        bool

        """
        return SCIPisLPSolBasic(self._scip)

    def allColsInLP(self):
        """
        Checks if all columns, i.e. every variable with non-empty column is present in the LP.
        This is not True when performing pricing for instance.

        Returns
        -------
        bool

        """

        return SCIPallColsInLP(self._scip)

    def getNCuts(self):
        """
        Retrieve total number of cuts in storage.

        Returns
        -------
        int

        """
        return SCIPgetNCuts(self._scip)

    def getNCutsApplied(self):
        """
        Retrieve number of currently applied cuts.

        Returns
        -------
        int

        """
        return SCIPgetNCutsApplied(self._scip)

    def getNSepaRounds(self):
        """
        Retrieve the number of separation rounds that have been performed
        at the current node.

        Returns
        -------
        int

        """
        return SCIPgetNSepaRounds(self._scip)

    def separateSol(self, Solution sol = None, pretendroot = False, allowlocal = True, onlydelayed = False):
        """
        Separates the given primal solution or the current LP solution by calling
        the separators and constraint handlers' separation methods;
        the generated cuts are stored in the separation storage and can be accessed
        with the methods SCIPgetCuts() and SCIPgetNCuts();
        after evaluating the cuts, you have to call SCIPclearCuts() in order to remove the cuts from the
        separation storage; it is possible to call SCIPseparateSol() multiple times with
        different solutions and evaluate the found cuts afterwards.

        Parameters
        ----------
        sol : Solution or None, optional
            solution to separate, None to use current lp solution (Default value = None)
        pretendroot : bool, optional
            should the cut separators be called as if we are at the root node? (Default value = "False")
        allowlocal : bool, optional
            should the separator be asked to separate local cuts (Default value = True)
        onlydelayed : bool, optional
            should only separators be called that were delayed in the previous round? (Default value = False)

        Returns
        -------
        delayed : bool
            whether a separator was delayed
        cutoff : bool
            whether the node can be cut off

        """
        cdef SCIP_Bool delayed
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL( SCIPseparateSol(self._scip, NULL if sol is None else sol.sol, pretendroot, allowlocal, onlydelayed, &delayed, &cutoff) )
        return delayed, cutoff

    def addConss(self, conss, name='', initial=True, separate=True,
                 enforce=True, check=True, propagate=True, local=False,
                 modifiable=False, dynamic=False, removable=False,
                 stickingatnode=False):
        """Adds multiple constraints.

        Each of the constraints is added to the model using Model.addCons().

        For all parameters, except `conss`, this method behaves differently depending on the
        type of the passed argument:
        1. If the value is iterable, it must be of the same length as `conss`. For each
        constraint, Model.addCons() will be called with the value at the corresponding index.
        2. Else, the (default) value will be applied to all of the constraints.

        Parameters
        ----------
        conss : iterable of ExprCons
            An iterable of constraint objects. Any iterable will be converted into a list before further processing.
        name : str or iterable of str, optional
            the name of the constraint, generic name if empty (Default value = '')
        initial : bool or iterable of bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool or iterable of bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool or iterable of bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool or iterable of bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool or iterable of bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool or iterable of bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool or iterable of bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool or iterable of bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool or iterable of bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool or iterable of bool, optional
            should the constraints always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        list of Constraint
            The created and added Constraint objects.

        """
        cdef int n_conss
        cdef int i

        if isinstance(conss, MatrixExprCons):
            conss = conss.flatten()

        def ensure_iterable(elem, length):
            if isinstance(elem, Iterable):
                return elem
            else:
                return list(repeat(elem, length))

        assert isinstance(conss, Iterable), "Given constraint list is not iterable."

        conss = list(conss)
        n_conss = len(conss)

        if isinstance(name, str):
            if name == "":
                name = ["" for i in range(n_conss)]
            else:
                name = ["%s_%s" % (name, i) for i in range(n_conss)]

        initial = ensure_iterable(initial, n_conss)
        separate = ensure_iterable(separate, n_conss)
        enforce = ensure_iterable(enforce, n_conss)
        check = ensure_iterable(check, n_conss)
        propagate = ensure_iterable(propagate, n_conss)
        local = ensure_iterable(local, n_conss)
        modifiable = ensure_iterable(modifiable, n_conss)
        dynamic = ensure_iterable(dynamic, n_conss)
        removable = ensure_iterable(removable, n_conss)
        stickingatnode = ensure_iterable(stickingatnode, n_conss)

        constraints = []
        for i, cons in enumerate(conss):
            constraints.append(
                self.addCons(cons, name[i], initial[i], separate[i], enforce[i],
                             check[i], propagate[i], local[i], modifiable[i],
                             dynamic[i], removable[i], stickingatnode[i])
            )

        return constraints

    def addMatrixCons(self,
                      cons: Union[ExprCons, MatrixExprCons],
                      name: Union[str, np.ndarray] ='',
                      initial: Union[bool, np.ndarray] = True,
                      separate: Union[bool, np.ndarray] = True,
                      enforce: Union[bool, np.ndarray] = True,
                      check: Union[bool, np.ndarray] = True,
                      propagate: Union[bool, np.ndarray] = True,
                      local: Union[bool, np.ndarray] = False,
                      modifiable: Union[bool, np.ndarray] = False,
                      dynamic: Union[bool, np.ndarray] = False,
                      removable: Union[bool, np.ndarray] = False,
                      stickingatnode: Union[bool, np.ndarray] = False):
        """
        Add a linear or nonlinear matrix constraint.

        Parameters
        ----------
        cons : ExprCons or MatrixExprCons
            The matrix expression constraint or expression constraint, that are not yet an actual constraint
        name : str or np.ndarray, optional
            the name of the matrix constraint, generic name if empty (Default value = "")
        initial : bool or np.ndarray, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool or np.ndarray, optional
            should the matrix constraint be separated during LP processing? (Default value = True)
        enforce : bool or np.ndarray, optional
            should the matrix constraint be enforced during node processing? (Default value = True)
        check : bool or np.ndarray, optional
            should the matrix constraint be checked for feasibility? (Default value = True)
        propagate : bool or np.ndarray, optional
            should the matrix constraint be propagated during node processing? (Default value = True)
        local : bool or np.ndarray, optional
            is the matrix constraint only valid locally? (Default value = False)
        modifiable : bool or np.ndarray, optional
            is the matrix constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool or np.ndarray, optional
            is the matrix constraint subject to aging? (Default value = False)
        removable : bool or np.ndarray, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool or np.ndarray, optional
            should the matrix constraints always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint or MatrixConstraint
            The created and added Constraint or MatrixConstraint.
        """
        if not isinstance(cons, (ExprCons, MatrixExprCons)):
            raise TypeError("given constraint is not MatrixExprCons nor ExprCons but %s" % cons.__class__.__name__)

        if isinstance(cons, ExprCons):
            return self.addCons(cons, name=name, initial=initial, separate=separate,
                        enforce=enforce, check=check, propagate=propagate,
                        local=local, modifiable=modifiable, dynamic=dynamic,
                        removable=removable, stickingatnode=stickingatnode)

        shape = cons.shape

        if isinstance(name, np.ndarray):
            assert name.shape == shape
        if isinstance(initial, np.ndarray):
            assert initial.shape == shape
        if isinstance(separate, np.ndarray):
            assert separate.shape == shape
        if isinstance(enforce, np.ndarray):
            assert enforce.shape == shape
        if isinstance(check, np.ndarray):
            assert check.shape == shape
        if isinstance(propagate, np.ndarray):
            assert propagate.shape == shape
        if isinstance(local, np.ndarray):
            assert local.shape == shape
        if isinstance(modifiable, np.ndarray):
            assert modifiable.shape == shape
        if isinstance(dynamic, np.ndarray):
            assert dynamic.shape == shape
        if isinstance(removable, np.ndarray):
            assert removable.shape == shape
        if isinstance(stickingatnode, np.ndarray):
            assert stickingatnode.shape == shape

        matrix_cons = np.empty(shape, dtype=object)

        if isinstance(shape, int):
            ndim = 1
        else:
            ndim = len(shape)
        # cdef np.ndarray[object, ndim=ndim] matrix_variable = np.empty(shape, dtype=object)
        matrix_variable = np.empty(shape, dtype=object)

        if isinstance(name, str):
            matrix_names = np.full(shape, name, dtype=object)
            if name != "":
                for idx in np.ndindex(matrix_variable.shape):
                    matrix_names[idx] = f"{name}_{'_'.join(map(str, idx))}"
        else:
            matrix_names = name

        if not isinstance(initial, np.ndarray):
            matrix_initial = np.full(shape, initial, dtype=bool)
        else:
            matrix_initial = initial

        if not isinstance(separate, np.ndarray):
            matrix_separate = np.full(shape, separate, dtype=bool)
        else:
            matrix_separate = separate

        if not isinstance(enforce, np.ndarray):
            matrix_enforce = np.full(shape, enforce, dtype=bool)
        else:
            matrix_enforce = enforce

        if not isinstance(check, np.ndarray):
            matrix_check = np.full(shape, check, dtype=bool)
        else:
            matrix_check = check

        if not isinstance(propagate, np.ndarray):
            matrix_propagate = np.full(shape, propagate, dtype=bool)
        else:
            matrix_propagate = propagate

        if not isinstance(local, np.ndarray):
            matrix_local = np.full(shape, local, dtype=bool)
        else:
            matrix_local = local

        if not isinstance(modifiable, np.ndarray):
            matrix_modifiable = np.full(shape, modifiable, dtype=bool)
        else:
            matrix_modifiable = modifiable

        if not isinstance(dynamic, np.ndarray):
            matrix_dynamic = np.full(shape, dynamic, dtype=bool)
        else:
            matrix_dynamic = dynamic

        if not isinstance(removable, np.ndarray):
            matrix_removable = np.full(shape, removable, dtype=bool)
        else:
            matrix_removable = removable

        if not isinstance(stickingatnode, np.ndarray):
            matrix_stickingatnode = np.full(shape, stickingatnode, dtype=bool)
        else:
            matrix_stickingatnode = stickingatnode

        for idx in np.ndindex(cons.shape):
            matrix_cons[idx] = self.addCons(cons[idx], name=matrix_names[idx], initial=matrix_initial[idx],
                                            separate=matrix_separate[idx], check=matrix_check[idx],
                                            propagate=matrix_propagate[idx], local=matrix_local[idx],
                                            modifiable=matrix_modifiable[idx], dynamic=matrix_dynamic[idx],
                                            removable=matrix_removable[idx], stickingatnode=matrix_stickingatnode[idx])

        return matrix_cons.view(MatrixConstraint)

    def getConsNVars(self, Constraint constraint):
        """
        Gets number of variables in a constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to get the number of variables from.

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If the associated constraint handler does not have this functionality

        """
        cdef int nvars
        cdef SCIP_Bool success

        PY_SCIP_CALL(SCIPgetConsNVars(self._scip, constraint.scip_cons, &nvars, &success))

        if not success:
            conshdlr = SCIPconsGetHdlr(constraint.scip_cons)
            conshdrlname = SCIPconshdlrGetName(conshdlr)
            raise TypeError("The constraint handler %s does not have this functionality." % conshdrlname)

        return nvars

    cdef SCIP_CONS* _getTransformedCons(self, Constraint cons, str expected_cons_type):
        """
        Validates constraint type and retrieves its transformed representation.

        Parameters
        ----------
        cons : Constraint
            Constraint to get the transformed representation for.
        expected_cons_type : str
            Expected constraint type name (e.g. 'linear', 'logicor', 'knapsack').

        Returns
        -------
        SCIP_CONS*
            Pointer to the transformed SCIP constraint, or NULL if transformed constraint does not exist.

        Raises
        ------
        Warning
            If the constraint handler type does not match expected_cons_type.

        """
        cdef SCIP_CONS* transcons

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == expected_cons_type:
            raise Warning(f"dual solution values not available for constraints of type {constype}")

        if cons.isOriginal():
            PY_SCIP_CALL(SCIPgetTransformedCons(self._scip, cons.scip_cons, &transcons))
            return transcons

        return cons.scip_cons

    def printCons(self, Constraint constraint):
        """
        Print the constraint

        Parameters
        ----------
        constraint : Constraint

        """
        return PY_SCIP_CALL(SCIPprintCons(self._scip, constraint.scip_cons, NULL))

    def setInitial(self, Constraint cons, newInit):
        """
        Set "initial" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newInit : bool

        """
        PY_SCIP_CALL(SCIPsetConsInitial(self._scip, cons.scip_cons, newInit))
    
    def setModifiable(self, Constraint cons, newMod):
        """
        Set "modifiable" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newMod : bool

        """
        PY_SCIP_CALL(SCIPsetConsModifiable(self._scip, cons.scip_cons, newMod))

    def setRemovable(self, Constraint cons, newRem):
        """
        Set "removable" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newRem : bool

        """
        PY_SCIP_CALL(SCIPsetConsRemovable(self._scip, cons.scip_cons, newRem))

    def setEnforced(self, Constraint cons, newEnf):
        """
        Set "enforced" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newEnf : bool

        """
        PY_SCIP_CALL(SCIPsetConsEnforced(self._scip, cons.scip_cons, newEnf))

    def setCheck(self, Constraint cons, newCheck):
        """
        Set "check" flag of a constraint.

        Parameters
        ----------
        cons : Constraint
        newCheck : bool

        """
        PY_SCIP_CALL(SCIPsetConsChecked(self._scip, cons.scip_cons, newCheck))

    def getTransformedCons(self, Constraint cons):
        """
        Retrieve transformed constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        Constraint

        """
        cdef SCIP_CONS* transcons
        PY_SCIP_CALL(SCIPgetTransformedCons(self._scip, cons.scip_cons, &transcons))
        return self._getOrCreateCons(transcons)

    def enableDebugSol(self):
        """
        Enables the debug solution mechanism, which allows tracing back the invalidation of
        a debug solution during the solution process of SCIP. It must be explicitly
        enabled for the SCIP data structure.
        """
        if not PYSCIPOPT_WITH_DEBUG_SOLUTION:
            raise RuntimeError("SCIP must be built with `DEBUGSOL=true` to enable the debug solution mechanism.")
        SCIPenableDebugSol(self._scip)
    
    def disableDebugSol(self):
        """
        Disables the debug solution mechanism.
        """
        if not PYSCIPOPT_WITH_DEBUG_SOLUTION:
            raise RuntimeError("SCIP must be built with `DEBUGSOL=true` to disable the debug solution mechanism.")
        SCIPdisableDebugSol(self._scip)

    def getConss(self, transformed=True):
        """
        Retrieve all constraints.

        Parameters
        ----------
        transformed : bool, optional
            get transformed variables instead of original (Default value = True)

        Returns
        -------
        list of Constraint

        """
        cdef SCIP_CONS** conss
        cdef int nconss
        cdef int i

        if transformed:
            conss = SCIPgetConss(self._scip)
            nconss = SCIPgetNConss(self._scip)
        else:
            conss = SCIPgetOrigConss(self._scip)
            nconss = SCIPgetNOrigConss(self._scip)

        return [self._getOrCreateCons(conss[i]) for i in range(nconss)]

    def getNConss(self, transformed=True):
        """
        Retrieve number of all constraints.

        Parameters
        ----------
        transformed : bool, optional
            get number of transformed variables instead of original (Default value = True)

        Returns
        -------
        int

        """
        if transformed:
            return SCIPgetNConss(self._scip)
        else:
            return SCIPgetNOrigConss(self._scip)

    def delCons(self, Constraint cons):
        """
        Delete constraint from the model

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
        del self._modelconss[cons.ptr()]
        PY_SCIP_CALL(SCIPdelCons(self._scip, cons.scip_cons))
        # Remove from tracking and invalidate pointer. See issue #604.
        cons.scip_cons = NULL

    def delConsLocal(self, Constraint cons):
        """
        Delete constraint from the current node and its children.

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
        PY_SCIP_CALL(SCIPdelConsLocal(self._scip, cons.scip_cons))
    
    def optimize(self):
        """Optimize the problem."""
        PY_SCIP_CALL(SCIPsolve(self._scip))
        self._bestSol = Solution.create(self._scip, SCIPgetBestSol(self._scip))

    def optimizeNogil(self):
        """Optimize the problem without GIL."""
        cdef SCIP_RETCODE rc;
        with nogil:
            rc = SCIPsolve(self._scip)
        PY_SCIP_CALL(rc)
        self._bestSol = Solution.create(self._scip, SCIPgetBestSol(self._scip))

    def solveConcurrent(self):
        """Transforms, presolves, and solves problem using additional solvers which emphasize on
        finding solutions.
        WARNING: This feature is still experimental and prone to some errors."""
        if SCIPtpiGetNumThreads() == 1:
            warnings.warn("SCIP was compiled without task processing interface. Parallel solve not possible - using optimize() instead of solveConcurrent()")
            self.optimize()
        else:
            PY_SCIP_CALL(SCIPsolveConcurrent(self._scip))
            self._bestSol = Solution.create(self._scip, SCIPgetBestSol(self._scip))

    def presolve(self):
        """Presolve the problem."""
        if self.getStage() not in [SCIP_STAGE_PROBLEM, SCIP_STAGE_TRANSFORMED,\
                                SCIP_STAGE_PRESOLVING, SCIP_STAGE_PRESOLVED, \
                                SCIP_STAGE_SOLVED]:
            raise Warning("method cannot be called in stage %i." % self.getStage())

        PY_SCIP_CALL(SCIPpresolve(self._scip))
        self._bestSol = Solution.create(self._scip, SCIPgetBestSol(self._scip))

    # Benders' decomposition methods
    def initBendersDefault(self, subproblems):
        """
        Initialises the default Benders' decomposition with a dictionary of subproblems.

        Parameters
        ----------
        subproblems : Model or dict of object to Model
            a single Model instance or dictionary of Model instances

        """
        cdef SCIP** subprobs
        cdef SCIP_BENDERS* benders
        cdef int nsubproblems
        cdef int i

        # checking whether subproblems is a dictionary
        if isinstance(subproblems, dict):
            isdict = True
            nsubproblems = len(subproblems)
        else:
            isdict = False
            nsubproblems = 1

        # create array of SCIP instances for the subproblems
        subprobs = <SCIP**> malloc(nsubproblems * sizeof(SCIP*))

        # if subproblems is a dictionary, then the dictionary is turned into a c array
        if isdict:
            for i, subprob in enumerate(subproblems.values()):
                subprobs[i] = (<BaseModel>subprob)._scip
        else:
            subprobs[0] = (<BaseModel>subproblems)._scip

        # creating the default Benders' decomposition
        PY_SCIP_CALL(SCIPcreateBendersDefault(self._scip, subprobs, nsubproblems))
        benders = SCIPfindBenders(self._scip, "default")

        # Free the temporary array (SCIPcreateBendersDefault copies the pointers internally)
        free(subprobs)

        # Store references to subproblem Models and transfer ownership to master.
        # The master will free the Benders subproblems in its __dealloc__ method.
        # Set _freescip = False on the subproblem Model(s) to prevent double-free
        # if they are deallocated before the master.
        if isdict:
            self._benders_subproblems = list(subproblems.values())
            for subprob in self._benders_subproblems:
                (<BaseModel>subprob)._freescip = False
        else:
            self._benders_subproblems = [subproblems]
            (<BaseModel>subproblems)._freescip = False

        # activating the Benders' decomposition constraint handlers
        self.setBoolParam("constraints/benderslp/active", True)
        self.setBoolParam("constraints/benders/active", True)
        #self.setIntParam("limits/maxorigsol", 0)

    def computeBestSolSubproblems(self):
        """Solves the subproblems with the best solution to the master problem.
        Afterwards, the best solution from each subproblem can be queried to get
        the solution to the original problem.
        If the user wants to resolve the subproblems, they must free them by
        calling freeBendersSubproblems()
        """
        cdef SCIP_BENDERS** benders = SCIPgetBenders(self._scip)
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool solvecip = True
        cdef int nbenders = SCIPgetNActiveBenders(self._scip)
        cdef int nsubproblems
        cdef int i
        cdef int j

        # solving all subproblems from all Benders' decompositions
        for i in range(nbenders):
            nsubproblems = SCIPbendersGetNSubproblems(benders[i])
            for j in range(nsubproblems):
                PY_SCIP_CALL(SCIPsetupBendersSubproblem(self._scip,
                    benders[i], self._bestSol.sol, j, SCIP_BENDERSENFOTYPE_CHECK))
                PY_SCIP_CALL(SCIPsolveBendersSubproblem(self._scip,
                    benders[i], self._bestSol.sol, j, &infeasible, solvecip, NULL))

    def freeBendersSubproblems(self):
        """Deprecated: This method is no longer needed.

        Benders subproblems are now automatically managed and freed by the master
        Model when it is deallocated. Calling this method explicitly has no effect.

        .. deprecated:: 5.7.0
            This method is deprecated and will be removed in a future version.
            Subproblem memory management is now handled automatically.
        """
        warnings.warn(
            "freeBendersSubproblems() is deprecated and no longer needed. "
            "Benders subproblems are automatically freed when the master Model "
            "is deallocated.",
            DeprecationWarning,
            stacklevel=2
        )

    cdef SCIP_BENDERS* _getBenders(self, object benders):
        """
        Retrieves the SCIP_BENDERS pointer for the given Benders object or the default decomposition.

        Parameters
        ----------
        benders : Benders or None
            The Benders' decomposition object, or None to use the default Benders' decomposition.

        Returns
        -------
        SCIP_BENDERS*
            Pointer to the underlying SCIP_BENDERS structure, or NULL if not found.

        """
        if benders is None:
            return SCIPfindBenders(self._scip, "default")
        return (<Benders>benders)._benders

    def updateBendersLowerbounds(self, lowerbounds, Benders benders=None):
        """
        Updates the subproblem lower bounds for benders using
        the lowerbounds dict. If benders is None, then the default
        Benders' decomposition is updated.

        Parameters
        ----------
        lowerbounds : dict of int to float
        benders : Benders or None, optional

        """
        cdef SCIP_BENDERS* _benders = self._getBenders(benders)

        assert type(lowerbounds) is dict

        for d in lowerbounds.keys():
            SCIPbendersUpdateSubproblemLowerbound(_benders, d, lowerbounds[d])

    def activateBenders(self, Benders benders, int nsubproblems):
        """
        Activates the Benders' decomposition plugin with the input name.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition to which the subproblem belongs to
        nsubproblems : int
            the number of subproblems in the Benders' decomposition

        """
        PY_SCIP_CALL(SCIPactivateBenders(self._scip, benders._benders, nsubproblems))

    def addBendersSubproblem(self, Benders benders, subproblem):
        """
        Adds a subproblem to the Benders' decomposition given by the input
        name.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition to which the subproblem belongs to
        subproblem : Model
            the subproblem to add to the decomposition

        """
        PY_SCIP_CALL(SCIPaddBendersSubproblem(self._scip, benders._benders, (<BaseModel>subproblem)._scip))

    def setBendersSubproblemIsConvex(self, Benders benders, probnumber, isconvex = True):
        """
        Sets a flag indicating whether the subproblem is convex.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition which contains the subproblem
        probnumber : int
            the problem number of the subproblem that the convexity will be set for
        isconvex : bool, optional
            flag to indicate whether the subproblem is convex (default=True)

        """
        SCIPbendersSetSubproblemIsConvex(benders._benders, probnumber, isconvex)

    def setupBendersSubproblem(self, probnumber, Benders benders = None, Solution solution = None, checktype = PY_SCIP_BENDERSENFOTYPE.LP):
        """
        Sets up the Benders' subproblem given the master problem solution.

        Parameters
        ----------
        probnumber : int
            the index of the problem that is to be set up
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem belongs to
        solution : Solution or None, optional
            the master problem solution that is used for the set up, if None, then the LP solution is used
        checktype : PY_SCIP_BENDERSENFOTYPE
            the type of solution check that prompted the solving of the Benders' subproblems, either
            PY_SCIP_BENDERSENFOTYPE: LP, RELAX, PSEUDO or CHECK. Default is LP.

        """
        cdef SCIP_BENDERS* scip_benders = self._getBenders(benders)
        cdef SCIP_SOL* scip_sol

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        retcode = SCIPsetupBendersSubproblem(self._scip, scip_benders, scip_sol, probnumber, checktype)

        PY_SCIP_CALL(retcode)

    def solveBendersSubproblem(self, probnumber, solvecip, Benders benders = None, Solution solution = None):
        """
        Solves the Benders' decomposition subproblem. The convex relaxation will be solved unless
        the parameter solvecip is set to True.

        Parameters
        ----------
        probnumber : int
            the index of the problem that is to be set up
        solvecip : bool
            whether the CIP of the subproblem should be solved. If False, then only the convex relaxation is solved.
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem belongs to
        solution : Solution or None, optional
            the master problem solution that is used for the set up, if None, then the LP solution is used

        Returns
        -------
        infeasible : bool
            returns whether the current subproblem is infeasible
        objective : float or None
            the objective function value of the subproblem, can be None

        """

        cdef SCIP_BENDERS* scip_benders = self._getBenders(benders)
        cdef SCIP_SOL* scip_sol
        cdef SCIP_Real objective
        cdef SCIP_Bool infeasible

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        PY_SCIP_CALL(SCIPsolveBendersSubproblem(self._scip, scip_benders, scip_sol,
            probnumber, &infeasible, solvecip, &objective))

        return infeasible, objective

    def getBendersSubproblem(self, probnumber, Benders benders = None):
        """
        Returns a Model object that wraps around the SCIP instance of the subproblem.
        NOTE: This Model object is just a place holder and SCIP instance will not be
        freed when the object is destroyed.

        Parameters
        ----------
        probnumber : int
            the problem number for subproblem that is required
        benders : Benders or None, optional
            the Benders' decomposition object that the subproblem belongs to (Default = None)

        Returns
        -------
        Model

        """
        cdef SCIP_BENDERS* scip_benders = self._getBenders(benders)
        cdef SCIP* scip_subprob

        scip_subprob = SCIPbendersSubproblem(scip_benders, probnumber)

        return BaseModel.create(scip_subprob)

    def getBendersVar(self, Variable var, Benders benders = None, probnumber = -1):
        """
        Returns the variable for the subproblem or master problem
        depending on the input probnumber.

        Parameters
        ----------
        var : Variable
            the source variable for which the target variable is requested
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem variables belong to
        probnumber : int, optional
            the problem number for which the target variable belongs, -1 for master problem

        Returns
        -------
        Variable or None

        """
        cdef SCIP_BENDERS* _benders = self._getBenders(benders)
        cdef SCIP_VAR* _mappedvar

        if probnumber == -1:
            PY_SCIP_CALL(SCIPgetBendersMasterVar(self._scip, _benders, var.scip_var, &_mappedvar))
        else:
            PY_SCIP_CALL(SCIPgetBendersSubproblemVar(self._scip, _benders, var.scip_var, &_mappedvar, probnumber))

        if _mappedvar == NULL:
            return None
        return self._getOrCreateVar(_mappedvar)

    def getBendersAuxiliaryVar(self, probnumber, Benders benders = None):
        """
        Returns the auxiliary variable that is associated with the input problem number

        Parameters
        ----------
        probnumber : int
            the problem number for which the target variable belongs, -1 for master problem
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem variables belong to

        Returns
        -------
        Variable

        """
        cdef SCIP_BENDERS* _benders = self._getBenders(benders)
        cdef SCIP_VAR* _auxvar

        _auxvar = SCIPbendersGetAuxiliaryVar(_benders, probnumber)
        return self._getOrCreateVar(_auxvar)

    def checkBendersSubproblemOptimality(self, Solution solution, probnumber, Benders benders = None):
        """
        Returns whether the subproblem is optimal w.r.t the master problem auxiliary variables.

        Parameters
        ----------
        solution : Solution
            the master problem solution that is being checked for optimamlity
        probnumber : int
            the problem number for which optimality is being checked
        benders : Benders or None, optional
            the Benders' decomposition to which the subproblem belongs to

        Returns
        -------
        optimal : bool
            flag to indicate whether the current subproblem is optimal for the master

        """
        cdef SCIP_BENDERS* _benders = self._getBenders(benders)
        cdef SCIP_SOL* scip_sol
        cdef SCIP_Bool optimal

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        PY_SCIP_CALL( SCIPcheckBendersSubproblemOptimality(self._scip, _benders,
            scip_sol, probnumber, &optimal) )

        return optimal

    def includeBendersDefaultCuts(self, Benders benders):
        """
        Includes the default Benders' decomposition cuts to the custom Benders' decomposition plugin.

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition that the default cuts will be applied to

        """
        PY_SCIP_CALL( SCIPincludeBendersDefaultCuts(self._scip, benders._benders) )

    def includeEventhdlr(self, Eventhdlr eventhdlr, name, desc):
        """
        Include an event handler.

        Parameters
        ----------
        eventhdlr : Eventhdlr
            event handler
        name : str
            name of event handler
        desc : str
            description of event handler

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeEventhdlr(self._scip, n, d,
                                          PyEventCopy,
                                          PyEventFree,
                                          PyEventInit,
                                          PyEventExit,
                                          PyEventInitsol,
                                          PyEventExitsol,
                                          PyEventDelete,
                                          PyEventExec,
                                          <SCIP_EVENTHDLRDATA*>eventhdlr))
        eventhdlr.model = self
        eventhdlr.name = name
        self._plugins.append(eventhdlr)

    def includePricer(self, Pricer pricer, name, desc, priority=1, delay=True):
        """
        Include a pricer.

        Parameters
        ----------
        pricer : Pricer
            pricer
        name : str
            name of pricer
        desc : str
            description of pricer
        priority : int, optional
            priority of pricer (Default value = 1)
        delay : bool, optional
            should the pricer be delayed until no other pricers or already existing problem variables
            with negative reduced costs are found? (Default value = True)

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
        self._plugins.append(pricer)
        pricer.scip_pricer = scip_pricer

    def includeConshdlr(self, Conshdlr conshdlr, name, desc, sepapriority=0,
                        enfopriority=0, chckpriority=0, sepafreq=-1, propfreq=-1,
                        eagerfreq=100, maxprerounds=-1, delaysepa=False,
                        delayprop=False, needscons=True,
                        proptiming=PY_SCIP_PROPTIMING.BEFORELP,
                        presoltiming=PY_SCIP_PRESOLTIMING.MEDIUM):
        """
        Include a constraint handler.

        Parameters
        ----------
        conshdlr : Conshdlr
            constraint handler
        name : str
            name of constraint handler
        desc : str
            description of constraint handler
        sepapriority : int, optional
            priority for separation (Default value = 0)
        enfopriority : int, optional
            priority for constraint enforcing (Default value = 0)
        chckpriority : int, optional
            priority for checking feasibility (Default value = 0)
        sepafreq : int, optional
            frequency for separating cuts; 0 = only at root node (Default value = -1)
        propfreq : int, optional
            frequency for propagating domains; 0 = only preprocessing propagation (Default value = -1)
        eagerfreq : int, optional
            frequency for using all instead of only the useful constraints in separation,
             propagation and enforcement; -1 = no eager evaluations, 0 = first only
             (Default value = 100)
        maxprerounds : int, optional
            maximal number of presolving rounds the constraint handler participates in (Default value = -1)
        delaysepa : bool, optional
            should separation method be delayed, if other separators found cuts? (Default value = False)
        delayprop : bool, optional
            should propagation method be delayed, if other propagators found reductions? (Default value = False)
        needscons : bool, optional
            should the constraint handler be skipped, if no constraints are available? (Default value = True)
        proptiming : PY_SCIP_PROPTIMING
            positions in the node solving loop where propagation method of constraint handlers
             should be executed (Default value = SCIP_PROPTIMING.BEFORELP)
        presoltiming : PY_SCIP_PRESOLTIMING
            timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING.MEDIUM)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeConshdlr(self._scip, n, d, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                                              maxprerounds, delaysepa, delayprop, needscons, proptiming, presoltiming,
                                              PyConshdlrCopy, PyConsFree, PyConsInit, PyConsExit, PyConsInitpre, PyConsExitpre,
                                              PyConsInitsol, PyConsExitsol, PyConsDelete, PyConsTrans, PyConsInitlp, PyConsSepalp, PyConsSepasol,
                                              PyConsEnfolp, PyConsEnforelax, PyConsEnfops, PyConsCheck, PyConsProp, PyConsPresol, PyConsResprop, PyConsLock,
                                              PyConsActive, PyConsDeactive, PyConsEnable, PyConsDisable, PyConsDelvars, PyConsPrint, PyConsCopy,
                                              PyConsParse, PyConsGetvars, PyConsGetnvars, PyConsGetdivebdchgs, PyConsGetPermSymGraph, PyConsGetSignedPermSymGraph,
                                              <SCIP_CONSHDLRDATA*>conshdlr))
        conshdlr.model = self
        conshdlr.name = name
        self._plugins.append(conshdlr)

    def deactivatePricer(self, Pricer pricer):
        """
        Deactivate the given pricer.

        Parameters
        ----------
        pricer : Pricer
            the pricer to deactivate
        """
        cdef SCIP_PRICER* scip_pricer
        PY_SCIP_CALL(SCIPdeactivatePricer(self._scip, pricer.scip_pricer))

    def translateSubSol(self, BaseModel sub_model, Solution sol, heur) -> Solution:
        """
		Translates a solution of a model copy into a solution of the main model

		Parameters
		----------
		sub_model : Model
			The python-wrapper of the subscip
		sol : Solution
			The python-wrapper of the solution of the subscip
		heur : Heur
			The python-wrapper of the heuristic that found the solution

		Returns
		-------
		solution : Solution
			The corresponding solution in the main model
        """
        cdef SCIP_VAR** vars
        cdef SCIP_SOL* real_sol
        cdef SCIP_SOL* subscip_sol = sol.sol
        cdef SCIP_HEUR* _heur
        cdef SCIP_Bool success
        cdef int i
        cdef _VarArray wrapper
        
        assert sub_model.getNVars(False) >= self.getNVars(True), "The sub_model must have at least as many variables as the main model."

        wrapper = _VarArray(sub_model.getVars(False))
        _heur = SCIPfindHeur(self._scip, str_conversion(heur.name))
        PY_SCIP_CALL( SCIPtranslateSubSol(self._scip, sub_model._scip, subscip_sol, _heur, wrapper.ptr, &real_sol) )
        solution = Solution.create(self._scip, real_sol)

        return solution

    def includePresol(self, Presol presol, name, desc, priority, maxrounds, timing=SCIP_PRESOLTIMING_FAST):
        """
        Include a presolver.

        Parameters
        ----------
        presol : Presol
            presolver
        name : str
            name of presolver
        desc : str
            description of presolver
        priority : int
            priority of the presolver (>= 0: before, < 0: after constraint handlers)
        maxrounds : int
            maximal number of presolving rounds the presolver participates in (-1: no limit)
        timing : PY_SCIP_PRESOLTIMING, optional
             timing mask of presolver (Default value = SCIP_PRESOLTIMING_FAST)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePresol(self._scip, n, d, priority, maxrounds, timing, PyPresolCopy, PyPresolFree, PyPresolInit,
                                            PyPresolExit, PyPresolInitpre, PyPresolExitpre, PyPresolExec, <SCIP_PRESOLDATA*>presol))
        presol.model = self
        self._plugins.append(presol)

    def includeSepa(self, Sepa sepa, name, desc, priority=0, freq=10, maxbounddist=1.0, usessubscip=False, delay=False):
        """
        Include a separator

        :param Sepa sepa: separator
        :param name: name of separator
        :param desc: description of separator
        :param priority: priority of separator (>= 0: before, < 0: after constraint handlers)
        :param freq: frequency for calling separator
        :param maxbounddist: maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separation
        :param usessubscip: does the separator use a secondary SCIP instance? (Default value = False)
        :param delay: should separator be delayed, if other separators found cuts? (Default value = False)


        Parameters
        ----------
        sepa : Sepa
            separator
        name : str
            name of separator
        desc : str
            description of separator
        priority : int, optional
            priority of separator (>= 0: before, < 0: after constraint handlers) (default=0)
        freq : int, optional
            frequency for calling separator (default=10)
        maxbounddist : float, optional
            maximal relative distance from current node's dual bound to primal
            bound compared to best node's dual bound for applying separation.
            (default = 1.0)
        usessubscip : bool, optional
            does the separator use a secondary SCIP instance? (Default value = False)
        delay : bool, optional
            should separator be delayed if other separators found cuts? (Default value = False)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeSepa(self._scip, n, d, priority, freq, maxbounddist, usessubscip, delay, PySepaCopy, PySepaFree,
                                          PySepaInit, PySepaExit, PySepaInitsol, PySepaExitsol, PySepaExeclp, PySepaExecsol, <SCIP_SEPADATA*>sepa))
        sepa.model = self
        sepa.name = name
        self._plugins.append(sepa)

    def includeReader(self, Reader reader, name, desc, ext):
        """
        Include a reader.

        Parameters
        ----------
        reader : Reader
            reader
        name : str
            name of reader
        desc : str
            description of reader
        ext : str
            file extension of reader

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        e = str_conversion(ext)
        PY_SCIP_CALL(SCIPincludeReader(self._scip, n, d, e, PyReaderCopy, PyReaderFree,
                                          PyReaderRead, PyReaderWrite, <SCIP_READERDATA*>reader))
        reader.model = self
        reader.name = name
        self._plugins.append(reader)

    def includeProp(self, Prop prop, name, desc, presolpriority, presolmaxrounds,
                    proptiming, presoltiming=SCIP_PRESOLTIMING_FAST, priority=1, freq=1, delay=True):
        """
        Include a propagator.

        Parameters
        ----------
        prop : Prop
            propagator
        name : str
            name of propagator
        desc : str
            description of propagator
        presolpriority : int
            presolving priority of the propgator (>= 0: before, < 0: after constraint handlers)
        presolmaxrounds : int
            maximal number of presolving rounds the propagator participates in (-1: no limit)
        proptiming : SCIP_PROPTIMING
            positions in the node solving loop where propagation method of constraint handlers should be executed
        presoltiming : PY_SCIP_PRESOLTIMING, optional
            timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING_FAST)
        priority : int, optional
            priority of the propagator (Default value = 1)
        freq : int, optional
            frequency for calling propagator (Default value = 1)
        delay : bool, optional
            should propagator be delayed if other propagators have found reductions? (Default value = True)

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
        self._plugins.append(prop)

    def includeHeur(self, Heur heur, name, desc, dispchar, priority=10000, freq=1, freqofs=0,
                    maxdepth=-1, timingmask=SCIP_HEURTIMING_BEFORENODE, usessubscip=False):
        """
        Include a primal heuristic.

        Parameters
        ----------
        heur : Heur
            heuristic
        name : str
            name of heuristic
        desc : str
            description of heuristic
        dispchar : str
            display character of heuristic. Please use a single length string.
        priority : int. optional
            priority of the heuristic (Default value = 10000)
        freq : int, optional
            frequency for calling heuristic (Default value = 1)
        freqofs : int. optional
            frequency offset for calling heuristic (Default value = 0)
        maxdepth : int, optional
            maximal depth level to call heuristic at (Default value = -1)
        timingmask : PY_SCIP_HEURTIMING, optional
            positions in the node solvingreturn {
            'result': SCIP_RESULT.SUCCESS,
            'lowerbound': 10e4
        } loop where heuristic should be executed
            (Default value = SCIP_HEURTIMING_BEFORENODE)
        usessubscip : bool, optional
            does the heuristic use a secondary SCIP instance? (Default value = False)

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
        self._plugins.append(heur)
    
    def includeIISfinder(self, IISfinder iisfinder, name, desc, priority=10000, freq=1):
        """
        Include an IIS (Irreducible Infeasible Set) finder handler.

        Parameters
        ----------
        iisfinder : IISfinder
            IIS finder
        name : str
            name of IIS finder
        desc : str
            description of IIS finder
        priority : int, optional
            priority of the IISfinder (#todo description)
        freq : int, optional
            frequency for calling IIS finder

        """
        cdef SCIP_IISFINDER* scip_iisfinder

        nam = str_conversion(name)
        des = str_conversion(desc)

        iisfinder.iis = IIS()

        PY_SCIP_CALL(SCIPincludeIISfinder(self._scip, nam, des, priority, PyiisfinderCopy, PyiisfinderFree,
                                         PyiisfinderExec, <SCIP_IISFINDERDATA*> iisfinder))

        scip_iisfinder = SCIPfindIISfinder(self._scip, nam)
        iisfinder.model = self
        iisfinder.name = name
        self._plugins.append(iisfinder)
        iisfinder.scip_iisfinder = scip_iisfinder

    def generateIIS(self):
        """
        Generates an Irreducible Infeasible Subsystem (IIS) from the current
        problem.

        Returns
        -------
        IIS

        """
        cdef SCIP_IIS* _iis

        PY_SCIP_CALL( SCIPgenerateIIS(self._scip) )

        _iis = SCIPgetIIS(self._scip)
        return IIS.create(_iis)
    
    def getIIS(self):
        """
        Get the IIS object.
        Note: Needs to be called after generateIIS, or after a single execution of the iisfinderExec.
        
        Returns
        -------
        IIS 
        """
        cdef SCIP_IIS* _iis

        _iis = SCIPgetIIS(self._scip)
        assert _iis != NULL, "No IIS exists. You need to first call generateIIS() or run the iisfinderexec method of your custom IISfinder class."

        return IIS.create(_iis)

    def includeRelax(self, Relax relax, name, desc, priority=10000, freq=1):
        """
        Include a relaxation handler.

        Parameters
        ----------
        relax : Relax
            relaxation handler
        name : str
            name of relaxation handler
        desc : str
            description of relaxation handler
        priority : int, optional
            priority of the relaxation handler (negative: after LP, non-negative: before LP, Default value = 10000)
        freq : int, optional
            frequency for calling relaxation handler

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeRelax(self._scip, nam, des, priority, freq, PyRelaxCopy, PyRelaxFree, PyRelaxInit, PyRelaxExit,
                                          PyRelaxInitsol, PyRelaxExitsol, PyRelaxExec, <SCIP_RELAXDATA*> relax))
        relax.model = self
        relax.name = name
        self._plugins.append(relax)

    def includeCutsel(self, Cutsel cutsel, name, desc, priority):
        """
        Include a cut selector.

        Parameters
        ----------
        cutsel : Cutsel
            cut selector
        name : str
            name of cut selector
        desc : str
            description of cut selector
        priority : int
            priority of the cut selector

        """

        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeCutsel(self._scip, nam, des,
                                       priority, PyCutselCopy, PyCutselFree, PyCutselInit, PyCutselExit,
                                       PyCutselInitsol, PyCutselExitsol, PyCutselSelect,
                                       <SCIP_CUTSELDATA*> cutsel))
        cutsel.model = self
        self._plugins.append(cutsel)

    def includeBranchrule(self, Branchrule branchrule, name, desc, priority, maxdepth, maxbounddist):
        """
        Include a branching rule.

        Parameters
        ----------
        branchrule : Branchrule
            branching rule
        name : str
            name of branching rule
        desc : str
            description of branching rule
        priority : int
            priority of branching rule
        maxdepth : int
            maximal depth level up to which this branching rule should be used (or -1)
        maxbounddist : float
            maximal relative distance from current node's dual bound to primal bound
            compared to best node's dual bound for applying branching rule
            (0.0: only on current best node, 1.0: on all nodes)

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBranchrule(self._scip, nam, des,
                                          priority, maxdepth, maxbounddist,
                                          PyBranchruleCopy, PyBranchruleFree, PyBranchruleInit, PyBranchruleExit,
                                          PyBranchruleInitsol, PyBranchruleExitsol, PyBranchruleExeclp, PyBranchruleExecext,
                                          PyBranchruleExecps, <SCIP_BRANCHRULEDATA*> branchrule))
        branchrule.model = self
        self._plugins.append(branchrule)

    def includeNodesel(self, Nodesel nodesel, name, desc, stdpriority, memsavepriority):
        """
        Include a node selector.

        Parameters
        ----------
        nodesel : Nodesel
            node selector
        name : str
            name of node selector
        desc : str
            description of node selector
        stdpriority : int
            priority of the node selector in standard mode
        memsavepriority : int
            priority of the node selector in memory saving mode

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeNodesel(self._scip, nam, des,
                                          stdpriority, memsavepriority,
                                          PyNodeselCopy, PyNodeselFree, PyNodeselInit, PyNodeselExit,
                                          PyNodeselInitsol, PyNodeselExitsol, PyNodeselSelect, PyNodeselComp,
                                          <SCIP_NODESELDATA*> nodesel))
        nodesel.model = self
        self._plugins.append(nodesel)

    def includeBenders(self, Benders benders, name, desc, priority=1, cutlp=True, cutpseudo=True, cutrelax=True,
            shareaux=False):
        """
        Include a Benders' decomposition.

        Parameters
        ----------
        benders : Benders
            the Benders decomposition
        name : str
            the name
        desc : str
            the description
        priority : int, optional
            priority of the Benders' decomposition
        cutlp : bool, optional
            should Benders' cuts be generated from LP solutions
        cutpseudo : bool, optional
            should Benders' cuts be generated from pseudo solutions
        cutrelax : bool, optional
            should Benders' cuts be generated from relaxation solutions
        shareaux : bool, optional
            should the Benders' decomposition share the auxiliary variables of the
            highest priority Benders' decomposition

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBenders(self._scip, n, d,
                                            priority, cutlp, cutrelax, cutpseudo, shareaux,
                                            PyBendersCopy, PyBendersFree, PyBendersInit, PyBendersExit, PyBendersInitpre,
                                            PyBendersExitpre, PyBendersInitsol, PyBendersExitsol, PyBendersGetvar,
                                            PyBendersCreatesub, PyBendersPresubsolve, PyBendersSolvesubconvex,
                                            PyBendersSolvesub, PyBendersPostsolve, PyBendersFreesub,
                                            <SCIP_BENDERSDATA*>benders))
        cdef SCIP_BENDERS* scip_benders
        scip_benders = SCIPfindBenders(self._scip, n)
        benders.model = self
        benders.name = name
        benders._benders = scip_benders
        self._plugins.append(benders)

    def includeBenderscut(self, Benders benders, Benderscut benderscut, name, desc, priority=1, islpcut=True):
        """
        Include a Benders' decomposition cutting method

        Parameters
        ----------
        benders : Benders
            the Benders' decomposition that this cutting method is attached to
        benderscut : Benderscut
            the Benders' decomposition cutting method
        name : str
            the name
        desc : str
            the description
        priority : int. optional
            priority of the Benders' decomposition (Default = 1)
        islpcut : bool, optional
            is this cutting method suitable for generating cuts for convex relaxations?
            (Default = True)

        """
        cdef SCIP_BENDERS* _benders

        _benders = benders._benders

        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBenderscut(self._scip, _benders, n, d, priority, islpcut,
                                            PyBenderscutCopy, PyBenderscutFree, PyBenderscutInit, PyBenderscutExit,
                                            PyBenderscutInitsol, PyBenderscutExitsol, PyBenderscutExec,
                                            <SCIP_BENDERSCUTDATA*>benderscut))

        cdef SCIP_BENDERSCUT* scip_benderscut
        scip_benderscut = SCIPfindBenderscut(_benders, n)
        benderscut.model = self
        benderscut.benders = benders
        benderscut.name = name
        self._plugins.append(benderscut)

    def getNLPBranchCands(self):
        """
        Gets number of branching candidates for LP solution branching (number of fractional variables)

        Returns
        -------
        int
    	    number of LP branching candidates

        """
        return SCIPgetNLPBranchCands(self._scip)

    def getPseudoBranchCands(self):
        """
        Gets branching candidates for pseudo solution branching (non-fixed variables)
        along with the number of candidates.

        Returns
        -------
        list of Variable
            list of variables of pseudo branching candidates
        int
            number of pseudo branching candidates
        int
            number of candidates with maximal priority

        """
        cdef SCIP_VAR** pseudocands
        cdef int npseudocands
        cdef int npriopseudocands
        cdef int i

        PY_SCIP_CALL(SCIPgetPseudoBranchCands(self._scip, &pseudocands, &npseudocands, &npriopseudocands))

        return ([self._getOrCreateVar(pseudocands[i]) for i in range(npseudocands)], npseudocands, npriopseudocands)

    def branchVar(self, Variable variable):
        """
        Branch on a non-continuous variable.

        Parameters
        ----------
        variable : Variable
            Variable to branch on

        Returns
        -------
        Node
            Node created for the down (left) branch
        Node or None
            Node created for the equal child (middle child). Only exists if branch variable is integer
        Node
            Node created for the up (right) branch

        """
        cdef SCIP_NODE* downchild
        cdef SCIP_NODE* eqchild
        cdef SCIP_NODE* upchild
        cdef _VarArray wrapper = _VarArray(variable)

        PY_SCIP_CALL(SCIPbranchVar(self._scip, wrapper.ptr[0], &downchild, &eqchild, &upchild))
        return Node.create(downchild), Node.create(eqchild), Node.create(upchild)

    def createChild(self, nodeselprio, estimate):
        """
        Create a child node of the focus node.

        Parameters
        ----------
        nodeselprio : float
            node selection priority of new node
        estimate : float
            estimate for (transformed) objective value of best feasible solution in subtree

        Returns
        -------
        Node
            the child which was created

        """
        cdef SCIP_NODE* child
        PY_SCIP_CALL(SCIPcreateChild(self._scip, &child, nodeselprio, estimate))
        return Node.create(child)

    def inRepropagation(self):
        """
        Returns if the current node is already solved and only propagated again.

        Returns
        -------
        bool

        """
        return SCIPinRepropagation(self._scip)

    # Probing methods (Probing is tree based)
    def startProbing(self):
        """Initiates probing, making methods SCIPnewProbingNode(), SCIPbacktrackProbing(), SCIPchgVarLbProbing(),
           SCIPchgVarUbProbing(), SCIPfixVarProbing(), SCIPpropagateProbing(), SCIPsolveProbingLP(), etc available.
        """
        PY_SCIP_CALL( SCIPstartProbing(self._scip) )

    def endProbing(self):
        """Quits probing and resets bounds and constraints to the focus node's environment."""
        PY_SCIP_CALL( SCIPendProbing(self._scip) )

    def newProbingNode(self):
        """Creates a new probing sub node, whose changes can be undone by backtracking to a higher node in the
        probing path with a call to backtrackProbing().
        """
        PY_SCIP_CALL( SCIPnewProbingNode(self._scip) )

    def backtrackProbing(self, probingdepth):
        """
        Undoes all changes to the problem applied in probing up to the given probing depth.

        Parameters
        ----------
        probingdepth : int
            probing depth of the node in the probing path that should be reactivated

        """
        PY_SCIP_CALL( SCIPbacktrackProbing(self._scip, probingdepth) )

    def getProbingDepth(self):
        """Returns the current probing depth."""
        return SCIPgetProbingDepth(self._scip)

    def isObjChangedProbing(self):
        """
        Returns whether the objective function has changed during probing mode.

        Returns
        -------
        bool

        """
        return SCIPisObjChangedProbing(self._scip)

    def inProbing(self):
        """
        Returns whether we are in probing mode;
        probing mode is activated via startProbing() and stopped via endProbing().

        Returns
        -------
        bool

        """
        return SCIPinProbing(self._scip)

    def solveProbingLP(self, itlim = -1):
        """
        Solves the LP at the current probing node (cannot be applied at preprocessing stage)
        no separation or pricing is applied.

        Parameters
        ----------
        itlim : int
            maximal number of LP iterations to perform (Default value = -1, that is, no limit)

        Returns
        -------
        lperror : bool
            if an unresolved lp error occured
        cutoff : bool
            whether the LP was infeasible or the objective limit was reached

        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL( SCIPsolveProbingLP(self._scip, itlim, &lperror, &cutoff) )
        return lperror, cutoff

    def solveProbingLPWithPricing(self, pretendroot=False, displayinfo=False, maxpricerounds=-1):
        """
        Solves the LP at the current probing node (cannot be applied at preprocessing stage) and applies pricing
        until the LP is solved to optimality; no separation is applied.

        Parameters
        ----------
        pretendroot : bool
            should the pricers be called as if we are at the root node? (Default value = False)
        displayinfo : bool
            should info lines be displayed after each pricing round? (Default value = False)
        maxpricerounds : int
            maximal number of pricing rounds (-1: no limit); a finite limit means that the LP might not be
            solved to optimality! (Default value = -1)

        Returns
        -------
        lperror : bool
            whether an unresolved LP error occurred
        cutoff : bool
            whether the probing LP was infeasible or the objective limit was reached

        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL( SCIPsolveProbingLPWithPricing(self._scip, pretendroot, displayinfo, maxpricerounds, &lperror, &cutoff) )
        return lperror, cutoff

    def applyCutsProbing(self):
        """
        Applies the cuts in the separation storage to the LP and clears the storage afterwards;
        this method can only be applied during probing; the user should resolve the probing LP afterwards
        in order to get a new solution.
        returns:

        Returns
        -------
        cutoff : bool
            whether an empty domain was created

        """
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL( SCIPapplyCutsProbing(self._scip, &cutoff) )
        return cutoff

    def propagateProbing(self, maxproprounds):
        """
        Applies domain propagation on the probing sub problem, that was changed after SCIPstartProbing() was called;
        the propagated domains of the variables can be accessed with the usual bound accessing calls SCIPvarGetLbLocal()
        and SCIPvarGetUbLocal(); the propagation is only valid locally, i.e. the local bounds as well as the changed
        bounds due to SCIPchgVarLbProbing(), SCIPchgVarUbProbing(), and SCIPfixVarProbing() are used for propagation.

        Parameters
        ----------
        maxproprounds : int
            maximal number of propagation rounds (Default value = -1, that is, no limit)

        Returns
        -------
        cutoff : bool
            whether the probing node can be cutoff
        ndomredsfound : int
            number of domain reductions found

        """
        cdef SCIP_Bool cutoff
        cdef SCIP_Longint ndomredsfound

        PY_SCIP_CALL( SCIPpropagateProbing(self._scip, maxproprounds, &cutoff, &ndomredsfound) )
        return cutoff, ndomredsfound

    def interruptSolve(self):
        """Interrupt the solving process as soon as possible."""
        PY_SCIP_CALL(SCIPinterruptSolve(self._scip))

    def restartSolve(self):
        """Restarts the solving process as soon as possible."""
        PY_SCIP_CALL(SCIPrestartSolve(self._scip))

    # Solution functions

    cdef SCIP_HEUR* _getHeur(self, object heur):
        """
        Retrieves the SCIP_HEUR pointer corresponding to a heuristic object.

        Parameters
        ----------
        heur : Heur or None
            The heuristic object to resolve.

        Returns
        -------
        SCIP_HEUR*
            Pointer to the SCIP heuristic, or NULL if heur is not a Heur instance or not found.

        """
        if isinstance(heur, Heur):
            n = str_conversion(heur.name)
            return SCIPfindHeur(self._scip, n)
        return NULL

    def checkSol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, original=False):
        """
        Check given primal solution for feasibility without adding it to the storage.

        Parameters
        ----------
        solution : Solution
            solution to store
        printreason : bool, optional
            should all reasons of violations be printed? (Default value = True)
        completely : bool, optional
            should all violation be checked? (Default value = False)
        checkbounds : bool, optional
            should the bounds of the variables be checked? (Default value = True)
        checkintegrality : bool, optional
            has integrality to be checked? (Default value = True)
        checklprows : bool, optional
            have current LP rows (both local and global) to be checked? (Default value = True)
        original : bool, optional
            must the solution be checked against the original problem (Default value = False)

        Returns
        -------
        feasible : bool
            whether the given solution was feasible or not

        """
        cdef SCIP_Bool feasible
        if original:
            PY_SCIP_CALL(SCIPcheckSolOrig(self._scip, solution.sol, &feasible, printreason, completely))
        else:
            PY_SCIP_CALL(SCIPcheckSol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &feasible))
        return feasible

    def addSol(self, Solution solution, free=True):
        """
        Try to add a solution to the storage.

        Parameters
        ----------
        solution : Solution
            solution to store
        free : bool, optional
            should solution be freed afterwards? (Default value = True)

        Returns
        -------
        stored : bool
            stores whether given solution was good enough to keep

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPaddSolFree(self._scip, &solution.sol, &stored))
        else:
            PY_SCIP_CALL(SCIPaddSol(self._scip, solution.sol, &stored))
        return stored

    def freeSol(self, Solution solution):
        """
        Free given solution

        Parameters
        ----------
        solution : Solution
            solution to be freed

        """
        PY_SCIP_CALL(SCIPfreeSol(self._scip, &solution.sol))

    def getNSols(self):
        """
        Gets number of feasible primal solutions stored in the solution storage in case the problem is transformed;
        in case the problem stage is SCIP_STAGE_PROBLEM, the number of solution in the original solution candidate
        storage is returned.

        Returns
        -------
        int

        """
        return SCIPgetNSols(self._scip)

    def getNSolsFound(self):
        """
        Gets number of feasible primal solutions found so far.

        Returns
        -------
        int

        """
        return SCIPgetNSolsFound(self._scip)

    def getNLimSolsFound(self):
        """
        Gets number of feasible primal solutions respecting the objective limit found so far.

        Returns
        -------
        int

        """
        return SCIPgetNLimSolsFound(self._scip)

    def getNBestSolsFound(self):
        """
        Gets number of feasible primal solutions found so far,
        that improved the primal bound at the time they were found.

        Returns
        -------
        int

        """
        return SCIPgetNBestSolsFound(self._scip)

    def getSols(self):
        """
        Retrieve list of all feasible primal solutions stored in the solution storage.

        Returns
        -------
        list of Solution

        """
        cdef SCIP_SOL** _sols = SCIPgetSols(self._scip)
        cdef int nsols = SCIPgetNSols(self._scip)
        cdef int i

        sols = []

        for i in range(nsols):
            sols.append(Solution.create(self._scip, _sols[i]))

        return sols

    def getBestSol(self):
        """
        Retrieve currently best known feasible primal solution.

        Returns
        -------
        Solution or None

        """
        cdef SCIP_SOL* _sol = SCIPgetBestSol(self._scip)
        if _sol == NULL:
            return None
        self._bestSol = Solution.create(self._scip, _sol)
        return self._bestSol

    def getSolTime(self, Solution sol):
        """
        Get clock time when this solution was found.

        Parameters
        ----------
        sol : Solution

        Returns
        -------
        float

        """
        if sol is None or sol.sol == NULL:
            raise ValueError("Cannot get solution time: solution is None or NULL")
        return SCIPgetSolTime(self._scip, sol.sol)

    cdef _ensureFreshBestSol(self):
        """
        Refreshes the internal _bestSol attribute if the underlying SCIP pointer has changed.
        """
        cdef SCIP_SOL* current_best_sol = SCIPgetBestSol(self._scip)
        if self._bestSol is None or self._bestSol.sol != current_best_sol:
            self._bestSol = Solution.create(self._scip, current_best_sol)

    def getObjVal(self, original=True):
        """
        Retrieve the objective value of the best solution.

        Parameters
        ----------
        original : bool, optional
            objective value in original space (Default value = True)

        Returns
        -------
        float

        """

        self._ensureFreshBestSol()

        if SCIPgetNSols(self._scip) == 0:
            if self.getStage() != SCIP_STAGE_SOLVING:
                raise Warning("Without a solution, method can only be called in stage SOLVING.")
        else:
            assert self._bestSol.sol != NULL

            if SCIPsolIsOriginal(self._bestSol.sol):
                min_stage_requirement = SCIP_STAGE_PROBLEM
            else:
                min_stage_requirement = SCIP_STAGE_TRANSFORMING

            if not self.getStage() >= min_stage_requirement:
                raise Warning("method cannot be called in stage %i." % self.getStage)

        return self.getSolObjVal(self._bestSol, original)

    def hasPrimalRay(self):
        """
        Returns whether a primal ray is stored that proves unboundedness of the LP relaxation.

        Returns
        -------
        bool

        """
        return SCIPhasPrimalRay(self._scip)

    def writeName(self, Variable var):
        """
        Write the name of the variable to the std out.

        Parameters
        ----------
        var : Variable

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPwriteVarName(self._scip, NULL, var.scip_var, False))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def getStage(self):
        """
        Retrieve current SCIP stage.

        Returns
        -------
        int

        """
        return SCIPgetStage(self._scip)

    def getStageName(self):
        """
        Returns name of current stage as string.

        Returns
        -------
        str

        """
        if not StageNames:
            self._getStageNames()
        return StageNames[self.getStage()]

    def _getStageNames(self):
        """Gets names of stages."""
        for name in dir(PY_SCIP_STAGE):
            attr = getattr(PY_SCIP_STAGE, name)
            if isinstance(attr, int):
                StageNames[attr] = name

    def getStatus(self):
        """
        Retrieve solution status.

        Returns
        -------
        str
            The status of SCIP.

        """
        cdef SCIP_STATUS stat = SCIPgetStatus(self._scip)
        if stat == SCIP_STATUS_OPTIMAL:
            return "optimal"
        elif stat == SCIP_STATUS_TIMELIMIT:
            return "timelimit"
        elif stat == SCIP_STATUS_INFEASIBLE:
            return "infeasible"
        elif stat == SCIP_STATUS_UNBOUNDED:
            return "unbounded"
        elif stat == SCIP_STATUS_USERINTERRUPT:
            return "userinterrupt"
        elif stat == SCIP_STATUS_INFORUNBD:
            return "inforunbd"
        elif stat == SCIP_STATUS_NODELIMIT:
            return "nodelimit"
        elif stat == SCIP_STATUS_TOTALNODELIMIT:
            return "totalnodelimit"
        elif stat == SCIP_STATUS_STALLNODELIMIT:
            return "stallnodelimit"
        elif stat == SCIP_STATUS_GAPLIMIT:
            return "gaplimit"
        elif stat == SCIP_STATUS_MEMLIMIT:
            return "memlimit"
        elif stat == SCIP_STATUS_SOLLIMIT:
            return "sollimit"
        elif stat == SCIP_STATUS_BESTSOLLIMIT:
            return "bestsollimit"
        elif stat == SCIP_STATUS_RESTARTLIMIT:
            return  "restartlimit"
        elif stat == SCIP_STATUS_PRIMALLIMIT:
            return "primallimit"
        elif stat == SCIP_STATUS_DUALLIMIT:
            return "duallimit"
        else:
            return "unknown"

    def getObjectiveSense(self):
        """
        Retrieve objective sense.

        Returns
        -------
        str

        """
        cdef SCIP_OBJSENSE sense = SCIPgetObjsense(self._scip)
        if sense == SCIP_OBJSENSE_MAXIMIZE:
            return "maximize"
        elif sense == SCIP_OBJSENSE_MINIMIZE:
            return "minimize"
        else:
            return "unknown"

    cdef SCIP_EVENTHDLR* _getEventhdlr(self, object eventhdlr):
        """
        Resolves an event handler object to its underlying SCIP_EVENTHDLR pointer.

        Parameters
        ----------
        eventhdlr : Eventhdlr
            The event handler object to resolve.

        Returns
        -------
        SCIP_EVENTHDLR*
            Pointer to the SCIP event handler, or NULL if not found.

        Raises
        ------
        Warning
            If eventhdlr is not an instance of Eventhdlr.

        """
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            return SCIPfindEventhdlr(self._scip, n)
        raise Warning(f"Expected an Eventhdlr instance, got {type(eventhdlr).__name__}")

    def catchEvent(self, eventtype, Eventhdlr eventhdlr):
        """
        Catches a global (not variable or row dependent) event.

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr = self._getEventhdlr(eventhdlr)

        PY_SCIP_CALL(SCIPcatchEvent(self._scip, eventtype, _eventhdlr, NULL, NULL))
        eventhdlr._caught_events.append(eventtype)

    def dropEvent(self, eventtype, Eventhdlr eventhdlr):
        """
        Drops a global event (stops tracking the event).

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr = self._getEventhdlr(eventhdlr)

        PY_SCIP_CALL(SCIPdropEvent(self._scip, eventtype, _eventhdlr, NULL, -1))
        if eventtype in eventhdlr._caught_events:
            eventhdlr._caught_events.remove(eventtype)

    def catchVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        """
        Catches an objective value or domain change event on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr = self._getEventhdlr(eventhdlr)
        PY_SCIP_CALL(SCIPcatchVarEvent(self._scip, var.scip_var, eventtype, _eventhdlr, NULL, NULL))

    def dropVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        """
        Drops an objective value or domain change event (stops tracking the event) on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr = self._getEventhdlr(eventhdlr)
        PY_SCIP_CALL(SCIPdropVarEvent(self._scip, var.scip_var, eventtype, _eventhdlr, NULL, -1))

    def catchRowEvent(self, Row row, eventtype, Eventhdlr eventhdlr):
        """
        Catches a row coefficient, constant, or side change event on the given row.

        Parameters
        ----------
        row : Row
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr = self._getEventhdlr(eventhdlr)
        PY_SCIP_CALL(SCIPcatchRowEvent(self._scip, row.scip_row, eventtype, _eventhdlr, NULL, NULL))

    def dropRowEvent(self, Row row, eventtype, Eventhdlr eventhdlr):
        """
        Drops a row coefficient, constant, or side change event (stops tracking the event) on the given row.

        Parameters
        ----------
        row : Row
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr = self._getEventhdlr(eventhdlr)
        PY_SCIP_CALL(SCIPdropRowEvent(self._scip, row.scip_row, eventtype, _eventhdlr, NULL, -1))

    # Statistic Methods

    def printStatistics(self, filename=None):
        """
        Print statistics.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default = None)

        """ 

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        if not filename:
            PY_SCIP_CALL(SCIPprintStatistics(self._scip, NULL))
        else:
            with open(filename, "w") as f:
                cfile = fdopen(f.fileno(), "w")
                PY_SCIP_CALL(SCIPprintStatistics(self._scip, cfile))

        locale.setlocale(locale.LC_NUMERIC,user_locale)
    
    def printStatisticsJson(self, filename=None):
        """
        Print statistics in JSON format.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default = None)

        """ 

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        if not filename:
            PY_SCIP_CALL(SCIPprintStatisticsJson(self._scip, NULL))
        else:
            with open(filename, "w") as f:
                cfile = fdopen(f.fileno(), "w")
                PY_SCIP_CALL(SCIPprintStatisticsJson(self._scip, cfile))

        locale.setlocale(locale.LC_NUMERIC,user_locale)
    
    def printStatisticsJson(self):
        """Print statistics in JSON format."""
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPprintStatisticsJson(self._scip, NULL))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeStatistics(self, filename="origprob.stats"):
        """
        Write statistics to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default = "origprob.stats")

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintStatistics(self._scip, cfile))

        locale.setlocale(locale.LC_NUMERIC,user_locale)
    

    def writeStatisticsJson(self, filename="origprob.stats.json"):
        """
        Write statistics to a JSON file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default = "origprob.stats.json")

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintStatisticsJson(self._scip, cfile))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def getMemUsed(self):
        """
        Gets the total number of bytes used in block and buffer memory.

        Returns
        -------
        int

        """
        return SCIPgetMemUsed(self._scip)

    def getMemTotal(self):
        """
        Gets the total number of bytes in block and buffer memory
        (i.e., total allocated, including unused).

        Returns
        -------
        int

        """
        return SCIPgetMemTotal(self._scip)

    def getMemExternEstim(self):
        """
        Gets the estimated number of bytes used by external software,
        e.g., the LP solver.

        Returns
        -------
        int

        """
        return SCIPgetMemExternEstim(self._scip)

    def getNLPs(self):
        """
        Gets total number of LPs solved so far.

        Returns
        -------
        int

        """
        return SCIPgetNLPs(self._scip)

    # Verbosity Methods

    def hideOutput(self, quiet = True):
        """
        Hide the output.

        Parameters
        ----------
        quiet : bool, optional
            hide output? (Default value = True)

        """
        SCIPsetMessagehdlrQuiet(self._scip, quiet)

    # Output Methods

    def redirectOutput(self):
        """Send output to python instead of terminal."""

        cdef SCIP_MESSAGEHDLR *myMessageHandler

        PY_SCIP_CALL(SCIPmessagehdlrCreate(&myMessageHandler, False, NULL, False, relayMessage, relayMessage, relayMessage, NULL, NULL))
        PY_SCIP_CALL(SCIPsetMessagehdlr(self._scip, myMessageHandler))
        SCIPmessageSetErrorPrinting(relayErrorMessage, NULL)

    def setLogfile(self, path):
        """
        Sets the log file name for the currently installed message handler.

        Parameters
        ----------
        path : str or None
            name of log file, or None (no log)

        """
        if path:
            c_path = str_conversion(path)
            SCIPsetMessagehdlrLogfile(self._scip, c_path)
        else:
            SCIPsetMessagehdlrLogfile(self._scip, NULL)

    # Parameter Methods

    def setBoolParam(self, name, value):
        """
        Set a boolean-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : bool
            value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, value))

    def setIntParam(self, name, value):
        """
        Set an int-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : int
            value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, value))

    def setLongintParam(self, name, value):
        """
        Set a long-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : int
            value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, value))

    def setRealParam(self, name, value):
        """
        Set a real-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : float
            value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, value))

    def setCharParam(self, name, value):
        """
        Set a char-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : str
            value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, ord(value)))

    def setStringParam(self, name, value):
        """
        Set a string-valued parameter.

        Parameters
        ----------
        name : str
            name of parameter
        value : str
            value of parameter

        """
        n = str_conversion(name)
        v = str_conversion(value)
        PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, v))

    def setParam(self, name, value):
        """Set a parameter with value in int, bool, real, long, char or str.

        Parameters
        ----------
        name : str
            name of parameter
        value : object
            value of parameter

        """
        cdef SCIP_PARAM* param

        n = str_conversion(name)
        param = SCIPgetParam(self._scip, n)

        if param == NULL:
            raise KeyError("Not a valid parameter name")

        paramtype =  SCIPparamGetType(param)

        if paramtype == SCIP_PARAMTYPE_BOOL:
            PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, bool(int(value))))
        elif paramtype == SCIP_PARAMTYPE_INT:
            PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, int(value)))
        elif paramtype == SCIP_PARAMTYPE_LONGINT:
            PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, int(value)))
        elif paramtype == SCIP_PARAMTYPE_REAL:
            PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, float(value)))
        elif paramtype == SCIP_PARAMTYPE_CHAR:
            PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, ord(value)))
        elif paramtype == SCIP_PARAMTYPE_STRING:
            v = str_conversion(value)
            PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, v))

    def getParam(self, name):
        """
        Get the value of a parameter of type
        int, bool, real, long, char or str.

        Parameters
        ----------
        name : str
            name of parameter

        Returns
        -------
        object

        """
        cdef SCIP_PARAM* param

        n = str_conversion(name)
        param = SCIPgetParam(self._scip, n)

        if param == NULL:
            raise KeyError("Not a valid parameter name")

        paramtype =  SCIPparamGetType(param)

        if paramtype == SCIP_PARAMTYPE_BOOL:
            return SCIPparamGetBool(param)
        elif paramtype == SCIP_PARAMTYPE_INT:
            return SCIPparamGetInt(param)
        elif paramtype == SCIP_PARAMTYPE_LONGINT:
            return SCIPparamGetLongint(param)
        elif paramtype == SCIP_PARAMTYPE_REAL:
            return SCIPparamGetReal(param)
        elif paramtype == SCIP_PARAMTYPE_CHAR:
            return chr(SCIPparamGetChar(param))
        elif paramtype == SCIP_PARAMTYPE_STRING:
            return SCIPparamGetString(param).decode('utf-8')

    def getParams(self):
        """
        Gets the values of all parameters as a dict mapping parameter names
        to their values.

        Returns
        -------
        dict of str to object
            dict mapping parameter names to their values.

        """
        cdef SCIP_PARAM** params = SCIPgetParams(self._scip)
        cdef int i

        result = {}
        for i in range(SCIPgetNParams(self._scip)):
          name = SCIPparamGetName(params[i]).decode('utf-8')
          result[name] = self.getParam(name)

        return result

    def setParams(self, params):
        """
        Sets multiple parameters at once.

        Parameters
        ----------
        params : dict of str to object
            dict mapping parameter names to their values.

        """
        for name, value in params.items():
          self.setParam(name, value)

    def readParams(self, file):
        """
        Read an external parameter file.

        Parameters
        ----------
        file : str
            file to read

        """
        absfile = str_conversion(abspath(file))

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPreadParams(self._scip, absfile))

        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def writeParams(self, filename='param.set', comments=True, onlychanged=True, verbose=True):
        """
        Write parameter settings to an external file.

        Parameters
        ----------
        filename : str, optional
            file to be written (Default value = 'param.set')
        comments : bool, optional
            write parameter descriptions as comments? (Default value = True)
        onlychanged : bool, optional
            write only modified parameters (Default value = True)
        verbose : bool, optional
            indicates whether a success message should be printed

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        str_absfile = abspath(filename)
        absfile = str_conversion(str_absfile)
        PY_SCIP_CALL(SCIPwriteParams(self._scip, absfile, comments, onlychanged))

        if verbose:
            print('wrote parameter settings to file ' + str_absfile)

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def resetParam(self, name):
        """
        Reset parameter setting to its default value

        Parameters
        ----------
        name : str
            parameter to reset

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPresetParam(self._scip, n))

    def resetParams(self):
        """Reset parameter settings to their default values."""
        PY_SCIP_CALL(SCIPresetParams(self._scip))

    def setEmphasis(self, paraemphasis, quiet = True):
        """
        Set emphasis settings

        Parameters
        ----------
        paraemphasis : PY_SCIP_PARAMEMPHASIS
            emphasis to set
        quiet : bool, optional
            hide output? (Default value = True)

        """
        PY_SCIP_CALL(SCIPsetEmphasis(self._scip, paraemphasis, quiet))

    def readProblem(self, filename, extension = None):
        """
        Read a problem instance from an external file.

        Parameters
        ----------
        filename : str
            problem file name
        extension : str or None
            specify file extension/type (Default value = None)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        absfile = str_conversion(abspath(filename))
        if extension is None:
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = str_conversion(extension)
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, extension))

        locale.setlocale(locale.LC_NUMERIC, user_locale)

    # Counting functions

    def count(self):
        """Counts the number of feasible points of problem."""
        PY_SCIP_CALL(SCIPcount(self._scip))

    def getNReaders(self):
        """
        Get number of currently available readers.

        Returns
        -------
        int

        """
        return SCIPgetNReaders(self._scip)

    def getNCountedSols(self):
        """
        Get number of feasible solution.

        Returns
        -------
        int

        """
        cdef SCIP_Bool valid
        cdef SCIP_Longint nsols

        nsols = SCIPgetNCountedSols(self._scip, &valid)
        if not valid:
            print('total number of solutions found is not valid!')
        return nsols

    def getCountedSols(self):
        """
        Get the feasible solutions collected by `count()`.

        Requires `constraints/countsols/collect` to be True and presolving to be off
        before calling `count()`, and a problem without continuous variables.

        Returns
        -------
        list of dict
            Each solution maps the variable names to their values.

        """
        cdef SCIP_VAR** vars
        cdef SCIP_SPARSESOL** sparsesols
        cdef SCIP_Longint* vals
        cdef SCIP_Bool valid
        cdef int nvars
        cdef int nsparsesols
        cdef int i
        cdef int s

        if not self.getParam("constraints/countsols/collect"):
            raise Warning("set constraints/countsols/collect to True before calling count()")
        if self.getParam("presolving/maxrounds") != 0:
            raise Warning("turn presolving off before calling count()")
        if SCIPgetNContVars(self._scip) > 0:
            raise Warning("counted solutions are only available for problems without continuous variables")

        SCIPgetNCountedSols(self._scip, &valid)
        if not valid:
            raise Warning("number of counted solutions does not fit into a long integer, cannot collect them")

        SCIPgetCountedSparseSols(self._scip, &vars, &nvars, &sparsesols, &nsparsesols)
        # vars are transformed, strip the "t_" prefix to get the original names
        names = [bytes(SCIPvarGetName(vars[i])).decode("utf-8")[2:] for i in range(nvars)]

        vals = <SCIP_Longint*> malloc(nvars * sizeof(SCIP_Longint))
        sols = []
        for s in range(nsparsesols):
            SCIPsparseSolGetFirstSol(sparsesols[s], vals, nvars)
            while True:
                sols.append({names[i]: vals[i] for i in range(nvars)})
                if not SCIPsparseSolGetNextSol(sparsesols[s], vals, nvars):
                    break

        free(vals)

        return sols

    def setParamsCountsols(self):
        """Sets SCIP parameters such that a valid counting process is possible."""
        PY_SCIP_CALL(SCIPsetParamsCountsols(self._scip))

    def freeReoptSolve(self):
        """Frees all solution process data and prepares for reoptimization."""

        if not SCIPisReoptEnabled(self._scip):
            raise ValueError("freeReoptSolve requires reoptimization to be enabled. "
                             "Call enableReoptimization() before solving.")

        if self.getStage() not in [SCIP_STAGE_INIT,
                                 SCIP_STAGE_PROBLEM,
                                 SCIP_STAGE_TRANSFORMED,
                                 SCIP_STAGE_PRESOLVING,
                                 SCIP_STAGE_PRESOLVED,
                                 SCIP_STAGE_SOLVING,
                                 SCIP_STAGE_SOLVED]:
            raise Warning("method cannot be called in stage %i." % self.getStage())
        PY_SCIP_CALL(SCIPfreeReoptSolve(self._scip))

    def chgVarBranchPriority(self, Variable var, priority):
        """
        Sets the branch priority of the variable.
        Variables with higher branch priority are always preferred to variables with
        lower priority in selection of branching variable.

        Parameters
        ----------
        var : Variable
            variable to change priority of
        priority : int
            the new priority of the variable (the default branching priority is 0)

        """
        assert isinstance(var, Variable), "The given variable is not a pyvar, but %s" % var.__class__.__name__
        PY_SCIP_CALL(SCIPchgVarBranchPriority(self._scip, var.scip_var, priority))

    def startStrongbranch(self):
        """Start strong branching. Needs to be called before any strong branching. Must also later end strong branching.
        TODO: Propagation option has currently been disabled via Python.
        If propagation is enabled then strong branching is not done on the LP, but on additionally created nodes
        (has some overhead). """

        PY_SCIP_CALL(SCIPstartStrongbranch(self._scip, False))

    def endStrongbranch(self):
        """End strong branching. Needs to be called if startStrongBranching was called previously.
        Between these calls the user can access all strong branching functionality."""

        PY_SCIP_CALL(SCIPendStrongbranch(self._scip))

    def getVarStrongbranchNode(self, Variable var):
        """
        Get the node number from the last time strong branching was called on the variable.

        Parameters
        ----------
        var : Variable
            variable to get the previous strong branching node from

        Returns
        -------
        int

        """

        cdef SCIP_Longint node_num
        node_num = SCIPgetVarStrongbranchNode(self._scip, var.scip_var)

        return node_num

    def updateVarPseudocost(self, Variable var, valdelta, objdelta, weight):
        """
        Updates the pseudo costs of the given variable and the global pseudo costs after a change of valdelta
        in the variable's solution value and resulting change of objdelta in the LP's objective value.
        Update is ignored if objdelts is infinite. Weight is in range (0, 1], and affects how it updates
        the global weighted sum.

        Parameters
        ----------
        var : Variable
            Variable whos pseudo cost will be updated
        valdelta : float
            The change in variable value (e.g. the fractional amount removed or added by branching)
        objdelta : float
            The change in objective value of the LP after valdelta change of the variable
        weight : float
            the weight in range (0,1] of how the update affects the stored weighted sum.

        """

        PY_SCIP_CALL(SCIPupdateVarPseudocost(self._scip, var.scip_var, valdelta, objdelta, weight))

    def getBranchScoreMultiple(self, Variable var, gains):
        """
        Calculates the branching score out of the gain predictions for a branching with
        arbitrarily many children.

        Parameters
        ----------
        var : Variable
            variable to calculate the score for
        gains : list of float
            list of gains for each child.

        Returns
        -------
        float

        """
        assert isinstance(gains, list)
        cdef int nchildren = len(gains)
        cdef SCIP_Real* _gains = <SCIP_Real*> malloc(nchildren * sizeof(SCIP_Real))
        cdef int i

        for i in range(nchildren):
            _gains[i] = gains[i]

        score = SCIPgetBranchScoreMultiple(self._scip, var.scip_var, nchildren, _gains)

        free(_gains)

        return score

    def getTreesizeEstimation(self):
        """
        Get an estimate of the final tree size.

        Returns
        -------
        float

        """
        return SCIPgetTreesizeEstimation(self._scip)


@dataclass
class Statistics:
    """
    Attributes
    ----------
    status: str
        Status of the problem (optimal solution found, infeasible, etc.)
    total_time : float
        Total time since model was created
    solving_time: float
        Time spent solving the problem
    presolving_time: float
        Time spent on presolving
    reading_time: float
        Time spent on reading
    copying_time: float
        Time spent on copying
    problem_name: str
        Name of problem
    presolved_problem_name: str
        Name of presolved problem
    n_nodes: int
        The number of nodes explored in the branch-and-bound tree
    n_solutions_found: int
        number of found solutions
    first_solution: float
        objective value of first found solution
    primal_bound: float
        The best primal bound found
    dual_bound: float
        The best dual bound found
    gap: float
        The gap between the primal and dual bounds
    primal_dual_integral: float
        The primal-dual integral
    n_vars: int
        number of variables in the model
    n_binary_vars: int
        number of binary variables in the model
    n_integer_vars: int
        number of integer variables in the model
    n_implicit_integer_vars: int
        number of implicit integer variables in the model
    n_continuous_vars: int
        number of continuous variables in the model
    n_presolved_vars: int
        number of variables in the presolved model
    n_presolved_continuous_vars: int
        number of continuous variables in the presolved model
    n_presolved_binary_vars: int
        number of binary variables in the presolved model
    n_presolved_integer_vars: int
        number of integer variables in the presolved model
    n_presolved_implicit_integer_vars: int
        number of implicit integer variables in the presolved model
    n_maximal_cons: int
        number of maximal constraints in the model
    n_initial_cons: int
        number of initial constraints in the presolved model
    n_presolved_maximal_cons: int
        number of maximal constraints in the presolved model
    n_presolved_conss: int
        number of initial constraints in the model
    """

    status: str
    total_time: float
    solving_time: float
    presolving_time: float
    reading_time: float
    copying_time: float
    problem_name: str
    presolved_problem_name: str
    _variables: dict             # Dictionary with number of variables by type
    _presolved_variables: dict   # Dictionary with number of presolved variables by type
    _constraints: dict           # Dictionary with number of constraints by type
    _presolved_constraints: dict # Dictionary with number of presolved constraints by type
    n_runs: int = None
    n_nodes: int = None
    n_solutions_found: int = -1
    first_solution: float = None
    primal_bound: float = None
    dual_bound: float = None
    gap: float = None
    primal_dual_integral: float = None

    # unpacking the _variables, _presolved_variables, _constraints
    # _presolved_constraints dictionaries
    @property
    def n_vars(self):
        return self._variables["total"]

    @property
    def n_binary_vars(self):
        return self._variables["binary"]

    @property
    def n_integer_vars(self):
        return self._variables["integer"]

    @property
    def n_implicit_integer_vars(self):
        return self._variables["implicit"]

    @property
    def n_continuous_vars(self):
        return self._variables["continuous"]

    @property
    def n_presolved_vars(self):
        return self._presolved_variables["total"]

    @property
    def n_presolved_binary_vars(self):
        return self._presolved_variables["binary"]

    @property
    def n_presolved_integer_vars(self):
        return self._presolved_variables["integer"]

    @property
    def n_presolved_implicit_integer_vars(self):
        return self._presolved_variables["implicit"]

    @property
    def n_presolved_continuous_vars(self):
        return self._presolved_variables["continuous"]

    @property
    def n_conss(self):
        return self._constraints["initial"]

    @property
    def n_maximal_cons(self):
        return self._constraints["maximal"]

    @property
    def n_presolved_conss(self):
        return self._presolved_constraints["initial"]

    @property
    def n_presolved_maximal_cons(self):
        return self._presolved_constraints["maximal"]

def readStatistics(filename):
        """
        Given a .stats file of a solved model, reads it and returns an instance of the Statistics class
        holding some statistics.

        Parameters
        ----------
        filename : str
            name of the input file

        Returns
        -------
        Statistics

        """
        cdef int i

        result = {}
        file = open(filename)
        data = file.readlines()

        if "optimal solution found" in data[0]:
            result["status"] = "optimal"
        elif "infeasible" in data[0]:
            result["status"] = "infeasible"
        elif "unbounded" in data[0]:
            result["status"] = "unbounded"
        elif "limit reached" in data[0]:
            result["status"] = "user_interrupt"
        else:
            raise "readStatistics can only be called if the problem was solved"

        available_stats = ["Total Time", "solving", "presolving", "reading", "copying",
                "Problem name", "Variables", "Constraints", "number of runs",
                "nodes", "Solutions found"]

        if result["status"] in ["optimal", "user_interrupt"]:
            available_stats.extend(["First Solution", "Primal Bound", "Dual Bound", "Gap", "primal-dual"])

        seen_cons = 0
        for i, line in enumerate(data):
            if line == "\n":
                continue

            split_line = line.split(":")
            split_line[1] = split_line[1][:-1] # removing \n
            stat_name = split_line[0].strip()

            if seen_cons == 2 and stat_name == "Constraints":
                continue

            if stat_name in available_stats:
                relevant_value = split_line[1].strip()

                if stat_name == "Variables":
                    relevant_value = relevant_value[:-1] # removing ")"
                    var_stats = {}
                    split_var = relevant_value.split("(")
                    var_stats["total"] = int(split_var[0])
                    split_var = split_var[1].split(",")

                    for var_type in split_var:
                        split_result = var_type.strip().split(" ")
                        var_stats[split_result[1]] = int(split_result[0])

                    if "Original" in data[i-2]:
                        result["Variables"] = var_stats
                    else:
                        result["Presolved Variables"] = var_stats

                    continue

                if stat_name == "Constraints":
                    seen_cons += 1
                    con_stats = {}
                    split_con = relevant_value.split(",")
                    for con_type in split_con:
                        split_result = con_type.strip().split(" ")
                        con_stats[split_result[1]] = int(split_result[0])

                    if "Original" in data[i-3]:
                        result["Constraints"] = con_stats
                    else:
                        result["Presolved Constraints"] = con_stats
                    continue

                relevant_value = relevant_value.split(" ")[0]
                if stat_name == "Problem name":
                    if "Original" in data[i-1]:
                        result["Problem name"] = relevant_value
                    else:
                        result["Presolved Problem name"] = relevant_value
                    continue

                if stat_name == "Gap":
                    relevant_value = relevant_value[:-1] # removing %

                try:
                    result[stat_name] = float(relevant_value)
                except (ValueError, TypeError):
                    result[stat_name] = relevant_value
                else:
                    if stat_name == "Solutions found" and result[stat_name] == 0:
                        break

        # changing keys to pythonic variable names
        treated_keys = {"status": "status", "Total Time": "total_time", "solving":"solving_time", "presolving":"presolving_time", "reading":"reading_time",
                        "copying":"copying_time", "Problem name": "problem_name", "Presolved Problem name": "presolved_problem_name", "Variables":"_variables",
                        "Presolved Variables":"_presolved_variables", "Constraints": "_constraints", "Presolved Constraints":"_presolved_constraints",
                        "number of runs": "n_runs", "nodes":"n_nodes", "Solutions found": "n_solutions_found"}

        if result["status"] in ["optimal", "user_interrupt"]:
            if result["Solutions found"] > 0:
                treated_keys["First Solution"]  = "first_solution"
                treated_keys["Primal Bound"]    = "primal_bound"
                treated_keys["Dual Bound"]      = "dual_bound"
                treated_keys["Gap"]         = "gap"
                treated_keys["primal-dual"]     = "primal_dual_integral"
        treated_result = dict((treated_keys[key], value) for (key, value) in result.items())

        stats = Statistics(**treated_result)

        return stats

# debugging memory management
def is_memory_freed():
    return BMSgetMemoryUsed() == 0

def print_memory_in_use():
    BMScheckEmptyMemory()

include "model.pxi"
include "exact.pxi"
