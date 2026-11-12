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

include "basemodel.pxi"
include "model.pxi"
include "exact.pxi"
