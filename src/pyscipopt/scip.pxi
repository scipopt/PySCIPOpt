##@file scip.pxi
#@brief holding functions in python that reference the SCIP public functions included in scip.pxd
import weakref
from os.path import abspath
from os.path import splitext
import os
import sys
import warnings
import locale

cimport cython
from cpython cimport Py_INCREF, Py_DECREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
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
include "presol.pxi"
include "pricer.pxi"
include "propagator.pxi"
include "sepa.pxi"
include "reader.pxi"
include "relax.pxi"
include "nodesel.pxi"
include "matrix.pxi"

# recommended SCIP version; major version is required
MAJOR = 9
MINOR = 2
PATCH = 1

# for external user functions use def; for functions used only inside the interface (starting with _) use cdef
# todo: check whether this is currently done like this

if sys.version_info >= (3, 0):
    str_conversion = lambda x:bytes(x,'utf-8')
else:
    str_conversion = lambda x:x

_SCIP_BOUNDTYPE_TO_STRING = {SCIP_BOUNDTYPE_UPPER: '<=',
                             SCIP_BOUNDTYPE_LOWER: '>='}

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
    DISABLED        = SCIP_EVENTTYPE_DISABLED
    VARADDED        = SCIP_EVENTTYPE_VARADDED
    VARDELETED      = SCIP_EVENTTYPE_VARDELETED
    VARFIXED        = SCIP_EVENTTYPE_VARFIXED
    VARUNLOCKED     = SCIP_EVENTTYPE_VARUNLOCKED
    OBJCHANGED      = SCIP_EVENTTYPE_OBJCHANGED
    GLBCHANGED      = SCIP_EVENTTYPE_GLBCHANGED
    GUBCHANGED      = SCIP_EVENTTYPE_GUBCHANGED
    LBTIGHTENED     = SCIP_EVENTTYPE_LBTIGHTENED
    LBRELAXED       = SCIP_EVENTTYPE_LBRELAXED
    UBTIGHTENED     = SCIP_EVENTTYPE_UBTIGHTENED
    UBRELAXED       = SCIP_EVENTTYPE_UBRELAXED
    GHOLEADDED      = SCIP_EVENTTYPE_GHOLEADDED
    GHOLEREMOVED    = SCIP_EVENTTYPE_GHOLEREMOVED
    LHOLEADDED      = SCIP_EVENTTYPE_LHOLEADDED
    LHOLEREMOVED    = SCIP_EVENTTYPE_LHOLEREMOVED
    IMPLADDED       = SCIP_EVENTTYPE_IMPLADDED
    PRESOLVEROUND   = SCIP_EVENTTYPE_PRESOLVEROUND
    NODEFOCUSED     = SCIP_EVENTTYPE_NODEFOCUSED
    NODEFEASIBLE    = SCIP_EVENTTYPE_NODEFEASIBLE
    NODEINFEASIBLE  = SCIP_EVENTTYPE_NODEINFEASIBLE
    NODEBRANCHED    = SCIP_EVENTTYPE_NODEBRANCHED
    NODEDELETE      = SCIP_EVENTTYPE_NODEDELETE
    FIRSTLPSOLVED   = SCIP_EVENTTYPE_FIRSTLPSOLVED
    LPSOLVED        = SCIP_EVENTTYPE_LPSOLVED
    LPEVENT         = SCIP_EVENTTYPE_LPEVENT
    POORSOLFOUND    = SCIP_EVENTTYPE_POORSOLFOUND
    BESTSOLFOUND    = SCIP_EVENTTYPE_BESTSOLFOUND
    ROWADDEDSEPA    = SCIP_EVENTTYPE_ROWADDEDSEPA
    ROWDELETEDSEPA  = SCIP_EVENTTYPE_ROWDELETEDSEPA
    ROWADDEDLP      = SCIP_EVENTTYPE_ROWADDEDLP
    ROWDELETEDLP    = SCIP_EVENTTYPE_ROWDELETEDLP
    ROWCOEFCHANGED  = SCIP_EVENTTYPE_ROWCOEFCHANGED
    ROWCONSTCHANGED = SCIP_EVENTTYPE_ROWCONSTCHANGED
    ROWSIDECHANGED  = SCIP_EVENTTYPE_ROWSIDECHANGED
    SYNC            = SCIP_EVENTTYPE_SYNC
    GBDCHANGED      = SCIP_EVENTTYPE_GBDCHANGED
    LBCHANGED       = SCIP_EVENTTYPE_LBCHANGED
    UBCHANGED       = SCIP_EVENTTYPE_UBCHANGED
    BOUNDTIGHTENED  = SCIP_EVENTTYPE_BOUNDTIGHTENED
    BOUNDRELAXED    = SCIP_EVENTTYPE_BOUNDRELAXED
    BOUNDCHANGED    = SCIP_EVENTTYPE_BOUNDCHANGED
    GHOLECHANGED    = SCIP_EVENTTYPE_GHOLECHANGED
    LHOLECHANGED    = SCIP_EVENTTYPE_LHOLECHANGED
    HOLECHANGED     = SCIP_EVENTTYPE_HOLECHANGED
    DOMCHANGED      = SCIP_EVENTTYPE_DOMCHANGED
    VARCHANGED      = SCIP_EVENTTYPE_VARCHANGED
    VAREVENT        = SCIP_EVENTTYPE_VAREVENT
    NODESOLVED      = SCIP_EVENTTYPE_NODESOLVED
    NODEEVENT       = SCIP_EVENTTYPE_NODEEVENT
    SOLFOUND        = SCIP_EVENTTYPE_SOLFOUND
    SOLEVENT        = SCIP_EVENTTYPE_SOLEVENT
    ROWCHANGED      = SCIP_EVENTTYPE_ROWCHANGED
    ROWEVENT        = SCIP_EVENTTYPE_ROWEVENT

cdef class PY_SCIP_LOCKTYPE:
    MODEL    = SCIP_LOCKTYPE_MODEL
    CONFLICT = SCIP_LOCKTYPE_CONFLICT

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

cdef class Column:
    """Base class holding a pointer to corresponding SCIP_COL."""

    @staticmethod
    cdef create(SCIP_COL* scipcol):
        """
        Main method for creating a Column class. Is used instead of __init__.

        Parameters
        ----------
        scipcol : SCIP_COL*
            A pointer to the SCIP_COL

        Returns
        -------
        col : Column
            The Python representative of the SCIP_COL

        """
        if scipcol == NULL:
            raise Warning("cannot create Column with SCIP_COL* == NULL")
        col = Column()
        col.scip_col = scipcol
        return col

    def getLPPos(self):
        """
        Gets position of column in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
        return SCIPcolGetLPPos(self.scip_col)

    def getBasisStatus(self):
        """
        Gets the basis status of a column in the LP solution

        Returns
        -------
        str
            Possible values are "lower", "basic", "upper", and "zero"

        Raises
        ------
        Exception
            If SCIP returns an unknown basis status

        Notes
        -----
        Returns basis status "zero" for columns not in the current SCIP LP.

        """
        cdef SCIP_BASESTAT stat = SCIPcolGetBasisStatus(self.scip_col)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            return "zero"
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        """
        Returns whether the associated variable is of integral type (binary, integer, implicit integer).

        Returns
        -------
        bool

        """
        return SCIPcolIsIntegral(self.scip_col)

    def getVar(self):
        """
        Gets variable this column represents.

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* var = SCIPcolGetVar(self.scip_col)
        return Variable.create(var)

    def getPrimsol(self):
        """
        Gets the primal LP solution of a column.

        Returns
        -------
        float

        """
        return SCIPcolGetPrimsol(self.scip_col)

    def getLb(self):
        """
        Gets lower bound of column.

        Returns
        -------
        float

        """
        return SCIPcolGetLb(self.scip_col)

    def getUb(self):
        """
        Gets upper bound of column.

        Returns
        -------
        float

        """
        return SCIPcolGetUb(self.scip_col)

    def getObjCoeff(self):
        """
        Gets objective value coefficient of a column.

        Returns
        -------
        float

        """
        return SCIPcolGetObj(self.scip_col)

    def getAge(self):
        """
        Gets the age of the column, i.e., the total number of successive times a column was in the LP
        and was 0.0 in the solution.

        Returns
        -------
        int

        """
        return SCIPcolGetAge(self.scip_col)

    def __hash__(self):
        return hash(<size_t>self.scip_col)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_col == (<Column>other).scip_col)

cdef class Row:
    """Base class holding a pointer to corresponding SCIP_ROW."""

    @staticmethod
    cdef create(SCIP_ROW* sciprow):
        """
        Main method for creating a Row class. Is used instead of __init__.

        Parameters
        ----------
        sciprow : SCIP_ROW*
            A pointer to the SCIP_ROW

        Returns
        -------
        row : Row
            The Python representative of the SCIP_ROW

        """
        if sciprow == NULL:
            raise Warning("cannot create Row with SCIP_ROW* == NULL")
        row = Row()
        row.scip_row = sciprow
        return row

    property name:
        def __get__(self):
            cname = bytes( SCIProwGetName(self.scip_row) )
            return cname.decode('utf-8')

    def getLhs(self):
        """
        Returns the left hand side of row.

        Returns
        -------
        float

        """
        return SCIProwGetLhs(self.scip_row)

    def getRhs(self):
        """
        Returns the right hand side of row.

        Returns
        -------
        float

        """
        return SCIProwGetRhs(self.scip_row)

    def getConstant(self):
        """
        Gets constant shift of row.

        Returns
        -------
        float

        """
        return SCIProwGetConstant(self.scip_row)

    def getDualsol(self):
        """
        Returns the dual solution of row.

        Returns
        -------
        float

        """
        return SCIProwGetDualsol(self.scip_row)

    def getDualfarkas(self):
        """
        Returns the dual Farkas solution of row.

        Returns
        -------
        float

        """
        return SCIProwGetDualfarkas(self.scip_row)

    def getLPPos(self):
        """
        Gets position of row in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
        return SCIProwGetLPPos(self.scip_row)

    def getBasisStatus(self):
        """
        Gets the basis status of a row in the LP solution.

        Returns
        -------
        str
            Possible values are "lower", "basic", and "upper"

        Raises
        ------
        Exception
            If SCIP returns an unknown or "zero" basis status

        Notes
        -----
        Returns basis status "basic" for rows not in the current SCIP LP.

        """
        cdef SCIP_BASESTAT stat = SCIProwGetBasisStatus(self.scip_row)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            # this shouldn't happen!
            raise Exception('SCIP returned base status zero for a row!')
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        """
        Returns TRUE iff the activity of the row (without the row's constant)
        is always integral in a feasible solution.

        Returns
        -------
        bool

        """
        return SCIProwIsIntegral(self.scip_row)

    def isLocal(self):
        """
        Returns TRUE iff the row is only valid locally.

        Returns
        -------
        bool

        """
        return SCIProwIsLocal(self.scip_row)

    def isModifiable(self):
        """
        Returns TRUE iff row is modifiable during node processing (subject to column generation).

        Returns
        -------
        bool

        """
        return SCIProwIsModifiable(self.scip_row)

    def isRemovable(self):
        """
        Returns TRUE iff row is removable from the LP (due to aging or cleanup).

        Returns
        -------
        bool

        """
        return SCIProwIsRemovable(self.scip_row)

    def isInGlobalCutpool(self):
        """
        Return TRUE iff row is a member of the global cut pool.

        Returns
        -------
        bool

        """
        return SCIProwIsInGlobalCutpool(self.scip_row)

    def getOrigintype(self):
        """
        Returns type of origin that created the row.

        Returns
        -------
        PY_SCIP_ROWORIGINTYPE

        """
        return SCIProwGetOrigintype(self.scip_row)

    def getConsOriginConshdlrtype(self):
        """
        Returns type of constraint handler that created the row.

        Returns
        -------
        str

        """
        cdef SCIP_CONS* scip_con = SCIProwGetOriginCons(self.scip_row)
        return bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(scip_con))).decode('UTF-8')

    def getNNonz(self):
        """
        Get number of nonzero entries in row vector.

        Returns
        -------
        int

        """
        return SCIProwGetNNonz(self.scip_row)

    def getNLPNonz(self):
        """
        Get number of nonzero entries in row vector that correspond to columns currently in the SCIP LP.

        Returns
        -------
        int

        """
        return SCIProwGetNLPNonz(self.scip_row)

    def getCols(self):
        """
        Gets list with columns of nonzero entries

        Returns
        -------
        list of Column

        """
        cdef SCIP_COL** cols = SCIProwGetCols(self.scip_row)
        cdef int i
        return [Column.create(cols[i]) for i in range(self.getNNonz())]

    def getVals(self):
        """
        Gets list with coefficients of nonzero entries.

        Returns
        -------
        list of int

        """
        cdef SCIP_Real* vals = SCIProwGetVals(self.scip_row)
        cdef int i
        return [vals[i] for i in range(self.getNNonz())]

    def getAge(self):
        """
        Gets the age of the row. (The consecutive times the row has been non-active in the LP).

        Returns
        -------
        int

        """
        return SCIProwGetAge(self.scip_row)

    def getNorm(self):
        """
        Gets Euclidean norm of row vector.

        Returns
        -------
        float

        """
        return SCIProwGetNorm(self.scip_row)

    def __hash__(self):
        return hash(<size_t>self.scip_row)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_row == (<Row>other).scip_row)

cdef class NLRow:
    """Base class holding a pointer to corresponding SCIP_NLROW."""

    @staticmethod
    cdef create(SCIP_NLROW* scipnlrow):
        """
        Main method for creating a NLRow class. Is used instead of __init__.

        Parameters
        ----------
        scipnlrow : SCIP_NLROW*
            A pointer to the SCIP_NLROW

        Returns
        -------
        nlrow : NLRow
            The Python representative of the SCIP_NLROW

        """
        if scipnlrow == NULL:
            raise Warning("cannot create NLRow with SCIP_NLROW* == NULL")
        nlrow = NLRow()
        nlrow.scip_nlrow = scipnlrow
        return nlrow

    property name:
        def __get__(self):
            cname = bytes( SCIPnlrowGetName(self.scip_nlrow) )
            return cname.decode('utf-8')

    def getConstant(self):
        """
        Returns the constant of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetConstant(self.scip_nlrow)

    def getLinearTerms(self):
        """
        Returns a list of tuples (var, coef) representing the linear part of a nonlinear row.

        Returns
        -------
        list of tuple

        """
        cdef SCIP_VAR** linvars = SCIPnlrowGetLinearVars(self.scip_nlrow)
        cdef SCIP_Real* lincoefs = SCIPnlrowGetLinearCoefs(self.scip_nlrow)
        cdef int nlinvars = SCIPnlrowGetNLinearVars(self.scip_nlrow)
        cdef int i
        return [(Variable.create(linvars[i]), lincoefs[i]) for i in range(nlinvars)]

    def getLhs(self):
        """
        Returns the left hand side of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetLhs(self.scip_nlrow)

    def getRhs(self):
        """
        Returns the right hand side of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetRhs(self.scip_nlrow)

    def getDualsol(self):
        """
        Gets the dual NLP solution of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetDualsol(self.scip_nlrow)

    def __hash__(self):
        return hash(<size_t>self.scip_nlrow)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_nlrow == (<NLRow>other).scip_nlrow)

cdef class Solution:
    """Base class holding a pointer to corresponding SCIP_SOL."""

    # We are raising an error here to avoid creating a solution without an associated model. See Issue #625
    def __init__(self, raise_error = False):
        if not raise_error:
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
        sol = Solution(True)
        sol.sol = scip_sol
        sol.scip = scip
        return sol

    def __getitem__(self, expr: Union[Expr, MatrixExpr]):
        if isinstance(expr, MatrixExpr):
            result = np.zeros(expr.shape, dtype=np.float64)
            for idx in np.ndindex(expr.shape):
                result[idx] = self.__getitem__(expr[idx])
            return result

        # fast track for Variable
        cdef SCIP_Real coeff
        if isinstance(expr, Variable):
            self._checkStage("SCIPgetSolVal")
            var = <Variable> expr
            return SCIPgetSolVal(self.scip, self.sol, var.scip_var)
        return sum(self._evaluate(term)*coeff for term, coeff in expr.terms.items() if coeff != 0)

    def _evaluate(self, term):
        self._checkStage("SCIPgetSolVal")
        result = 1
        for var in term.vartuple:
            result *= SCIPgetSolVal(self.scip, self.sol, (<Variable> var).scip_var)
        return result

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

    def translate(self, Model target):
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
        return "{} {} {}".format(self.getVar(),
                                 _SCIP_BOUNDTYPE_TO_STRING[self.getBoundtype()],
                                 self.getNewBound())

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

    property name:
        def __get__(self):
            cname = bytes( SCIPvarGetName(self.scip_var) )
            return cname.decode('utf-8')

    def ptr(self):
        """ """
        return <size_t>(self.scip_var)

    def __repr__(self):
        return self.name

    def vtype(self):
        """
        Retrieve the variables type (BINARY, INTEGER, IMPLINT or CONTINUOUS)

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
        elif vartype == SCIP_VARTYPE_IMPLINT:
            return "IMPLINT"

    def isOriginal(self):
        """
        Retrieve whether the variable belongs to the original problem

        Returns
        -------
        bool

        """
        return SCIPvarIsOriginal(self.scip_var)

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
            mayround[idx] = self[idx].varMayRound()
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

    property name:
        def __get__(self):
            cname = bytes( SCIPconsGetName(self.scip_cons) )
            return cname.decode('utf-8')

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
    
    def isCumulative(self):
        """
        Returns True if constraint is cumulative

        Returns
        -------
        bool

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.scip_cons))).decode('UTF-8')
        return constype == 'cumulative'

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

# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
##
#@anchor Model
##
cdef class Model:

    def __init__(self, problemName='model', defaultPlugins=True, Model sourceModel=None, origcopy=False, globalcopy=True, enablepricing=False, createscip=True, threadsafe=False):
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
            whether to enable pricing in copy (default False)
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
        self._generated_event_handlers_count = 0

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


    def __dealloc__(self):
        # call C function directly, because we can no longer call this object's methods, according to
        # http://docs.cython.org/src/reference/extension_types.html#finalization-dealloc
        if self._scip is not NULL and self._freescip and PY_SCIP_CALL:
           PY_SCIP_CALL( SCIPfree(&self._scip) )

    def __hash__(self):
        return hash(<size_t>self._scip)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self._scip == (<Model>other)._scip)

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
        return model

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
        model = Model.create(<SCIP*>PyCapsule_GetPointer(capsule, "scip"))
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

        self._modelvars = {
            var: value
            for var, value in self._modelvars.items()
            if value.isOriginal()
        }
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

    def getNLPIterations(self):
        """
        Returns the total number of LP iterations so far.

        Returns
        -------
        int

        """
        return SCIPgetNLPIterations(self._scip)

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

    def getNLeaves(self):
        """
        Gets number of leaves in the tree.

        Returns
        -------
        int

        """
        return SCIPgetNLeaves(self._scip)

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

    def getCurrentNode(self):
        """
        Retrieve current node.

        Returns
        -------
        Node

        """
        return Node.create(SCIPgetCurrentNode(self._scip))

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

    def cutoffNode(self, Node node):
        """
        marks node and whole subtree to be cut off from the branch and bound tree.

        Parameters
        ----------
        node : Node
        """
        PY_SCIP_CALL( SCIPcutoffNode(self._scip, node.scip_node) )

    def infinity(self):
        """
        Retrieve SCIP's infinity value.

        Returns
        -------
        int

        """
        return SCIPinfinity(self._scip)

    def epsilon(self):
        """
        Retrieve epsilon for e.g. equality checks.

        Returns
        -------
        float

        """
        return SCIPepsilon(self._scip)

    def feastol(self):
        """
        Retrieve feasibility tolerance.

        Returns
        -------
        float

        """
        return SCIPfeastol(self._scip)

    def feasFrac(self, value):
        """
        Returns fractional part of value, i.e. x - floor(x) in feasible tolerance: x - floor(x+feastol).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasFrac(self._scip, value)

    def frac(self, value):
        """
        Returns fractional part of value, i.e. x - floor(x) in epsilon tolerance: x - floor(x+eps).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfrac(self._scip, value)

    def feasFloor(self, value):
        """
        Rounds value + feasibility tolerance down to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasFloor(self._scip, value)

    def feasCeil(self, value):
        """
        Rounds value - feasibility tolerance up to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasCeil(self._scip, value)

    def feasRound(self, value):
        """
        Rounds value to the nearest integer in feasibility tolerance.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasRound(self._scip, value)

    def isZero(self, value):
        """
        Returns whether abs(value) < eps.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisZero(self._scip, value)

    def isFeasZero(self, value):
        """
        Returns whether abs(value) < feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasZero(self._scip, value)

    def isInfinity(self, value):
        """
        Returns whether value is SCIP's infinity.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisInfinity(self._scip, value)

    def isFeasNegative(self, value):
        """
        Returns whether value < -feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasNegative(self._scip, value)

    def isFeasPositive(self, value):
        """
        Returns whether value > feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasPositive(self._scip, value)

    def isFeasIntegral(self, value):
        """
        Returns whether value is integral within the LP feasibility bounds.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasIntegral(self._scip, value)

    def isEQ(self, val1, val2):
        """
        Checks, if values are in range of epsilon.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisEQ(self._scip, val1, val2)

    def isFeasEQ(self, val1, val2):
        """
        Returns if relative difference between val1 and val2 is in range of feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasEQ(self._scip, val1, val2)
    
    def isFeasLT(self, val1, val2):
        """
        Returns whether relative difference between val1 and val2 is lower than minus feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasLT(self._scip, val1, val2)
    
    def isFeasLE(self, val1, val2):
        """
        Returns whether relative difference between val1 and val2 is not greater than feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasLE(self._scip, val1, val2)
    
    def isFeasGT(self, val1, val2):
        """
        Returns whether relative difference between val1 and val2 is greater than feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasGT(self._scip, val1, val2)
    
    def isFeasGE(self, val1, val2):
        """
        Returns whether relative difference of val1 and val2 is not lower than minus feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasGE(self._scip, val1, val2)

    def isLE(self, val1, val2):
        """
        Returns whether val1 <= val2 + eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisLE(self._scip, val1, val2)

    def isLT(self, val1, val2):
        """
        Returns whether val1 < val2 - eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisLT(self._scip, val1, val2)

    def isGE(self, val1, val2):
        """
        Returns whether val1 >= val2 - eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisGE(self._scip, val1, val2)

    def isGT(self, val1, val2):
        """
        Returns whether val1 > val2 + eps.

        Parameters
        ----------
        val1 : float
        val2 : foat

        Returns
        -------
        bool

        """
        return SCIPisGT(self._scip, val1, val2)

    def isHugeValue(self, val):
        """
        Checks if value is huge and should be
        handled separately (e.g., in activity computation).

        Parameters
        ----------
        val : float

        Returns
        -------
        bool

        """
        return SCIPisHugeValue(self._scip, val)

    def isPositive(self, val):
        """
        Returns whether val > eps.

        Parameters
        ----------
        val : float

        Returns
        -------
        bool

        """
        return SCIPisPositive(self._scip, val)

    def isNegative(self, val):
        """
        Returns whether val < -eps.

        Parameters
        ----------
        val : float

        Returns
        -------
        bool

        """
        return SCIPisNegative(self._scip, val)

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

    def setObjlimit(self, objlimit):
        """
        Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Parameters
        ----------
        objlimit : float
            limit on the objective function

        """
        PY_SCIP_CALL(SCIPsetObjlimit(self._scip, objlimit))

    def getObjlimit(self):
        """
        Returns current limit on objective function.

        Returns
        -------
        float

        """
        return SCIPgetObjlimit(self._scip)

    def setObjective(self, expr, sense = 'minimize', clear = 'true'):
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        expr : Expr or float
            the objective function SCIP Expr, or constant value
        sense : str, optional
            the objective sense ("minimize" or "maximize") (Default value = 'minimize')
        clear : bool, optional
            set all other variables objective coefficient to zero (Default value = 'true')

        """
        cdef SCIP_VAR** vars
        cdef int nvars
        cdef SCIP_Real coef
        cdef int i

        # turn the constant value into an Expr instance for further processing
        if not isinstance(expr, Expr):
            assert(_is_number(expr)), "given coefficients are neither Expr or number but %s" % expr.__class__.__name__
            expr = Expr() + expr

        if expr.degree() > 1:
            raise ValueError("SCIP does not support nonlinear objective functions. Consider using set_nonlinear_objective in the pyscipopt.recipe.nonlinear")

        if clear:
            # clear existing objective function
            self.addObjoffset(-self.getObjoffset())
            vars = SCIPgetOrigVars(self._scip)
            nvars = SCIPgetNOrigVars(self._scip)
            for i in range(nvars):
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, vars[i], 0.0))

        if expr[CONST] != 0.0:
            self.addObjoffset(expr[CONST])

        for term, coef in expr.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, var.scip_var, coef))

        if sense == "minimize":
            self.setMinimize()
        elif sense == "maximize":
            self.setMaximize()
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

    def getObjective(self):
        """
        Retrieve objective function as Expr.

        Returns
        -------
        Expr

        """
        variables = self.getVars()
        objective = Expr()
        for var in variables:
            coeff = var.getObj()
            if coeff != 0:
                objective += coeff * var
        objective.normalize()
        return objective

    def addObjoffset(self, offset, solutions = False):
        """
        Add constant offset to objective.

        Parameters
        ----------
        offset : float
            offset to add
        solutions : bool, optional
            add offset also to existing solutions (Default value = False)

        """
        if solutions:
            PY_SCIP_CALL(SCIPaddObjoffset(self._scip, offset))
        else:
            PY_SCIP_CALL(SCIPaddOrigObjoffset(self._scip, offset))

    def getObjoffset(self, original = True):
        """
        Retrieve constant objective offset

        Parameters
        ----------
        original : bool, optional
            offset of original or transformed problem (Default value = True)

        Returns
        -------
        float

        """
        if original:
            return SCIPgetOrigObjoffset(self._scip)
        else:
            return SCIPgetTransObjoffset(self._scip)

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

    def getLocalEstimate(self, original = False):
        """
        Gets estimate of best primal solution w.r.t. original or transformed problem contained in current subtree.

        Parameters
        ----------
        original : bool, optional
            get estimate of original or transformed problem (Default value = False)

        Returns
        -------
        float

        """
        if original:
            return SCIPgetLocalOrigEstimate(self._scip)
        else:
            return SCIPgetLocalTransEstimate(self._scip)

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

    # Variable Functions

    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, pricedVar=False, pricedVarScore=1.0):
        """
        Create a new variable. Default variable is non-negative and continuous.

        Parameters
        ----------
        name : str, optional
            name of the variable, generic if empty (Default value = '')
        vtype : str, optional
            type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
            (Default value = 'C')
        lb : float or None, optional
            lower bound of the variable, use None for -infinity (Default value = 0.0)
        ub : float or None, optional
            upper bound of the variable, use None for +infinity (Default value = None)
        obj : float, optional
            objective value of variable (Default value = 0.0)
        pricedVar : bool, optional
            is the variable a pricing candidate? (Default value = False)
        pricedVarScore : float, optional
            score of variable in case it is priced, the higher the better (Default value = 1.0)

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* scip_var

        # replace empty name with generic one
        if name == '':
            name = 'x'+str(SCIPgetNVars(self._scip)+1)
        cname = str_conversion(name)

        # replace None with corresponding infinity
        if lb is None:
            lb = -SCIPinfinity(self._scip)
        if ub is None:
            ub = SCIPinfinity(self._scip)

        vtype = vtype.upper()
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS))
        elif vtype in ['B', 'BINARY']:
            if ub > 1.0:
                ub = 1.0
            if lb < 0.0:
                lb = 0.0
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_BINARY))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_INTEGER))
        elif vtype in ['M', 'IMPLINT']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_IMPLINT))
        else:
            raise Warning("unrecognized variable type")

        if pricedVar:
            PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, pricedVarScore))
        else:
            PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

        pyVar = Variable.create(scip_var)

        # store variable in the model to avoid creating new python variable objects in getVars()
        assert not pyVar.ptr() in self._modelvars
        self._modelvars[pyVar.ptr()] = pyVar

        #setting the variable data
        SCIPvarSetData(scip_var, <SCIP_VARDATA*>pyVar)
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &scip_var))
        return pyVar

    def addMatrixVar(self,
                     shape: Union[int, Tuple],
                     name: Union[str, np.ndarray] = '',
                     vtype: Union[str, np.ndarray] = 'C',
                     lb: Union[int, float, np.ndarray, None] = 0.0,
                     ub: Union[int, float, np.ndarray, None] = None,
                     obj: Union[int, float, np.ndarray] = 0.0,
                     pricedVar: Union[bool, np.ndarray] = False,
                     pricedVarScore: Union[int, float, np.ndarray] = 1.0
                     ) -> MatrixVariable:
        """
        Create a new matrix of variable. Default matrix variables are non-negative and continuous.

        Parameters
        ----------
        shape : int or tuple
            the shape of the resultant MatrixVariable
        name : str or np.ndarray, optional
            name of the matrix variable, generic if empty (Default value = '')
        vtype : str or np.ndarray, optional
            type of the matrix variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
            (Default value = 'C')
        lb : float or np.ndarray or None, optional
            lower bound of the matrix variable, use None for -infinity (Default value = 0.0)
        ub : float or np.ndarray or None, optional
            upper bound of the matrix variable, use None for +infinity (Default value = None)
        obj : float or np.ndarray, optional
            objective value of matrix variable (Default value = 0.0)
        pricedVar : bool or np.ndarray, optional
            is the matrix variable a pricing candidate? (Default value = False)
        pricedVarScore : float or np.ndarray, optional
            score of matrix variable in case it is priced, the higher the better (Default value = 1.0)

        Returns
        -------
        MatrixVariable

        """
        # assert has_numpy, "Numpy is not installed. Please install numpy to use matrix variables."

        if isinstance(name, np.ndarray):
            assert name.shape == shape
        if isinstance(vtype, np.ndarray):
            assert vtype.shape == shape
        if isinstance(lb, np.ndarray):
            assert lb.shape == shape
        if isinstance(ub, np.ndarray):
            assert ub.shape == shape
        if isinstance(obj, np.ndarray):
            assert obj.shape == shape
        if isinstance(pricedVar, np.ndarray):
            assert pricedVar.shape == shape
        if isinstance(pricedVarScore, np.ndarray):
            assert pricedVarScore.shape == shape

        if isinstance(shape, int):
            ndim = 1
        else:
            ndim = len(shape)

        matrix_variable = np.empty(shape, dtype=object)

        if isinstance(name, str):
            matrix_names = np.full(shape, name, dtype=object)
            if name != "":
                for idx in np.ndindex(matrix_variable.shape):
                    matrix_names[idx] = f"{name}_{'_'.join(map(str, idx))}"
        else:
            matrix_names = name

        if not isinstance(vtype, np.ndarray):
            matrix_vtypes = np.full(shape, vtype, dtype=str)
        else:
            matrix_vtypes = vtype

        if not isinstance(lb, np.ndarray):
            matrix_lbs = np.full(shape, lb, dtype=object)
        else:
            matrix_lbs = lb

        if not isinstance(ub, np.ndarray):
            matrix_ubs = np.full(shape, ub, dtype=object)
        else:
            matrix_ubs = ub

        if not isinstance(obj, np.ndarray):
            matrix_objs = np.full(shape, obj, dtype=float)
        else:
            matrix_objs = obj

        if not isinstance(pricedVar, np.ndarray):
            matrix_priced_vars = np.full(shape, pricedVar, dtype=bool)
        else:
            matrix_priced_vars = pricedVar

        if not isinstance(pricedVarScore, np.ndarray):
            matrix_priced_var_scores = np.full(shape, pricedVarScore, dtype=float)
        else:
            matrix_priced_var_scores = pricedVarScore

        for idx in np.ndindex(matrix_variable.shape):
            matrix_variable[idx] = self.addVar(name=matrix_names[idx], vtype=matrix_vtypes[idx], lb=matrix_lbs[idx],
                                               ub=matrix_ubs[idx], obj=matrix_objs[idx], pricedVar=matrix_priced_vars[idx],
                                               pricedVarScore=matrix_priced_var_scores[idx])

        return matrix_variable.view(MatrixVariable)

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

        return Variable.create(_tvar)

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

    def fixVar(self, Variable var, val):
        """
        Fixes the variable var to the value val if possible.

        Parameters
        ----------
        var : Variable
            variable to fix
        val : float
            the fix value

        Returns
        -------
        infeasible : bool
            Is the fixing infeasible?
        fixed : bool
            Was the fixing performed?

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool fixed
        PY_SCIP_CALL(SCIPfixVar(self._scip, var.scip_var, val, &infeasible, &fixed))
        return infeasible, fixed

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
        if var.ptr() in self._modelvars:
            del self._modelvars[var.ptr()]
        PY_SCIP_CALL(SCIPdelVar(self._scip, var.scip_var, &deleted))
        return deleted

    def tightenVarLb(self, Variable var, lb, force=False):
        """
        Tighten the lower bound in preprocessing or current node, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        lb : float
            possible new lower bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance (default = False)

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarLb(self._scip, var.scip_var, lb, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarUb(self, Variable var, ub, force=False):
        """
        Tighten the upper bound in preprocessing or current node, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        ub : float
            possible new upper bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarUb(self._scip, var.scip_var, ub, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarUbGlobal(self, Variable var, ub, force=False):
        """
        Tighten the global upper bound, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        ub : float
            possible new upper bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarUbGlobal(self._scip, var.scip_var, ub, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarLbGlobal(self, Variable var, lb, force=False):
        """Tighten the global lower bound, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        lb : float
            possible new lower bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarLbGlobal(self._scip, var.scip_var, lb, force, &infeasible, &tightened))
        return infeasible, tightened

    def chgVarLb(self, Variable var, lb):
        """
        Changes the lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLb(self._scip, var.scip_var, lb))

    def chgVarUb(self, Variable var, ub):
        """Changes the upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUb(self._scip, var.scip_var, ub))

    def chgVarLbGlobal(self, Variable var, lb):
        """Changes the global lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbGlobal(self._scip, var.scip_var, lb))

    def chgVarUbGlobal(self, Variable var, ub):
        """Changes the global upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbGlobal(self._scip, var.scip_var, ub))

    def chgVarLbNode(self, Node node, Variable var, lb):
        """Changes the lower bound of the specified variable at the given node.

        Parameters
        ----------
        node : Node
            Node at which the variable bound will be changed
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """

        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbNode(self._scip, node.scip_node, var.scip_var, lb))

    def chgVarUbNode(self, Node node, Variable var, ub):
        """Changes the upper bound of the specified variable at the given node.

        Parameters
        ----------
        node : Node
            Node at which the variable bound will be changed
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbNode(self._scip, node.scip_node, var.scip_var, ub))


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
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.scip_var, SCIP_VARTYPE_IMPLINT, &infeasible))
        else:
            raise Warning("unrecognized variable type")
        if infeasible:
            print('could not change variable type of variable %s' % var)

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
            ptr = <size_t>(_vars[i])

            # check whether the corresponding variable exists already
            if ptr in self._modelvars:
                vars.append(self._modelvars[ptr])
            else:
                # create a new variable
                var = Variable.create(_vars[i])
                assert var.ptr() == ptr
                self._modelvars[ptr] = var
                vars.append(var)

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

    def getVarDict(self, transformed=False):
        """
        Gets dictionary with variables names as keys and current variable values as items.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = False)

        Returns
        -------
        dict of str to float

        """
        var_dict = {}
        for var in self.getVars(transformed=transformed):
            var_dict[var.name] = self.getVal(var)
        return var_dict

    def updateNodeLowerbound(self, Node node, lb):
        """
        If given value is larger than the node's lower bound (in transformed problem),
        sets the node's lower bound to the new value.

        Parameters
        ----------
        node : Node
            the node to update
        lb : float
            new bound (if greater) for the node

        """
        PY_SCIP_CALL(SCIPupdateNodeLowerbound(self._scip, node.scip_node, lb))

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


    # LP Methods
    def getLPSolstat(self):
        """
        Gets solution status of current LP.

        Returns
        -------
        SCIP_LPSOLSTAT

        """
        return SCIPgetLPSolstat(self._scip)


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

    def getLPObjVal(self):
        """
        Gets objective value of current LP (which is the sum of column and loose objective value).

        Returns
        -------
        float

        """

        return SCIPgetLPObjval(self._scip)

    def getLPColsData(self):
        """
        Retrieve current LP columns.

        Returns
        -------
        list of Column

        """
        cdef SCIP_COL** cols
        cdef int ncols
        cdef int i

        PY_SCIP_CALL(SCIPgetLPColsData(self._scip, &cols, &ncols))

        return [Column.create(cols[i]) for i in range(ncols)]

    def getLPRowsData(self):
        """
        Retrieve current LP rows.

        Returns
        -------
        list of Row

        """
        cdef SCIP_ROW** rows
        cdef int nrows
        cdef int i

        PY_SCIP_CALL(SCIPgetLPRowsData(self._scip, &rows, &nrows))

        return [Row.create(rows[i]) for i in range(nrows)]

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

    def getLPBInvRow(self, row):
        """
        Gets a row from the inverse basis matrix B^-1

        Parameters
        ----------
        row : int
            The row index of the inverse basis matrix

        Returns
        -------
        list of float

        """
        # TODO: sparsity information
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBInvRow(self._scip, row, coefs, NULL, NULL))
        result = [coefs[i] for i in range(nrows)]
        free(coefs)

        return result

    def getLPBInvARow(self, row):
        """
        Gets a row from B^-1 * A.

        Parameters
        ----------
        row : int
            The row index of the inverse basis matrix multiplied by the coefficient matrix

        Returns
        -------
        list of float

        """
        # TODO: sparsity information
        cdef int ncols = SCIPgetNLPCols(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBInvARow(self._scip, row, NULL, coefs, NULL, NULL))
        result = [coefs[i] for i in range(ncols)]
        free(coefs)

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

    # LP Col Methods
    def getColRedCost(self, Column col):
        """
        Gets the reduced cost of the column in the current LP.

        Parameters
        ----------
        col : Column

        Returns
        -------
        float

        """
        return SCIPgetColRedcost(self._scip, col.scip_col)

    #TODO: documentation!!
    # LP Row Methods
    def createEmptyRowSepa(self, Sepa sepa, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        """
        Creates and captures an LP row without any coefficients from a separator.

        Parameters
        ----------
        sepa : Sepa
            separator that creates the row
        name : str, optional
            name of row (Default value = "row")
        lhs : float or None, optional
            left hand side of row (Default value = 0)
        rhs : float or None, optional
            right hand side of row (Default value = None)
        local : bool, optional
            is row only valid locally? (Default value = True)
        modifiable : bool, optional
            is row modifiable during node processing (subject to column generation)? (Default value = False)
        removable : bool, optional
            should the row be removed from the LP due to aging or cleanup? (Default value = True)

        Returns
        -------
        Row

        """
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        scip_sepa = SCIPfindSepa(self._scip, str_conversion(sepa.name))
        PY_SCIP_CALL(SCIPcreateEmptyRowSepa(self._scip, &row, scip_sepa, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def createEmptyRowUnspec(self, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        """
        Creates and captures an LP row without any coefficients from an unspecified source.

        Parameters
        ----------
        name : str, optional
            name of row (Default value = "row")
        lhs : float or None, optional
            left hand side of row (Default value = 0)
        rhs : float or None, optional
            right hand side of row (Default value = None)
        local : bool, optional
            is row only valid locally? (Default value = True)
        modifiable : bool, optional
            is row modifiable during node processing (subject to column generation)? (Default value = False)
        removable : bool, optional
            should the row be removed from the LP due to aging or cleanup? (Default value = True)

        Returns
        -------
        Row

        """
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        PY_SCIP_CALL(SCIPcreateEmptyRowUnspec(self._scip, &row, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def getRowActivity(self, Row row):
        """
        Returns the activity of a row in the last LP or pseudo solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIPgetRowActivity(self._scip, row.scip_row)

    def getRowLPActivity(self, Row row):
        """
        Returns the activity of a row in the last LP solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIPgetRowLPActivity(self._scip, row.scip_row)

    # TODO: do we need this? (also do we need release var??)
    def releaseRow(self, Row row not None):
        """
        Decreases usage counter of LP row, and frees memory if necessary.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPreleaseRow(self._scip, &row.scip_row))

    def cacheRowExtensions(self, Row row not None):
        """
        Informs row that all subsequent additions of variables to the row
        should be cached and not directly applied;
        after all additions were applied, flushRowExtensions() must be called;
        while the caching of row extensions is activated, information methods of the
        row give invalid results; caching should be used, if a row is build with addVarToRow()
        calls variable by variable to increase the performance.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPcacheRowExtensions(self._scip, row.scip_row))

    def flushRowExtensions(self, Row row not None):
        """
        Flushes all cached row extensions after a call of cacheRowExtensions()
        and merges coefficients with equal columns into a single coefficient

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPflushRowExtensions(self._scip, row.scip_row))

    def addVarToRow(self, Row row not None, Variable var not None, value):
        """
        Resolves variable to columns and adds them with the coefficient to the row.

        Parameters
        ----------
        row : Row
            Row in which the variable will be added
        var : Variable
            Variable which will be added to the row
        value : float
            Coefficient on the variable when placed in the row

        """
        PY_SCIP_CALL(SCIPaddVarToRow(self._scip, row.scip_row, var.scip_var, value))

    def printRow(self, Row row not None):
        """
        Prints row.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPprintRow(self._scip, row.scip_row, NULL))

    def getRowNumIntCols(self, Row row):
        """
        Returns number of intergal columns in the row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        int

        """
        return SCIPgetRowNumIntCols(self._scip, row.scip_row)

    def getRowObjParallelism(self, Row row):
        """
        Returns 1 if the row is parallel, and 0 if orthogonal.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIPgetRowObjParallelism(self._scip, row.scip_row)

    def getRowParallelism(self, Row row1, Row row2, orthofunc=101):
        """
        Returns the degree of parallelism between hyplerplanes. 1 if perfectly parallel, 0 if orthogonal.
        For two row vectors v, w the parallelism is calculated as: abs(v*w)/(abs(v)*abs(w)).
        101 in this case is an 'e' (euclidean) in ASCII. The other acceptable input is 100 (d for discrete).

        Parameters
        ----------
        row1 : Row
        row2 : Row
        orthofunc : int, optional
            101 (default) is an 'e' (euclidean) in ASCII. Alternate value is 100 (d for discrete)

        Returns
        -------
        float

        """
        return SCIProwGetParallelism(row1.scip_row, row2.scip_row, orthofunc)

    def getRowDualSol(self, Row row):
        """
        Gets the dual LP solution of a row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIProwGetDualsol(row.scip_row)

    # Cutting Plane Methods
    def addPoolCut(self, Row row not None):
        """
        If not already existing, adds row to global cut pool.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPaddPoolCut(self._scip, row.scip_row))

    def getCutEfficacy(self, Row cut not None, Solution sol = None):
        """
        Returns efficacy of the cut with respect to the given primal solution or the
        current LP solution: e = -feasibility/norm

        Parameters
        ----------
        cut : Row
        sol : Solution or None, optional

        Returns
        -------
        float

        """
        return SCIPgetCutEfficacy(self._scip, NULL if sol is None else sol.sol, cut.scip_row)

    def isCutEfficacious(self, Row cut not None, Solution sol = None):
        """
        Returns whether the cut's efficacy with respect to the given primal solution or the
        current LP solution is greater than the minimal cut efficacy.

        Parameters
        ----------
        cut : Row
        sol : Solution or None, optional

        Returns
        -------
        float

        """
        return SCIPisCutEfficacious(self._scip, NULL if sol is None else sol.sol, cut.scip_row)

    def getCutLPSolCutoffDistance(self, Row cut not None, Solution sol not None):
        """
        Returns row's cutoff distance in the direction of the given primal solution.

        Parameters
        ----------
        cut : Row
        sol : Solution

        Returns
        -------
        float

        """
        return SCIPgetCutLPSolCutoffDistance(self._scip, sol.sol, cut.scip_row)

    def addCut(self, Row cut not None, forcecut = False):
        """
        Adds cut to separation storage and returns whether cut has been detected to be infeasible for local bounds.

        Parameters
        ----------
        cut : Row
            The cut that will be added
        forcecut : bool, optional
            Whether the cut should be forced or not, i.e., selected no matter what

        Returns
        -------
        infeasible : bool
            Whether the cut has been detected to be infeasible from local bounds

        """
        cdef SCIP_Bool infeasible
        PY_SCIP_CALL(SCIPaddRow(self._scip, cut.scip_row, forcecut, &infeasible))
        return infeasible

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

    def _createConsLinear(self, ExprCons lincons, **kwargs):
        """
        The function for creating a linear constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        lincons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        assert isinstance(lincons, ExprCons), "given constraint is not ExprCons but %s" % lincons.__class__.__name__

        assert lincons.expr.degree() <= 1, "given constraint is not linear, degree == %d" % lincons.expr.degree()
        terms = lincons.expr.terms

        cdef int nvars = len(terms.items())
        cdef SCIP_VAR** vars_array = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_Real* coeffs_array = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))
        cdef SCIP_CONS* scip_cons
        cdef SCIP_Real coeff
        cdef int i

        for i, (key, coeff) in enumerate(terms.items()):
            vars_array[i] = <SCIP_VAR*>(<Variable>key[0]).scip_var
            coeffs_array[i] = <SCIP_Real>coeff

        PY_SCIP_CALL(SCIPcreateConsLinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']), nvars, vars_array, coeffs_array,
            kwargs['lhs'], kwargs['rhs'], kwargs['initial'],
            kwargs['separate'], kwargs['enforce'], kwargs['check'],
            kwargs['propagate'], kwargs['local'], kwargs['modifiable'],
            kwargs['dynamic'], kwargs['removable'], kwargs['stickingatnode']))

        PyCons = Constraint.create(scip_cons)

        free(vars_array)
        free(coeffs_array)

        return PyCons

    def _createConsQuadratic(self, ExprCons quadcons, **kwargs):
        """
        The function for creating a quadratic constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        quadcons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2, "given constraint is not quadratic, degree == %d" % quadcons.expr.degree()

        cdef SCIP_CONS* scip_cons
        cdef SCIP_EXPR* prodexpr
        PY_SCIP_CALL(SCIPcreateConsQuadraticNonlinear(
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
                PY_SCIP_CALL(SCIPaddLinearVarNonlinear(self._scip, scip_cons, var.scip_var, c))
            else: # nonlinear
                assert len(v) == 2, 'term length must be 1 or 2 but it is %s' % len(v)

                varexprs = <SCIP_EXPR**> malloc(2 * sizeof(SCIP_EXPR*))
                var1, var2 = <Variable>v[0], <Variable>v[1]
                PY_SCIP_CALL( SCIPcreateExprVar(self._scip, &varexprs[0], var1.scip_var, NULL, NULL) )
                PY_SCIP_CALL( SCIPcreateExprVar(self._scip, &varexprs[1], var2.scip_var, NULL, NULL) )
                PY_SCIP_CALL( SCIPcreateExprProduct(self._scip, &prodexpr, 2, varexprs, 1.0, NULL, NULL) )

                PY_SCIP_CALL( SCIPaddExprNonlinear(self._scip, scip_cons, prodexpr, c) )

                PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &prodexpr) )
                PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &varexprs[1]) )
                PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &varexprs[0]) )
                free(varexprs)

        PyCons = Constraint.create(scip_cons)

        return PyCons

    def _createConsNonlinear(self, cons, **kwargs):
        """
        The function for creating a non-linear constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        cons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        cdef SCIP_EXPR* expr
        cdef SCIP_EXPR** varexprs
        cdef SCIP_EXPR** monomials
        cdef SCIP_CONS* scip_cons
        cdef int* idxs
        cdef int i
        cdef int j

        terms = cons.expr.terms

        # collect variables
        variables = {var.ptr(): var for term in terms for var in term}
        variables = list(variables.values())
        varindex = {var.ptr(): i for (i, var) in enumerate(variables)}

        # create monomials for terms
        monomials = <SCIP_EXPR**> malloc(len(terms) * sizeof(SCIP_EXPR*))
        termcoefs = <SCIP_Real*> malloc(len(terms) * sizeof(SCIP_Real))
        for i, (term, coef) in enumerate(terms.items()):
            termvars = <SCIP_VAR**> malloc(len(term) * sizeof(SCIP_VAR*))
            for j, var in enumerate(term):
                termvars[j] = (<Variable>var).scip_var
            PY_SCIP_CALL( SCIPcreateExprMonomial(self._scip, &monomials[i], <int>len(term), termvars, NULL, NULL, NULL) )
            termcoefs[i] = <SCIP_Real>coef
            free(termvars)

        # create polynomial from monomials
        PY_SCIP_CALL( SCIPcreateExprSum(self._scip, &expr, <int>len(terms), monomials, termcoefs, 0.0, NULL, NULL))

        # create nonlinear constraint for expr
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip,
            &scip_cons,
            str_conversion(kwargs['name']),
            expr,
            kwargs['lhs'],
            kwargs['rhs'],
            kwargs['initial'],
            kwargs['separate'],
            kwargs['enforce'],
            kwargs['check'],
            kwargs['propagate'],
            kwargs['local'],
            kwargs['modifiable'],
            kwargs['dynamic'],
            kwargs['removable']) )

        PyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &expr) )
        for i in range(<int>len(terms)):
            PY_SCIP_CALL(SCIPreleaseExpr(self._scip, &monomials[i]))
        free(monomials)
        free(termcoefs)

        return PyCons

    def _createConsGenNonlinear(self, cons, **kwargs):
        """
        The function for creating a general non-linear constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        cons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        cdef SCIP_EXPR** childrenexpr
        cdef SCIP_EXPR** scipexprs
        cdef SCIP_CONS* scip_cons
        cdef int nchildren
        cdef int c
        cdef int i

        # get arrays from python's expression tree
        expr = cons.expr
        nodes = expr_to_nodes(expr)

        # in nodes we have a list of tuples: each tuple is of the form
        # (operator, [indices]) where indices are the indices of the tuples
        # that are the children of this operator. This is sorted,
        # so we are going to do is:
        # loop over the nodes and create the expression of each
        # Note1: when the operator is Operator.const, [indices] stores the value
        # Note2: we need to compute the number of variable operators to find out
        # how many variables are there.
        nvars = 0
        for node in nodes:
            if node[0] == Operator.varidx:
                nvars += 1
        vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))

        varpos = 0
        scipexprs = <SCIP_EXPR**> malloc(len(nodes) * sizeof(SCIP_EXPR*))
        for i,node in enumerate(nodes):
            opidx = node[0]
            if opidx == Operator.varidx:
                assert len(node[1]) == 1
                pyvar = node[1][0] # for vars we store the actual var!
                PY_SCIP_CALL( SCIPcreateExprVar(self._scip, &scipexprs[i], (<Variable>pyvar).scip_var, NULL, NULL) )
                vars[varpos] = (<Variable>pyvar).scip_var
                varpos += 1
                continue
            if opidx == Operator.const:
                assert len(node[1]) == 1
                value = node[1][0]
                PY_SCIP_CALL( SCIPcreateExprValue(self._scip, &scipexprs[i], <SCIP_Real>value, NULL, NULL) )
                continue
            if opidx == Operator.add:
                nchildren = len(node[1])
                childrenexpr = <SCIP_EXPR**> malloc(nchildren * sizeof(SCIP_EXPR*))
                coefs = <SCIP_Real*> malloc(nchildren * sizeof(SCIP_Real))
                for c, pos in enumerate(node[1]):
                    childrenexpr[c] = scipexprs[pos]
                    coefs[c] = 1
                PY_SCIP_CALL( SCIPcreateExprSum(self._scip, &scipexprs[i], nchildren, childrenexpr, coefs, 0, NULL, NULL))
                free(coefs)
                free(childrenexpr)
                continue
            if opidx == Operator.prod:
                nchildren = len(node[1])
                childrenexpr = <SCIP_EXPR**> malloc(nchildren * sizeof(SCIP_EXPR*))
                for c, pos in enumerate(node[1]):
                    childrenexpr[c] = scipexprs[pos]
                PY_SCIP_CALL( SCIPcreateExprProduct(self._scip, &scipexprs[i], nchildren, childrenexpr, 1, NULL, NULL) )
                free(childrenexpr)
                continue
            if opidx == Operator.power:
                # the second child is the exponent which is a const
                valuenode = nodes[node[1][1]]
                assert valuenode[0] == Operator.const
                exponent = valuenode[1][0]
                PY_SCIP_CALL( SCIPcreateExprPow(self._scip, &scipexprs[i], scipexprs[node[1][0]], <SCIP_Real>exponent, NULL, NULL ))
                continue
            if opidx == Operator.exp:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPcreateExprExp(self._scip, &scipexprs[i], scipexprs[node[1][0]], NULL, NULL ))
                continue
            if opidx == Operator.log:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPcreateExprLog(self._scip, &scipexprs[i], scipexprs[node[1][0]], NULL, NULL ))
                continue
            if opidx == Operator.sqrt:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPcreateExprPow(self._scip, &scipexprs[i], scipexprs[node[1][0]], <SCIP_Real>0.5, NULL, NULL) )
                continue
            if opidx == Operator.sin:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPcreateExprSin(self._scip, &scipexprs[i], scipexprs[node[1][0]], NULL, NULL) )
                continue
            if opidx == Operator.cos:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPcreateExprCos(self._scip, &scipexprs[i], scipexprs[node[1][0]], NULL, NULL) )
                continue
            if opidx == Operator.fabs:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPcreateExprAbs(self._scip, &scipexprs[i], scipexprs[node[1][0]], NULL, NULL ))
                continue
            # default:
            raise NotImplementedError
        assert varpos == nvars

        # create nonlinear constraint for the expression root
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip,
            &scip_cons,
            str_conversion(kwargs['name']),
            scipexprs[len(nodes) - 1],
            kwargs['lhs'],
            kwargs['rhs'],
            kwargs['initial'],
            kwargs['separate'],
            kwargs['enforce'],
            kwargs['check'],
            kwargs['propagate'],
            kwargs['local'],
            kwargs['modifiable'],
            kwargs['dynamic'],
            kwargs['removable']) )
        PyCons = Constraint.create(scip_cons)
        for i in range(len(nodes)):
            PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &scipexprs[i]) )

        # free more memory
        free(scipexprs)
        free(vars)

        return PyCons

    def createConsFromExpr(self, cons, name='', initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """
        Create a linear or nonlinear constraint without adding it to the SCIP problem.
        This is useful for creating disjunction constraints without also enforcing the individual constituents.
        Currently, this can only be used as an argument to `.addConsElemDisjunction`. To add
        an individual linear/nonlinear constraint, prefer `.addCons()`.

        Parameters
        ----------
        cons : ExprCons
            The expression constraint that is not yet an actual constraint
        name : str, optional
            the name of the constraint, generic name if empty (Default value = '')
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
            even if it may be  moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The created Constraint object.

        """
        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode
                      )

        kwargs['lhs'] = -SCIPinfinity(self._scip) if cons._lhs is None else cons._lhs
        kwargs['rhs'] =  SCIPinfinity(self._scip) if cons._rhs is None else cons._rhs

        deg = cons.expr.degree()
        if deg <= 1:
            return self._createConsLinear(cons, **kwargs)
        elif deg <= 2:
            return self._createConsQuadratic(cons, **kwargs)
        elif deg == float('inf'): # general nonlinear
            return self._createConsGenNonlinear(cons, **kwargs)
        else:
            return self._createConsNonlinear(cons, **kwargs)

    # Constraint functions
    def addCons(self, cons, name='', initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """
        Add a linear or nonlinear constraint.

        Parameters
        ----------
        cons : ExprCons
            The expression constraint that is not yet an actual constraint
        name : str, optional
            the name of the constraint, generic name if empty (Default value = "")
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
            should the constraints always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The created and added Constraint object.

        """
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        cdef SCIP_CONS* scip_cons

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode
                      )
        #  we have to pass this back to a SCIP_CONS*
        # object to create a new python constraint & handle constraint release
        # correctly. Otherwise, segfaults when trying to query information
        # about the created constraint later.
        pycons_initial = self.createConsFromExpr(cons, **kwargs)
        scip_cons = (<Constraint>pycons_initial).scip_cons

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pycons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pycons

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
                      cons: MatrixExprCons,
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
        cons : MatrixExprCons
            The matrix expression constraint that is not yet an actual constraint
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
        MatrixConstraint
            The created and added MatrixConstraint object.

        """
        assert isinstance(cons, MatrixExprCons), (
                "given constraint is not MatrixExprCons but %s" % cons.__class__.__name__)

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

    def addConsDisjunction(self, conss, name = '', initial = True,
        relaxcons = None, enforce=True, check =True,
        local=False, modifiable = False, dynamic = False):
        """
        Add a disjunction constraint.

        Parameters
        ----------
        conss : iterable of ExprCons
            An iterable of constraint objects to be included initially in the disjunction.
            Currently, these must be expressions.
        name : str, optional
            the name of the disjunction constraint.
        initial : bool, optional
            should the LP relaxation of disjunction constraint be in the initial LP? (Default value = True)
        relaxcons : None, optional
            a conjunction constraint containing the linear relaxation of the disjunction constraint, or None.
            NOT YET SUPPORTED. (Default value = None)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)

        Returns
        -------
        Constraint
            The created disjunction constraint

        """
        cdef SCIP_EXPR* scip_expr
        cdef SCIP_CONS* scip_cons
        cdef SCIP_CONS* disj_cons
        cdef int n_conss
        cdef int i

        def ensure_iterable(elem, length):
            if isinstance(elem, Iterable):
                return elem
            else:
                return list(repeat(elem, length))
        assert isinstance(conss, Iterable), "Given constraint list is not iterable"

        conss = list(conss)
        n_conss = len(conss)

        PY_SCIP_CALL(SCIPcreateConsDisjunction(
            self._scip, &disj_cons, str_conversion(name), 0, &scip_cons, NULL,
            initial, enforce, check, local, modifiable, dynamic
        ))


        # TODO add constraints to disjunction
        for i, cons in enumerate(conss):
            pycons = self.createConsFromExpr(cons, name=name, initial = initial,
                                            enforce=enforce, check=check,
                                            local=local, modifiable=modifiable, dynamic=dynamic
                                            )
            PY_SCIP_CALL(SCIPaddConsElemDisjunction(self._scip,disj_cons, (<Constraint>pycons).scip_cons))
            PY_SCIP_CALL(SCIPreleaseCons(self._scip, &(<Constraint>pycons).scip_cons))
        PY_SCIP_CALL(SCIPaddCons(self._scip, disj_cons))
        PyCons = Constraint.create(disj_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &disj_cons))

        return PyCons

    def addConsElemDisjunction(self, Constraint disj_cons, Constraint cons):
        """
        Appends a constraint to a disjunction.

        Parameters
        ----------
        disj_cons : Constraint
             the disjunction constraint to append to.
        cons : Constraint
            the constraint to append

        Returns
        -------
        disj_cons : Constraint
            The disjunction constraint with `cons` appended.

        """
        PY_SCIP_CALL(SCIPaddConsElemDisjunction(self._scip, (<Constraint>disj_cons).scip_cons, (<Constraint>cons).scip_cons))
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &(<Constraint>cons).scip_cons))
        return disj_cons

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

    def getConsVars(self, Constraint constraint):
        """
        Gets variables in a constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to get the variables from.

        Returns
        -------
        list of Variable

        """
        cdef SCIP_VAR** _vars
        cdef int nvars
        cdef SCIP_Bool success
        cdef int i

        SCIPgetConsNVars(self._scip, constraint.scip_cons, &nvars, &success)
        _vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        SCIPgetConsVars(self._scip, constraint.scip_cons, _vars, nvars*sizeof(SCIP_VAR*), &success)

        vars = []
        for i in range(nvars):
            ptr = <size_t>(_vars[i])
            # check whether the corresponding variable exists already
            if ptr in self._modelvars:
                vars.append(self._modelvars[ptr])
            else:
                # create a new variable
                var = Variable.create(_vars[i])
                assert var.ptr() == ptr
                self._modelvars[ptr] = var
                vars.append(var)

        return vars

    def printCons(self, Constraint constraint):
        """
        Print the constraint

        Parameters
        ----------
        constraint : Constraint

        """
        return PY_SCIP_CALL(SCIPprintCons(self._scip, constraint.scip_cons, NULL))

    def addExprNonlinear(self, Constraint cons, expr, coef):
        """
        Add coef*expr to nonlinear constraint.

        Parameters
        ----------
        cons : Constraint
        expr : Expr or GenExpr
        coef : float

        """
        assert self.getStage() == 1, "addExprNonlinear cannot be called in stage %i." % self.getStage()
        assert cons.isNonlinear(), "addExprNonlinear can only be called with nonlinear constraints."

        cdef Constraint temp_cons
        cdef SCIP_EXPR* scip_expr

        temp_cons = self.addCons(expr <= 0)
        scip_expr = SCIPgetExprNonlinear(temp_cons.scip_cons)

        PY_SCIP_CALL(SCIPaddExprNonlinear(self._scip, cons.scip_cons, scip_expr, coef))
        self.delCons(temp_cons)

    def addConsCoeff(self, Constraint cons, Variable var, coeff):
        """
        Add coefficient to the constraint (if non-zero).

        Parameters
        ----------
        cons : Constraint
            Constraint to be changed
        var : Variable
            variable to be added
        coeff : float
            coefficient of new variable

        """

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, cons.scip_cons, var.scip_var, coeff))
        elif constype == 'knapsack':
            PY_SCIP_CALL(SCIPaddCoefKnapsack(self._scip, cons.scip_cons, var.scip_var, coeff))
        else:
            raise NotImplementedError("Adding coefficients to %s constraints is not implemented." % constype)

    def addConsNode(self, Node node, Constraint cons, Node validnode=None):
        """
        Add a constraint to the given node.

        Parameters
        ----------
        node : Node
            node at which the constraint will be added
        cons : Constraint
            the constraint to add to the node
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)

        """
        if isinstance(validnode, Node):
            PY_SCIP_CALL(SCIPaddConsNode(self._scip, node.scip_node, cons.scip_cons, validnode.scip_node))
        else:
            PY_SCIP_CALL(SCIPaddConsNode(self._scip, node.scip_node, cons.scip_cons, NULL))
        Py_INCREF(cons)

    def addConsLocal(self, Constraint cons, Node validnode=None):
        """
        Add a constraint to the current node.

        Parameters
        ----------
        cons : Constraint
            the constraint to add to the current node
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)

        """
        if isinstance(validnode, Node):
            PY_SCIP_CALL(SCIPaddConsLocal(self._scip, cons.scip_cons, validnode.scip_node))
        else:
            PY_SCIP_CALL(SCIPaddConsLocal(self._scip, cons.scip_cons, NULL))
        Py_INCREF(cons)

    def addConsCumulative(self, vars, durations, demands, capacity, name="",
                          initial=True, separate=True, enforce=True, check=True,
                          propagate=True, local=False, modifiable=False,
                          dynamic=False, removable=False, stickingatnode=False):

        cdef int n = len(vars)
        assert n == len(durations) == len(demands)

        if name == "":
            name = "c" + str(SCIPgetNConss(self._scip) + 1)

        cdef SCIP_VAR** c_vars = <SCIP_VAR**> malloc(n * sizeof(SCIP_VAR*))
        cdef int*       c_durs = <int*>       malloc(n * sizeof(int))
        cdef int*       c_dem  = <int*>       malloc(n * sizeof(int))
        cdef SCIP_CONS* cons
        cdef int i

        for i in range(n):
            c_vars[i] = (<Variable> vars[i]).scip_var
            c_durs[i] = <int> durations[i]
            c_dem[i]  = <int> demands[i]

        # --- Constraint erzeugen
        PY_SCIP_CALL(SCIPcreateConsCumulative(
            self._scip, &cons, str_conversion(name),
            n, c_vars, c_durs, c_dem, <int>capacity,
            initial, separate, enforce, check, propagate,
            local, modifiable, dynamic, removable, stickingatnode))

        # --- Constraint dem Modell hinzufügen  (verhindert spätere Segfaults!)
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons))

        # --- Hilfs‑Arrays freigeben
        free(c_vars)
        free(c_durs)
        free(c_dem)

        # --- Python‑Wrapper erstellen und zurückgeben
        pyCons = Constraint.create(cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &cons))
        return pyCons


    def addConsKnapsack(self, vars, weights, capacity, name="",
                initial=True, separate=True, enforce=True, check=True,
                modifiable=False, propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of int
            list of weights
        capacity: int
            capacity of the knapsack
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created knapsack constraint        
        """

        cdef int nvars = len(vars)
        cdef int i
        cdef SCIP_VAR** vars_array = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_Longint* weights_array = <SCIP_Longint*> malloc(nvars * sizeof(SCIP_Real))
        cdef SCIP_CONS* scip_cons

        assert nvars == len(weights), "Number of variables and weights must be the same."

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        for i in range(nvars):
            vars_array[i] = (<Variable>vars[i]).scip_var
            weights_array[i] = <SCIP_Longint>weights[i]

        PY_SCIP_CALL(SCIPcreateConsKnapsack(
            self._scip, &scip_cons, str_conversion(name), nvars, vars_array, weights_array,
            capacity, initial, separate, enforce, check, propagate, local, modifiable,
            dynamic, removable, stickingatnode))

        free(vars_array)
        free(weights_array)

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))

        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        
        return pyCons

    def addConsSOS1(self, vars, weights=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add an SOS1 constraint.

        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of float or None, optional
            list of weights (Default value = None)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created SOS1 constraint

        """
        cdef SCIP_CONS* scip_cons
        cdef int nvars
        cdef int i

        PY_SCIP_CALL(SCIPcreateConsSOS1(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, scip_cons, var.scip_var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, scip_cons, var.scip_var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))

        return Constraint.create(scip_cons)

    def addConsSOS2(self, vars, weights=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add an SOS2 constraint.

        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of float or None, optional
            list of weights (Default value = None)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created SOS2 constraint

        """
        cdef SCIP_CONS* scip_cons
        cdef int nvars
        cdef int i

        PY_SCIP_CALL(SCIPcreateConsSOS2(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, scip_cons, var.scip_var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, scip_cons, var.scip_var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))

        return Constraint.create(scip_cons)

    def addConsAnd(self, vars, resvar, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add an AND-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables to be included (operators)
        resvar : Variable
            BINARY variable (resultant)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created AND constraint

        """
        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_VAR* _resvar
        cdef SCIP_CONS* scip_cons
        cdef int i

        _resvar = (<Variable>resvar).scip_var
        for i, var in enumerate(vars):
            _vars[i] = (<Variable>var).scip_var

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsAnd(self._scip, &scip_cons, str_conversion(name), _resvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsOr(self, vars, resvar, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add an OR-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables to be included (operators)
        resvar : Variable
            BINARY variable (resultant)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created OR constraint

        """
        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_VAR* _resvar
        cdef SCIP_CONS* scip_cons
        cdef int i

        _resvar = (<Variable>resvar).scip_var
        for i, var in enumerate(vars):
            _vars[i] = (<Variable>var).scip_var

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsOr(self._scip, &scip_cons, str_conversion(name), _resvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsXor(self, vars, rhsvar, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add a XOR-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of binary variables to be included (operators)
        rhsvar : bool
            BOOLEAN value, explicit True, False or bool(obj) is needed (right-hand side)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created XOR constraint

        """
        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_CONS* scip_cons
        cdef int i

        assert type(rhsvar) is type(bool()), "Provide BOOLEAN value as rhsvar, you gave %s." % type(rhsvar)
        for i, var in enumerate(vars):
            _vars[i] = (<Variable>var).scip_var

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsXor(self._scip, &scip_cons, str_conversion(name), rhsvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsCardinality(self, consvars, cardval, indvars=None, weights=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add a cardinality constraint that allows at most 'cardval' many nonzero variables.

        Parameters
        ----------
        consvars : list of Variable
            list of variables to be included
        cardval : int
            nonnegative integer
        indvars : list of Variable or None, optional
            indicator variables indicating which variables may be treated as nonzero in
            cardinality constraint, or None if new indicator variables should be
            introduced automatically (Default value = None)
        weights : list of float or None, optional
            weights determining the variable order, or None if variables should be ordered
            in the same way they were added to the constraint (Default value = None)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created Cardinality constraint

        """
        cdef SCIP_VAR* indvar
        cdef SCIP_CONS* scip_cons
        cdef int i

        PY_SCIP_CALL(SCIPcreateConsCardinality(self._scip, &scip_cons, str_conversion(name), 0, NULL, cardval, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        # circumvent an annoying bug in SCIP 4.0.0 that does not allow uninitialized weights
        if weights is None:
            weights = list(range(1, len(consvars) + 1))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        for i, v in enumerate(consvars):
            var = <Variable>v
            if indvars:
                indvar = (<Variable>indvars[i]).scip_var
            else:
                indvar = NULL
            if weights is None:
                PY_SCIP_CALL(SCIPappendVarCardinality(self._scip, scip_cons, var.scip_var, indvar))
            else:
                PY_SCIP_CALL(SCIPaddVarCardinality(self._scip, scip_cons, var.scip_var, indvar, <SCIP_Real>weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsIndicator(self, cons, binvar=None, activeone=True, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an indicator constraint for the linear inequality `cons`.

        The `binvar` argument models the redundancy of the linear constraint. A solution for which
        `binvar` is 1 must satisfy the constraint.

        Parameters
        ----------
        cons : ExprCons
            a linear inequality of the form "<="
        binvar : Variable, optional
            binary indicator variable, or None if it should be created (Default value = None)
        activeone : bool, optional
            constraint should active if binvar is 1 (0 if activeone = False)
        name : str, optional
            name of the constraint (Default value = "")
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
            The newly created Indicator constraint

        """
        cdef SCIP_VAR* _binVar
        cdef SCIP_CONS* scip_cons
        cdef SCIP_Real coeff
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        if cons._lhs is not None and cons._rhs is not None:
            raise ValueError("expected inequality that has either only a left or right hand side")

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        if cons.expr.degree() > 1:
            raise ValueError("expected linear inequality, expression has degree %d" % cons.expr.degree())

        if cons._rhs is not None:
            rhs =  cons._rhs
            negate = False
        else:
            rhs = -cons._lhs
            negate = True

        if binvar is not None:
            _binVar = (<Variable>binvar).scip_var
            if not activeone:
                PY_SCIP_CALL(SCIPgetNegatedVar(self._scip, _binVar, &_binVar))
        else:
            _binVar = NULL

        PY_SCIP_CALL(SCIPcreateConsIndicator(self._scip, &scip_cons, str_conversion(name), _binVar, 0, NULL, NULL, rhs,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))
        terms = cons.expr.terms

        for key, coeff in terms.items():
            var = <Variable>key[0]
            if negate:
                coeff = -coeff
            PY_SCIP_CALL(SCIPaddVarIndicator(self._scip, scip_cons, var.scip_var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def getLinearConsIndicator(self, Constraint cons):
        """
        Get the linear constraint corresponding to the indicator constraint.

        Parameters
        ----------
        cons : Constraint
            The indicator constraint

        Returns
        -------
        Constraint or None
        """

        cdef SCIP_CONS* lincons = SCIPgetLinearConsIndicator(cons.scip_cons)
        if lincons == NULL:
            return None
        return Constraint.create(lincons)

    def getSlackVarIndicator(self, Constraint cons):
        """
        Get slack variable of an indicator constraint.


        Parameters
        ----------
        cons : Constraint
            The indicator constraint

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* var = SCIPgetSlackVarIndicator(cons.scip_cons)
        return Variable.create(var)

    def addPyCons(self, Constraint cons):
        """
        Adds a customly created cons.

        Parameters
        ----------
        cons : Constraint
            constraint to add

        """
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons.scip_cons))
        Py_INCREF(cons)

    def addVarSOS1(self, Constraint cons, Variable var, weight):
        """
        Add variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            new variable
        weight : weight
            weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, cons.scip_cons, var.scip_var, weight))

    def appendVarSOS1(self, Constraint cons, Variable var):
        """
        Append variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, cons.scip_cons, var.scip_var))

    def addVarSOS2(self, Constraint cons, Variable var, weight):
        """
        Add variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            new variable
        weight : weight
            weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, cons.scip_cons, var.scip_var, weight))

    def appendVarSOS2(self, Constraint cons, Variable var):
        """
        Append variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, cons.scip_cons, var.scip_var))

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

    def chgRhs(self, Constraint cons, rhs):
        """
        Change right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to change the right-hand side from
        rhs : float or None
            new right-hand side (set to None for +infinity)

        """

        if rhs is None:
           rhs = SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgRhsLinear(self._scip, cons.scip_cons, rhs))
        elif constype == 'nonlinear':
            PY_SCIP_CALL(SCIPchgRhsNonlinear(self._scip, cons.scip_cons, rhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getCapacityCumulative(self, Constraint cons):
        """
        Liefert die Kapazität einer cumulative‑Constraint.
        """
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative', \
            "Methode nur für cumulative‑Constraints geeignet"
        return SCIPgetCapacityCumulative(self._scip, cons.scip_cons)

    def getNVarsCumulative(self, Constraint cons):
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        return SCIPgetNVarsCumulative(self._scip, cons.scip_cons)

    def getVarsCumulative(self, Constraint cons):
        """
        Gibt die Startzeit‑Variablen als Python‑Liste zurück.
        """
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        cdef int nvars = SCIPgetNVarsCumulative(self._scip, cons.scip_cons)
        cdef SCIP_VAR** _vars = SCIPgetVarsCumulative(self._scip,
                                                      cons.scip_cons)

        return [ Variable.create(_vars[i]) for i in range(nvars) ]

    def getDurationsCumulative(self, Constraint cons):
        """
        Dict {varname: duration}
        """
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        cdef int nvars = SCIPgetNVarsCumulative(self._scip, cons.scip_cons)
        cdef SCIP_VAR** _vars = SCIPgetVarsCumulative(self._scip,
                                                      cons.scip_cons)
        cdef int* _durs = SCIPgetDurationsCumulative(self._scip,
                                                     cons.scip_cons)

        cdef dict durs = {}
        cdef int i
        for i in range(nvars):
            durs[ bytes(SCIPvarGetName(_vars[i])).decode('utf-8') ] = _durs[i]
        return durs

    def getDemandsCumulative(self, Constraint cons):
        """
        Dict {varname: demand}
        """
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        cdef int nvars = SCIPgetNVarsCumulative(self._scip, cons.scip_cons)
        cdef SCIP_VAR** _vars = SCIPgetVarsCumulative(self._scip,
                                                      cons.scip_cons)
        cdef int* _dem = SCIPgetDemandsCumulative(self._scip,
                                                  cons.scip_cons)

        cdef dict demands = {}
        cdef int i
        for i in range(nvars):
            demands[ bytes(SCIPvarGetName(_vars[i])).decode('utf-8') ] = _dem[i]
        return demands

    def getHminCumulative(self, Constraint cons):
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        return SCIPgetHminCumulative(self._scip, cons.scip_cons)

    def setHminCumulative(self, Constraint cons, hmin):
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        PY_SCIP_CALL(SCIPsetHminCumulative(self._scip, cons.scip_cons, hmin))

    def getHmaxCumulative(self, Constraint cons):
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        return SCIPgetHmaxCumulative(self._scip, cons.scip_cons)

    def setHmaxCumulative(self, Constraint cons, hmax):
        constype = bytes(SCIPconshdlrGetName(
                         SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'cumulative'
        PY_SCIP_CALL(SCIPsetHmaxCumulative(self._scip, cons.scip_cons, hmax))

    def chgCapacityKnapsack(self, Constraint cons, capacity):
        """
        Change capacity of a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint to change the capacity from
        capacity : float or None
            new capacity (set to None for +infinity)

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'knapsack', "method cannot be called for constraints of type " + constype

        if capacity is None:
           capacity = SCIPinfinity(self._scip)

        PY_SCIP_CALL(SCIPchgCapacityKnapsack(self._scip, cons.scip_cons, capacity))

    def chgLhs(self, Constraint cons, lhs):
        """
        Change left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to change the left-hand side from
        lhs : float or None
            new left-hand side (set to None for -infinity)

        """

        if lhs is None:
           lhs = -SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgLhsLinear(self._scip, cons.scip_cons, lhs))
        elif constype == 'nonlinear':
            PY_SCIP_CALL(SCIPchgLhsNonlinear(self._scip, cons.scip_cons, lhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getRhs(self, Constraint cons):
        """
        Retrieve right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to get the right-hand side from

        Returns
        -------
        float

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            return SCIPgetRhsLinear(self._scip, cons.scip_cons)
        elif constype == 'nonlinear':
            return SCIPgetRhsNonlinear(cons.scip_cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getCapacityKnapsack(self, Constraint cons):
        """
        Retrieve capacity of a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint to get the capacity from

        Returns
        -------
        float

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'knapsack', "method cannot be called for constraints of type " + constype

        return SCIPgetCapacityKnapsack(self._scip, cons.scip_cons)

    def getLhs(self, Constraint cons):
        """
        Retrieve left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or nonlinear constraint

        Returns
        -------
        float

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            return SCIPgetLhsLinear(self._scip, cons.scip_cons)
        elif constype == 'nonlinear':
            return SCIPgetLhsNonlinear(cons.scip_cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def chgCoefLinear(self, Constraint cons, Variable var, value):
        """
        Changes coefficient of variable in linear constraint;
        deletes the variable if coefficient is zero; adds variable if not yet contained in the constraint
        This method may only be called during problem creation stage for an original constraint and variable.
        This method requires linear time to search for occurences of the variable in the constraint data.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry
        value : float
            new coefficient of constraint entry

        """
        PY_SCIP_CALL( SCIPchgCoefLinear(self._scip, cons.scip_cons, var.scip_var, value) )

    def delCoefLinear(self, Constraint cons, Variable var):
        """
        Deletes variable from linear constraint
        This method may only be called during problem creation stage for an original constraint and variable.
        This method requires linear time to search for occurrences of the variable in the constraint data.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry

        """

        PY_SCIP_CALL( SCIPdelCoefLinear(self._scip, cons.scip_cons, var.scip_var) )

    def addCoefLinear(self, Constraint cons, Variable var, value):
        """
        Adds coefficient to linear constraint (if it is not zero)

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry
        value : float
            coefficient of constraint entry

        """

        PY_SCIP_CALL( SCIPaddCoefLinear(self._scip, cons.scip_cons, var.scip_var, value) )

    def addCoefKnapsack(self, Constraint cons, Variable var, weight):
        """
        Adds coefficient to knapsack constraint (if it is not zero)

        Parameters
        ----------
        cons : Constraint
            knapsack constraint
        var : Variable
            variable of constraint entry
        weight : float
            coefficient of constraint entry

        """

        PY_SCIP_CALL( SCIPaddCoefKnapsack(self._scip, cons.scip_cons, var.scip_var, weight) )

    def getActivity(self, Constraint cons, Solution sol = None):
        """
        Retrieve activity of given constraint.
        Can only be called after solving is completed.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        sol : Solution or None, optional
            solution to compute activity of, None to use current node's solution (Default value = None)

        Returns
        -------
        float

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol

        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            activity = SCIPgetActivityLinear(self._scip, cons.scip_cons, scip_sol)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        return activity


    def getSlack(self, Constraint cons, Solution sol = None, side = None):
        """
        Retrieve slack of given constraint.
        Can only be called after solving is completed.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        sol : Solution or None, optional
            solution to compute slack of, None to use current node's solution (Default value = None)
        side : str or None, optional
            whether to use 'lhs' or 'rhs' for ranged constraints, None to return minimum (Default value = None)

        Returns
        -------
        float

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol


        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            lhs = SCIPgetLhsLinear(self._scip, cons.scip_cons)
            rhs = SCIPgetRhsLinear(self._scip, cons.scip_cons)
            activity = SCIPgetActivityLinear(self._scip, cons.scip_cons, scip_sol)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        lhsslack = activity - lhs
        rhsslack = rhs - activity

        if side == 'lhs':
            return lhsslack
        elif side == 'rhs':
            return rhsslack
        else:
            return min(lhsslack, rhsslack)

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
        return Constraint.create(transcons)

    def isNLPConstructed(self):
        """
        Returns whether SCIP's internal NLP has been constructed.

        Returns
        -------
        bool

        """
        return SCIPisNLPConstructed(self._scip)

    def getNNlRows(self):
        """
        Gets current number of nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        int

        """
        return SCIPgetNNLPNlRows(self._scip)

    def getNlRows(self):
        """
        Returns a list with the nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        list of NLRow

        """
        cdef SCIP_NLROW** nlrows = SCIPgetNLPNlRows(self._scip)
        cdef int i

        return [NLRow.create(nlrows[i]) for i in range(self.getNNlRows())]

    def getNlRowSolActivity(self, NLRow nlrow, Solution sol = None):
        """
        Gives the activity of a nonlinear row for a given primal solution.

        Parameters
        ----------
        nlrow : NLRow
        sol : Solution or None, optional
            a primal solution, if None, then the current LP solution is used

        Returns
        -------
        float

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* solptr

        solptr = sol.sol if not sol is None else NULL
        PY_SCIP_CALL( SCIPgetNlRowSolActivity(self._scip, nlrow.scip_nlrow, solptr, &activity) )
        return activity

    def getNlRowSolFeasibility(self, NLRow nlrow, Solution sol = None):
        """
        Gives the feasibility of a nonlinear row for a given primal solution

        Parameters
        ----------
        nlrow : NLRow
        sol : Solution or None, optional
            a primal solution, if None, then the current LP solution is used

        Returns
        -------
        bool

        """
        cdef SCIP_Real feasibility
        cdef SCIP_SOL* solptr

        solptr = sol.sol if not sol is None else NULL
        PY_SCIP_CALL( SCIPgetNlRowSolFeasibility(self._scip, nlrow.scip_nlrow, solptr, &feasibility) )
        return feasibility

    def getNlRowActivityBounds(self, NLRow nlrow):
        """
        Gives the minimal and maximal activity of a nonlinear row w.r.t. the variable's bounds.

        Parameters
        ----------
        nlrow : NLRow

        Returns
        -------
        tuple of float

        """
        cdef SCIP_Real minactivity
        cdef SCIP_Real maxactivity

        PY_SCIP_CALL( SCIPgetNlRowActivityBounds(self._scip, nlrow.scip_nlrow, &minactivity, &maxactivity) )
        return (minactivity, maxactivity)

    def printNlRow(self, NLRow nlrow):
        """
        Prints nonlinear row.

        Parameters
        ----------
        nlrow : NLRow

        """
        PY_SCIP_CALL( SCIPprintNlRow(self._scip, nlrow.scip_nlrow, NULL) )

    def checkQuadraticNonlinear(self, Constraint cons):
        """
        Returns if the given constraint is quadratic.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bool

        """
        cdef SCIP_Bool isquadratic
        PY_SCIP_CALL( SCIPcheckQuadraticNonlinear(self._scip, cons.scip_cons, &isquadratic) )
        return isquadratic

    def getTermsQuadratic(self, Constraint cons):
        """
        Retrieve bilinear, quadratic, and linear terms of a quadratic constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bilinterms : list of tuple
        quadterms : list of tuple
        linterms : list of tuple

        """
        cdef SCIP_EXPR* expr
        cdef int termidx

        # linear terms
        cdef SCIP_EXPR** linexprs
        cdef SCIP_Real* lincoefs
        cdef SCIP_Real lincoef
        cdef int nlinvars

        # bilinear terms
        cdef SCIP_EXPR* bilinterm1
        cdef SCIP_EXPR* bilinterm2
        cdef SCIP_Real bilincoef
        cdef int nbilinterms

        # quadratic terms
        cdef SCIP_EXPR* sqrexpr
        cdef SCIP_Real sqrcoef
        cdef int nquadterms

        # variables
        cdef SCIP_VAR* scipvar1
        cdef SCIP_VAR* scipvar2

        assert cons.isNonlinear(), "constraint is not nonlinear"
        assert self.checkQuadraticNonlinear(cons), "constraint is not quadratic"

        expr = SCIPgetExprNonlinear(cons.scip_cons)
        SCIPexprGetQuadraticData(expr, NULL, &nlinvars, &linexprs, &lincoefs, &nquadterms, &nbilinterms, NULL, NULL)

        linterms   = []
        bilinterms = []
        quadterms  = []

        for termidx in range(nlinvars):
            var = Variable.create(SCIPgetVarExprVar(linexprs[termidx]))
            linterms.append((var, lincoefs[termidx]))

        for termidx in range(nbilinterms):
            SCIPexprGetQuadraticBilinTerm(expr, termidx, &bilinterm1, &bilinterm2, &bilincoef, NULL, NULL)
            scipvar1 = SCIPgetVarExprVar(bilinterm1)
            scipvar2 = SCIPgetVarExprVar(bilinterm2)
            var1 = Variable.create(scipvar1)
            var2 = Variable.create(scipvar2)
            if scipvar1 != scipvar2:
                bilinterms.append((var1,var2,bilincoef))
            else:
                quadterms.append((var1,bilincoef,0.0))

        for termidx in range(nquadterms):
            SCIPexprGetQuadraticQuadTerm(expr, termidx, NULL, &lincoef, &sqrcoef, NULL, NULL, &sqrexpr)
            if sqrexpr == NULL:
                continue
            var = Variable.create(SCIPgetVarExprVar(sqrexpr))
            quadterms.append((var,sqrcoef,lincoef))

        return (bilinterms, quadterms, linterms)

    def setRelaxSolVal(self, Variable var, val):
        """
        Sets the value of the given variable in the global relaxation solution.

        Parameters
        ----------
        var : Variable
        val : float

        """
        PY_SCIP_CALL(SCIPsetRelaxSolVal(self._scip, NULL, var.scip_var, val))

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

        return [Constraint.create(conss[i]) for i in range(nconss)]

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
        PY_SCIP_CALL(SCIPdelCons(self._scip, cons.scip_cons))

    def delConsLocal(self, Constraint cons):
        """
        Delete constraint from the current node and its children.

        Parameters
        ----------
        cons : Constraint
            constraint to be deleted

        """
        PY_SCIP_CALL(SCIPdelConsLocal(self._scip, cons.scip_cons))
    
    def getValsLinear(self, Constraint cons):
        """
        Retrieve the coefficients of a linear constraint

        Parameters
        ----------
        cons : Constraint
            linear constraint to get the coefficients of

        Returns
        -------
        dict of str to float

        """
        cdef SCIP_VAR** vars
        cdef SCIP_Real* vals
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("coefficients not available for constraints of type ", constype)

        vals = SCIPgetValsLinear(self._scip, cons.scip_cons)
        vars = SCIPgetVarsLinear(self._scip, cons.scip_cons)

        valsdict = {}
        for i in range(SCIPgetNVarsLinear(self._scip, cons.scip_cons)):
            valsdict[bytes(SCIPvarGetName(vars[i])).decode('utf-8')] = vals[i]

        return valsdict

    def getWeightsKnapsack(self, Constraint cons):
        """
        Retrieve the coefficients of a knapsack constraint

        Parameters
        ----------
        cons : Constraint
            knapsack constraint to get the coefficients of

        Returns
        -------
        dict of str to float

        """
        cdef SCIP_VAR** vars
        cdef SCIP_Longint* vals
        cdef int nvars
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'knapsack':
            raise Warning("weights not available for constraints of type ", constype)

        nvars = SCIPgetNVarsKnapsack(self._scip, cons.scip_cons)
        vals = SCIPgetWeightsKnapsack(self._scip, cons.scip_cons)
        vars = SCIPgetVarsKnapsack(self._scip, cons.scip_cons)

        valsdict = {}
        for i in range(nvars):
            var_name = bytes(SCIPvarGetName(vars[i])).decode('utf-8')
            valsdict[var_name] = vals[i]

        return valsdict

    def getRowLinear(self, Constraint cons):
        """
        Retrieve the linear relaxation of the given linear constraint as a row.
        May return NULL if no LP row was yet created; the user must not modify the row!

        Parameters
        ----------
        cons : Constraint
            linear constraint to get the coefficients of

        Returns
        -------
        Row

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("coefficients not available for constraints of type ", constype)

        cdef SCIP_ROW* row = SCIPgetRowLinear(self._scip, cons.scip_cons)
        return Row.create(row)

    def getDualsolLinear(self, Constraint cons):
        """
        Retrieve the dual solution to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("dual solution values not available for constraints of type ", constype)
        if cons.isOriginal():
            transcons = <Constraint>self.getTransformedCons(cons)
        else:
            transcons = cons
        return SCIPgetDualsolLinear(self._scip, transcons.scip_cons)

    def getDualsolKnapsack(self, Constraint cons):
        """
        Retrieve the dual solution to a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint

        Returns
        -------
        float

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'knapsack':
            raise Warning("dual solution values not available for constraints of type ", constype)
        if cons.isOriginal():
            transcons = cons
        else:
            transcons = <Constraint>self.getTransformedCons(cons)
        return SCIPgetDualsolKnapsack(self._scip, transcons.scip_cons)

    def getDualMultiplier(self, Constraint cons):
        """
        DEPRECATED: Retrieve the dual solution to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
        raise Warning("model.getDualMultiplier(cons) is deprecated: please use model.getDualsolLinear(cons)")
        return self.getDualsolLinear(cons)

    def getDualfarkasLinear(self, Constraint cons):
        """
        Retrieve the dual farkas value to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
        # TODO this should ideally be handled on the SCIP side
        if cons.isOriginal():
            transcons = <Constraint>self.getTransformedCons(cons)
            return SCIPgetDualfarkasLinear(self._scip, transcons.scip_cons)
        else:
            return SCIPgetDualfarkasLinear(self._scip, cons.scip_cons)
    
    def getDualfarkasKnapsack(self, Constraint cons):
        """
        Retrieve the dual farkas value to a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint

        Returns
        -------
        float

        """
        # TODO this should ideally be handled on the SCIP side
        if cons.isOriginal():
            return SCIPgetDualfarkasKnapsack(self._scip, cons.scip_cons)
        else:
            transcons = <Constraint>self.getTransformedCons(cons)
            return SCIPgetDualfarkasKnapsack(self._scip, transcons.scip_cons)

    def getVarRedcost(self, Variable var):
        """
        Retrieve the reduced cost of a variable.

        Parameters
        ----------
        var : Variable
            variable to get the reduced cost of

        Returns
        -------
        float

        """
        redcost = None
        try:
            redcost = SCIPgetVarRedcost(self._scip, var.scip_var)
            if self.getObjectiveSense() == "maximize":
                redcost = -redcost
        except:
            raise Warning("no reduced cost available for variable " + var.name)
        return redcost

    def getDualSolVal(self, Constraint cons, boundconstraint=False):
        """
        Returns dual solution value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to get the dual solution value of
        boundconstraint : bool, optional
            Decides whether to store a bool if the constraint is a bound constraint
            (default = False)

        Returns
        -------
        float

        """
        cdef SCIP_Real _dualsol
        cdef SCIP_Bool _bounded

        if boundconstraint:
            SCIPgetDualSolVal(self._scip, cons.scip_cons, &_dualsol, &_bounded)
        else:
            SCIPgetDualSolVal(self._scip, cons.scip_cons, &_dualsol, NULL)

        return _dualsol

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
                subprobs[i] = (<Model>subprob)._scip
        else:
            subprobs[0] = (<Model>subproblems)._scip

        # creating the default Benders' decomposition
        PY_SCIP_CALL(SCIPcreateBendersDefault(self._scip, subprobs, nsubproblems))
        benders = SCIPfindBenders(self._scip, "default")

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
        """Calls the free subproblem function for the Benders' decomposition.
        This will free all subproblems for all decompositions. """
        cdef SCIP_BENDERS** benders = SCIPgetBenders(self._scip)
        cdef int nbenders = SCIPgetNActiveBenders(self._scip)
        cdef int nsubproblems
        cdef int i
        cdef int j

        # solving all subproblems from all Benders' decompositions
        for i in range(nbenders):
            nsubproblems = SCIPbendersGetNSubproblems(benders[i])
            for j in range(nsubproblems):
                PY_SCIP_CALL(SCIPfreeBendersSubproblem(self._scip, benders[i],
                    j))

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
        cdef SCIP_BENDERS* _benders

        assert type(lowerbounds) is dict

        if benders is None:
            _benders = SCIPfindBenders(self._scip, "default")
        else:
            _benders = benders._benders

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
        PY_SCIP_CALL(SCIPaddBendersSubproblem(self._scip, benders._benders, (<Model>subproblem)._scip))

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
        cdef SCIP_BENDERS* scip_benders
        cdef SCIP_SOL* scip_sol

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        if benders is None:
            scip_benders = SCIPfindBenders(self._scip, "default")
        else:
            scip_benders = benders._benders

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

        cdef SCIP_BENDERS* scip_benders
        cdef SCIP_SOL* scip_sol
        cdef SCIP_Real objective
        cdef SCIP_Bool infeasible

        if isinstance(solution, Solution):
            scip_sol = solution.sol
        else:
            scip_sol = NULL

        if benders is None:
            scip_benders = SCIPfindBenders(self._scip, "default")
        else:
            scip_benders = benders._benders

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
        cdef SCIP_BENDERS* scip_benders
        cdef SCIP* scip_subprob

        if benders is None:
            scip_benders = SCIPfindBenders(self._scip, "default")
        else:
            scip_benders = benders._benders

        scip_subprob = SCIPbendersSubproblem(scip_benders, probnumber)

        return Model.create(scip_subprob)

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
        cdef SCIP_BENDERS* _benders
        cdef SCIP_VAR* _mappedvar

        if benders is None:
            _benders = SCIPfindBenders(self._scip, "default")
        else:
            _benders = benders._benders

        if probnumber == -1:
            PY_SCIP_CALL(SCIPgetBendersMasterVar(self._scip, _benders, var.scip_var, &_mappedvar))
        else:
            PY_SCIP_CALL(SCIPgetBendersSubproblemVar(self._scip, _benders, var.scip_var, &_mappedvar, probnumber))

        if _mappedvar == NULL:
            mappedvar = None
        else:
            mappedvar = Variable.create(_mappedvar)

        return mappedvar

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
        cdef SCIP_BENDERS* _benders
        cdef SCIP_VAR* _auxvar

        if benders is None:
            _benders = SCIPfindBenders(self._scip, "default")
        else:
            _benders = benders._benders

        _auxvar = SCIPbendersGetAuxiliaryVar(_benders, probnumber)
        auxvar = Variable.create(_auxvar)

        return auxvar

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
        cdef SCIP_BENDERS* _benders
        cdef SCIP_SOL* scip_sol
        cdef SCIP_Bool optimal

        if benders is None:
            _benders = SCIPfindBenders(self._scip, "default")
        else:
            _benders = benders._benders

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
        eventhdlr.model = <Model>weakref.proxy(self)
        eventhdlr.name = name
        Py_INCREF(eventhdlr)

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
        pricer.model = <Model>weakref.proxy(self)
        Py_INCREF(pricer)
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
        conshdlr.model = <Model>weakref.proxy(self)
        conshdlr.name = name
        Py_INCREF(conshdlr)

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

    def copyLargeNeighborhoodSearch(self, to_fix, fix_vals) -> Model:
        """
        Creates a configured copy of the transformed problem and applies provided fixings intended for LNS heuristics.

        Parameters
        ----------
        to_fix : List[Variable]
            A list of variables to fix in the copy
        fix_vals : List[Real]
            A list of the values to which to fix the variables in the copy (care their order)

        Returns
        -------
        model : Model
            A model containing the created copy
        """
        cdef SCIP* subscip
        cdef SCIP_HASHMAP* varmap
        cdef SCIP_VAR** orig_vars = SCIPgetVars(self._scip)
        cdef SCIP_VAR** vars = <SCIP_VAR**> malloc(len(to_fix) * sizeof(SCIP_VAR*))
        cdef SCIP_Real* vals = <SCIP_Real*> malloc(len(fix_vals) * sizeof(SCIP_Real))
        cdef SCIP_Real val
        cdef SCIP_Bool valid
        cdef SCIP_Bool success
        cdef int i
        cdef int j = 0

        name_to_val = {var.name: val for var, val in zip(to_fix, fix_vals)}
        for i, var in enumerate(self.getVars()):
            if var.name in name_to_val:
                vars[j] = orig_vars[i]
                vals[j] = <SCIP_Real>name_to_val[var.name]
                j+= 1

        PY_SCIP_CALL(SCIPcreate(&subscip))
        PY_SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(subscip), self.getNVars()) )
        PY_SCIP_CALL( SCIPcopyLargeNeighborhoodSearch(self._scip, subscip, varmap, "LNhS_subscip", vars, vals,
                                                      <int>len(to_fix), False, False, &success, &valid) )
        sub_model = Model.create(subscip)
        sub_model._freescip = True
        free(vars)
        free(vals)
        SCIPhashmapFree(&varmap)

        return sub_model

    def translateSubSol(self, Model sub_model, Solution sol, heur) -> Solution:
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
        cdef SCIP_VAR** vars = <SCIP_VAR**> malloc(self.getNVars() * sizeof(SCIP_VAR*))
        cdef SCIP_SOL* real_sol
        cdef SCIP_SOL* subscip_sol = sol.sol
        cdef SCIP_HEUR* _heur
        cdef SCIP_Bool success
        cdef int i

        for i, var in enumerate(sub_model.getVars()):
            vars[i] = (<Variable>var).scip_var

        name = str_conversion(heur.name)
        _heur = SCIPfindHeur(self._scip, name)
        PY_SCIP_CALL( SCIPtranslateSubSol(self._scip, sub_model._scip, subscip_sol, _heur, vars, &real_sol) )
        solution = Solution.create(self._scip, real_sol)
        free(vars)

        return solution

    def createCons(self, Conshdlr conshdlr, name, initial=True, separate=True, enforce=True, check=True, propagate=True,
                   local=False, modifiable=False, dynamic=False, removable=False, stickingatnode=False):
        """
        Create a constraint of a custom constraint handler.

        Parameters
        ----------
        conshdlr : Conshdlr
            constraint handler
        name : str
            name of constraint handler
        initial : bool, optional
            (Default value = True)
        separate : bool, optional
            (Default value = True)
        enforce : bool, optional
            (Default value = True)
        check : bool, optional
            (Default value = True)
        propagate : bool, optional
            (Default value = True)
        local : bool, optional
            (Default value = False)
        modifiable : bool, optional
            (Default value = False)
        dynamic : bool, optional
            (Default value = False)
        removable : bool, optional
            (Default value = False)
        stickingatnode : bool, optional
            (Default value = False)

        Returns
        -------
        Constraint

        """

        n = str_conversion(name)
        cdef SCIP_CONSHDLR* scip_conshdlr
        scip_conshdlr = SCIPfindConshdlr(self._scip, str_conversion(conshdlr.name))
        constraint = Constraint()
        PY_SCIP_CALL(SCIPcreateCons(self._scip, &(constraint.scip_cons), n, scip_conshdlr, <SCIP_CONSDATA*>constraint,
                                initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))
        return constraint

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
        presol.model = <Model>weakref.proxy(self)
        Py_INCREF(presol)

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
        sepa.model = <Model>weakref.proxy(self)
        sepa.name = name
        Py_INCREF(sepa)

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
        reader.model = <Model>weakref.proxy(self)
        reader.name = name
        Py_INCREF(reader)

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
        prop.model = <Model>weakref.proxy(self)
        Py_INCREF(prop)

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
            positions in the node solving loop where heuristic should be executed
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
        heur.model = <Model>weakref.proxy(self)
        heur.name = name
        Py_INCREF(heur)

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
        relax.model = <Model>weakref.proxy(self)
        relax.name = name

        Py_INCREF(relax)

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
        cutsel.model = <Model>weakref.proxy(self)
        Py_INCREF(cutsel)

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
        branchrule.model = <Model>weakref.proxy(self)
        Py_INCREF(branchrule)

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
        nodesel.model = <Model>weakref.proxy(self)
        Py_INCREF(nodesel)

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
        benders.model = <Model>weakref.proxy(self)
        benders.name = name
        benders._benders = scip_benders
        Py_INCREF(benders)

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
        benderscut.model = <Model>weakref.proxy(self)
        benderscut.benders = benders
        benderscut.name = name
        # TODO: It might be necessary in increment the reference to benders i.e Py_INCREF(benders)
        Py_INCREF(benderscut)


    def getLPBranchCands(self):
        """
        Gets branching candidates for LP solution branching (fractional variables) along with solution values,
        fractionalities, and number of branching candidates; The number of branching candidates does NOT account
        for fractional implicit integer variables which should not be used for branching decisions. Fractional
        implicit integer variables are stored at the positions nlpcands to nlpcands + nfracimplvars - 1
        branching rules should always select the branching candidate among the first npriolpcands of the candidate list

        Returns
        -------
        list of Variable
            list of variables of LP branching candidates
        list of float
            list of LP candidate solution values
        list of float
            list of LP candidate fractionalities
        int
            number of LP branching candidates
        int
            number of candidates with maximal priority
        int
            number of fractional implicit integer variables

        """
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac
        cdef int ncands
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac,
                                          &nlpcands, &npriolpcands, &nfracimplvars))

        return ([Variable.create(lpcands[i]) for i in range(nlpcands)], [lpcandssol[i] for i in range(nlpcands)],
                [lpcandsfrac[i] for i in range(nlpcands)], nlpcands, npriolpcands, nfracimplvars)

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

        return ([Variable.create(pseudocands[i]) for i in range(npseudocands)], npseudocands, npriopseudocands)

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

        PY_SCIP_CALL(SCIPbranchVar(self._scip, (<Variable>variable).scip_var, &downchild, &eqchild, &upchild))
        return Node.create(downchild), Node.create(eqchild), Node.create(upchild)


    def branchVarVal(self, variable, value):
        """
        Branches on variable using a value which separates the domain of the variable.

        Parameters
        ----------
        variable : Variable
            Variable to branch on
        value : float
            value to branch on

        Returns
        -------
        Node
            Node created for the down (left) branch
        Node or None
            Node created for the equal child (middle child). Only exists if the branch variable is integer
        Node
            Node created for the up (right) branch

        """
        cdef SCIP_NODE* downchild
        cdef SCIP_NODE* eqchild
        cdef SCIP_NODE* upchild

        PY_SCIP_CALL(SCIPbranchVarVal(self._scip, (<Variable>variable).scip_var, value, &downchild, &eqchild, &upchild))

        return Node.create(downchild), Node.create(eqchild), Node.create(upchild)

    def calcNodeselPriority(self, Variable variable, branchdir, targetvalue):
        """
        Calculates the node selection priority for moving the given variable's LP value
        to the given target value;
        this node selection priority can be given to the SCIPcreateChild() call.

        Parameters
        ----------
        variable : Variable
            variable on which the branching is applied
        branchdir : PY_SCIP_BRANCHDIR
            type of branching that was performed
        targetvalue : float
            new value of the variable in the child node

        Returns
        -------
        int
            node selection priority for moving the given variable's LP value to the given target value

        """
        return SCIPcalcNodeselPriority(self._scip, variable.scip_var, branchdir, targetvalue)

    def calcChildEstimate(self, Variable variable, targetvalue):
        """
        Calculates an estimate for the objective of the best feasible solution
        contained in the subtree after applying the given branching;
        this estimate can be given to the SCIPcreateChild() call.

        Parameters
        ----------
        variable : Variable
            Variable to compute the estimate for
        targetvalue : float
            new value of the variable in the child node

        Returns
        -------
        float
            objective estimate of the best solution in the subtree after applying the given branching

        """
        return SCIPcalcChildEstimate(self._scip, variable.scip_var, targetvalue)

    def createChild(self, nodeselprio, estimate):
        """
        Create a child node of the focus node.

        Parameters
        ----------
        nodeselprio : int
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

    # Diving methods (Diving is LP related)
    def startDive(self):
        """Initiates LP diving.
        It allows the user to change the LP in several ways, solve, change again, etc,
        without affecting the actual LP. When endDive() is called,
        SCIP will undo all changes done and recover the LP it had before startDive."""
        PY_SCIP_CALL(SCIPstartDive(self._scip))

    def endDive(self):
        """Quits probing and resets bounds and constraints to the focus node's environment."""
        PY_SCIP_CALL(SCIPendDive(self._scip))

    def chgVarObjDive(self, Variable var, newobj):
        """
        Changes (column) variable's objective value in current dive.

        Parameters
        ----------
        var : Variable
        newobj : float

        """
        PY_SCIP_CALL(SCIPchgVarObjDive(self._scip, var.scip_var, newobj))

    def chgVarLbDive(self, Variable var, newbound):
        """
        Changes variable's current lb in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
        PY_SCIP_CALL(SCIPchgVarLbDive(self._scip, var.scip_var, newbound))

    def chgVarUbDive(self, Variable var, newbound):
        """
        Changes variable's current ub in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
        PY_SCIP_CALL(SCIPchgVarUbDive(self._scip, var.scip_var, newbound))

    def getVarLbDive(self, Variable var):
        """
        Returns variable's current lb in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
        return SCIPgetVarLbDive(self._scip, var.scip_var)

    def getVarUbDive(self, Variable var):
        """
        Returns variable's current ub in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
        return SCIPgetVarUbDive(self._scip, var.scip_var)

    def chgRowLhsDive(self, Row row, newlhs):
        """
        Changes row lhs in current dive, change will be undone after diving
        ends, for permanent changes use SCIPchgRowLhs().

        Parameters
        ----------
        row : Row
        newlhs : float

        """
        PY_SCIP_CALL(SCIPchgRowLhsDive(self._scip, row.scip_row, newlhs))

    def chgRowRhsDive(self, Row row, newrhs):
        """
        Changes row rhs in current dive, change will be undone after diving
        ends. For permanent changes use SCIPchgRowRhs().

        Parameters
        ----------
        row : Row
        newrhs : float

        """
        PY_SCIP_CALL(SCIPchgRowRhsDive(self._scip, row.scip_row, newrhs))

    def addRowDive(self, Row row):
        """
        Adds a row to the LP in current dive.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPaddRowDive(self._scip, row.scip_row))

    def solveDiveLP(self, itlim = -1):
        """
        Solves the LP of the current dive. No separation or pricing is applied.

        Parameters
        ----------
        itlim : int, optional
            maximal number of LP iterations to perform (Default value = -1, that is, no limit)

        Returns
        -------
        lperror : bool
            whether an unresolved lp error occured
        cutoff : bool
            whether the LP was infeasible or the objective limit was reached

        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL(SCIPsolveDiveLP(self._scip, itlim, &lperror, &cutoff))
        return lperror, cutoff

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

    def chgVarObjProbing(self, Variable var, newobj):
        """Changes (column) variable's objective value during probing mode."""
        PY_SCIP_CALL( SCIPchgVarObjProbing(self._scip, var.scip_var, newobj) )

    def chgVarLbProbing(self, Variable var, lb):
        """
        Changes the variable lower bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbProbing(self._scip, var.scip_var, lb))

    def chgVarUbProbing(self, Variable var, ub):
        """
        Changes the variable upper bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        ub : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbProbing(self._scip, var.scip_var, ub))

    def fixVarProbing(self, Variable var, fixedval):
        """
        Fixes a variable at the current probing node.

        Parameters
        ----------
        var : Variable
        fixedval : float

        """
        PY_SCIP_CALL( SCIPfixVarProbing(self._scip, var.scip_var, fixedval) )

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

    def writeLP(self, filename="LP.lp"):
        """
        Writes current LP to a file.

        Parameters
        ----------
        filename : str, optional
            file name (Default value = "LP.lp")

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        absfile = str_conversion(abspath(filename))
        PY_SCIP_CALL( SCIPwriteLP(self._scip, absfile) )

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def createSol(self, Heur heur = None, initlp=False):
        """
        Create a new primal solution in the transformed space.

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
            PY_SCIP_CALL(SCIPcreateSol(self._scip, &_sol, _heur))
        else:
            PY_SCIP_CALL(SCIPcreateLPSol(self._scip, &_sol, _heur))
        solution = Solution.create(self._scip, _sol)
        return solution

    def createPartialSol(self, Heur heur = None):
        """
        Create a partial primal solution, initialized to unknown values.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)

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
        PY_SCIP_CALL(SCIPcreatePartialSol(self._scip, &_sol, _heur))
        partialsolution = Solution.create(self._scip, _sol)
        return partialsolution

    def createOrigSol(self, Heur heur = None):
        """
        Create a new primal solution in the original space.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)

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

        PY_SCIP_CALL(SCIPcreateOrigSol(self._scip, &_sol, _heur))
        solution = Solution.create(self._scip, _sol)
        return solution

    def printBestSol(self, write_zeros=False):
        """
        Prints the best feasible primal solution.

        Parameters
        ----------
        write_zeros : bool, optional
            include variables that are set to zero (Default = False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPprintBestSol(self._scip, NULL, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def printSol(self, Solution solution=None, write_zeros=False):
        """
        Print the given primal solution.

        Parameters
        ----------
        solution : Solution or None, optional
            solution to print (default = None)
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        if solution is None:
            PY_SCIP_CALL(SCIPprintSol(self._scip, NULL, NULL, write_zeros))
        else:
            PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, NULL, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeBestSol(self, filename="origprob.sol", write_zeros=False):
        """
        Write the best feasible primal solution to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintBestSol(self._scip, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeBestTransSol(self, filename="transprob.sol", write_zeros=False):
        """
        Write the best feasible primal solution for the transformed problem to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this double opening pattern to ensure that IOErrors are
        #   triggered early and in python not in C, Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintBestTransSol(self._scip, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeSol(self, Solution solution, filename="origprob.sol", write_zeros=False):
        """
        Write the given primal solution to a file.

        Parameters
        ----------
        solution : Solution
            solution to write
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeTransSol(self, Solution solution, filename="transprob.sol", write_zeros=False):
        """
        Write the given transformed primal solution to a file.

        Parameters
        ----------
        solution : Solution
            transformed solution to write
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintTransSol(self._scip, solution.sol, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    # perhaps this should not be included as it implements duplicated functionality
    #   (as does it's namesake in SCIP)
    def readSol(self, filename):
        """
        Reads a given solution file, problem has to be transformed in advance.

        Parameters
        ----------
        filename : str
            name of the input file

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        absfile = str_conversion(abspath(filename))
        PY_SCIP_CALL(SCIPreadSol(self._scip, absfile))

        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def readSolFile(self, filename):
        """
        Reads a given solution file.

        Solution is created but not added to storage/the model.
        Use 'addSol' OR 'trySol' to add it.

        Parameters
        ----------
        filename : str
            name of the input file

        Returns
        -------
        Solution

        """
        cdef SCIP_Bool partial
        cdef SCIP_Bool error
        cdef SCIP_Bool stored
        cdef Solution solution

        str_absfile = abspath(filename)
        absfile = str_conversion(str_absfile)
        solution = self.createSol()

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPreadSolFile(self._scip, absfile, solution.sol, False, &partial, &error))

        locale.setlocale(locale.LC_NUMERIC, user_locale)

        if error:
            raise Exception("SCIP: reading solution from file " + str_absfile + " failed!")

        return solution

    def setSolVal(self, Solution solution, Variable var, val):
        """
        Set a variable in a solution.

        Parameters
        ----------
        solution : Solution
            solution to be modified
        var : Variable
            variable in the solution
        val : float
            value of the specified variable

        """
        cdef SCIP_SOL* _sol
        _sol = <SCIP_SOL*>solution.sol

        assert _sol != NULL, "Cannot set value to a freed solution."
        PY_SCIP_CALL(SCIPsetSolVal(self._scip, _sol, var.scip_var, val))

    def trySol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, free=True):
        """
        Check given primal solution for feasibility and try to add it to the storage.

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
            does integrality have to be checked? (Default value = True)
        checklprows : bool, optional
            do current LP rows (both local and global) have to be checked? (Default value = True)
        free : bool, optional
            should solution be freed? (Default value = True)

        Returns
        -------
        stored : bool
            whether given solution was feasible and good enough to keep

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        else:
            PY_SCIP_CALL(SCIPtrySol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        return stored

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
        self._bestSol = Solution.create(self._scip, SCIPgetBestSol(self._scip))
        return self._bestSol

    def getSolObjVal(self, Solution sol, original=True):
        """
        Retrieve the objective value of the solution.

        Parameters
        ----------
        sol : Solution
        original : bool, optional
            objective value in original space (Default value = True)

        Returns
        -------
        float

        """
        if sol == None:
            sol = Solution.create(self._scip, NULL)

        sol._checkStage("getSolObjVal")

        if original:
            objval = SCIPgetSolOrigObj(self._scip, sol.sol)
        else:
            objval = SCIPgetSolTransObj(self._scip, sol.sol)

        return objval

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
        return SCIPgetSolTime(self._scip, sol.sol)

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

    def getSolVal(self, Solution sol, Expr expr):
        """
        Retrieve value of given variable or expression in the given solution or in
        the LP/pseudo solution if sol == None

        Parameters
        ----------
        sol : Solution
        expr : Expr
            polynomial expression to query the value of

        Returns
        -------
        float

        Notes
        -----
        A variable is also an expression.

        """
        # no need to create a NULL solution wrapper in case we have a variable
        if sol == None and isinstance(expr, Variable):
            var = <Variable> expr
            return SCIPgetSolVal(self._scip, NULL, var.scip_var)
        if sol == None:
            sol = Solution.create(self._scip, NULL)
        return sol[expr]

    def getVal(self, expr: Union[Expr, MatrixExpr] ):
        """
        Retrieve the value of the given variable or expression in the best known solution.
        Can only be called after solving is completed.

        Parameters
        ----------
        expr : Expr ot MatrixExpr
            polynomial expression to query the value of

        Returns
        -------
        float

        Notes
        -----
        A variable is also an expression.

        """
        stage_check = SCIPgetStage(self._scip) not in [SCIP_STAGE_INIT, SCIP_STAGE_FREE]

        if not stage_check or self._bestSol.sol == NULL and SCIPgetStage(self._scip) != SCIP_STAGE_SOLVING:
            raise Warning("Method cannot be called in stage ", self.getStage())

        if isinstance(expr, MatrixExpr):
            result = np.empty(expr.shape, dtype=float)
            for idx in np.ndindex(result.shape):
                result[idx] = self.getSolVal(self._bestSol, expr[idx])
        else:
            result = self.getSolVal(self._bestSol, expr)

        return result

    def hasPrimalRay(self):
        """
        Returns whether a primal ray is stored that proves unboundedness of the LP relaxation.

        Returns
        -------
        bool

        """
        return SCIPhasPrimalRay(self._scip)

    def getPrimalRayVal(self, Variable var):
        """
        Gets value of given variable in primal ray causing unboundedness of the LP relaxation.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
        assert SCIPhasPrimalRay(self._scip), "The problem does not have a primal ray."

        return SCIPgetPrimalRayVal(self._scip, var.scip_var)

    def getPrimalRay(self):
        """
        Gets primal ray causing unboundedness of the LP relaxation.

        Returns
        -------
        list of float

        """
        assert SCIPhasPrimalRay(self._scip), "The problem does not have a primal ray."
        cdef SCIP_VAR** vars = SCIPgetVars(self._scip)
        cdef int nvars = SCIPgetNVars(self._scip)
        cdef int i

        ray = []
        for i in range(nvars):
            ray.append(float(SCIPgetPrimalRayVal(self._scip, vars[i])))

        return ray

    def getPrimalbound(self):
        """
        Retrieve the best primal bound.

        Returns
        -------
        float

        """
        return SCIPgetPrimalbound(self._scip)

    def getDualbound(self):
        """
        Retrieve the best dual bound.

        Returns
        -------
        float

        """
        return SCIPgetDualbound(self._scip)

    def getDualboundRoot(self):
        """
        Retrieve the best root dual bound.

        Returns
        -------
        float

        """
        return SCIPgetDualboundRoot(self._scip)

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

    def catchEvent(self, eventtype, Eventhdlr eventhdlr):
        """
        Catches a global (not variable or row dependent) event.

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")

        Py_INCREF(self)
        PY_SCIP_CALL(SCIPcatchEvent(self._scip, eventtype, _eventhdlr, NULL, NULL))

    def dropEvent(self, eventtype, Eventhdlr eventhdlr):
        """
        Drops a global event (stops tracking the event).

        Parameters
        ----------
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")

        Py_DECREF(self)
        PY_SCIP_CALL(SCIPdropEvent(self._scip, eventtype, _eventhdlr, NULL, -1))

    def catchVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        """
        Catches an objective value or domain change event on the given transformed variable.

        Parameters
        ----------
        var : Variable
        eventtype : PY_SCIP_EVENTTYPE
        eventhdlr : Eventhdlr

        """
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
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
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
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
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
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
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropRowEvent(self._scip, row.scip_row, eventtype, _eventhdlr, NULL, -1))

    # Statistic Methods

    def printStatistics(self):
        """Print statistics."""
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPprintStatistics(self._scip, NULL))

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

    def setParamsCountsols(self):
        """Sets SCIP parameters such that a valid counting process is possible."""
        PY_SCIP_CALL(SCIPsetParamsCountsols(self._scip))

    def freeReoptSolve(self):
        """Frees all solution process data and prepares for reoptimization."""

        if self.getStage() not in [SCIP_STAGE_INIT,
                                 SCIP_STAGE_PROBLEM,
                                 SCIP_STAGE_TRANSFORMED,
                                 SCIP_STAGE_PRESOLVING,
                                 SCIP_STAGE_PRESOLVED,
                                 SCIP_STAGE_SOLVING,
                                 SCIP_STAGE_SOLVED]:
            raise Warning("method cannot be called in stage %i." % self.getStage())
        PY_SCIP_CALL(SCIPfreeReoptSolve(self._scip))

    def chgReoptObjective(self, coeffs, sense = 'minimize'):
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        coeffs : list of float
            the coefficients
        sense : str
            the objective sense (Default value = 'minimize')

        """
        cdef SCIP_VAR** vars
        cdef int nvars
        cdef SCIP_Real* _coeffs
        cdef SCIP_OBJSENSE objsense
        cdef SCIP_Real coef
        cdef int i

        if sense == "minimize":
            objsense = SCIP_OBJSENSE_MINIMIZE
        elif sense == "maximize":
            objsense = SCIP_OBJSENSE_MAXIMIZE
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

        assert isinstance(coeffs, Expr), "given coefficients are not Expr but %s" % coeffs.__class__.__name__

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            raise ValueError("Constant offsets in objective are not supported!")

        vars = SCIPgetOrigVars(self._scip)
        nvars = SCIPgetNOrigVars(self._scip)
        _coeffs = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))

        for i in range(nvars):
            _coeffs[i] = 0.0

        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                for i in range(nvars):
                    if vars[i] == var.scip_var:
                        _coeffs[i] = coef

        PY_SCIP_CALL(SCIPchgReoptObjective(self._scip, objsense, vars, &_coeffs[0], nvars))

        free(_coeffs)

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

    def getVarStrongbranchLast(self, Variable var):
        """
        Get the results of the last strong branching call on this variable (potentially was called
        at another node).

        Parameters
        ----------
        var : Variable
            variable to get the previous strong branching information from

        Returns
        -------
        down : float
            The dual bound of the LP after branching down on the variable
        up : float
            The dual bound of the LP after branchign up on the variable
        downvalid : bool
            stores whether the returned down value is a valid dual bound, or NULL
        upvalid : bool
            stores whether the returned up value is a valid dual bound, or NULL
        solval : float
            The solution value of the variable at the last strong branching call
        lpobjval : float
            The LP objective value at the time of the last strong branching call

        """

        cdef SCIP_Real down
        cdef SCIP_Real up
        cdef SCIP_Real solval
        cdef SCIP_Real lpobjval
        cdef SCIP_Bool downvalid
        cdef SCIP_Bool upvalid

        PY_SCIP_CALL(SCIPgetVarStrongbranchLast(self._scip, var.scip_var, &down, &up, &downvalid, &upvalid, &solval, &lpobjval))

        return down, up, downvalid, upvalid, solval, lpobjval

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

    def getVarStrongbranch(self, Variable var, itlim, idempotent=False, integral=False):
        """
        Strong branches and gets information on column variable.

        Parameters
        ----------
        var : Variable
            Variable to get strong branching information on
        itlim : int
            LP iteration limit for total strong branching calls
        idempotent : bool, optional
            Should SCIP's state remain the same after the call?
        integral : bool, optional
            Boolean on whether the variable is currently integer.

        Returns
        -------
        down : float
            The dual bound of the LP after branching down on the variable
        up : float
            The dual bound of the LP after branchign up on the variable
        downvalid : bool
            stores whether the returned down value is a valid dual bound, or NULL
        upvalid : bool
            stores whether the returned up value is a valid dual bound, or NULL
        downinf : bool
            store whether the downwards branch is infeasible
        upinf : bool
            store whether the upwards branch is infeasible
        downconflict : bool
            store whether a conflict constraint was created for an infeasible downwards branch
        upconflict : bool
            store whether a conflict constraint was created for an infeasible upwards branch
        lperror : bool
            whether an unresolved LP error occurred in the solving process

        """

        cdef SCIP_Real down
        cdef SCIP_Real up
        cdef SCIP_Bool downvalid
        cdef SCIP_Bool upvalid
        cdef SCIP_Bool downinf
        cdef SCIP_Bool upinf
        cdef SCIP_Bool downconflict
        cdef SCIP_Bool upconflict
        cdef SCIP_Bool lperror

        if integral:
            PY_SCIP_CALL(SCIPgetVarStrongbranchInt(self._scip, var.scip_var, itlim, idempotent, &down, &up, &downvalid,
                                                   &upvalid, &downinf, &upinf, &downconflict, &upconflict, &lperror))
        else:
            PY_SCIP_CALL(SCIPgetVarStrongbranchFrac(self._scip, var.scip_var, itlim, idempotent, &down, &up, &downvalid,
                                                    &upvalid, &downinf, &upinf, &downconflict, &upconflict, &lperror))

        return down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror

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


    def getBipartiteGraphRepresentation(self, prev_col_features=None, prev_edge_features=None, prev_row_features=None,
                                        static_only=False, suppress_warnings=False):
        """
        This function generates the bipartite graph representation of an LP, which was first used in
        the following paper:
        @inproceedings{conf/nips/GasseCFCL19,
        title={Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
        author={Gasse, Maxime and Chételat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
        booktitle={Advances in Neural Information Processing Systems 32},
        year={2019}
        }
        The exact features have been modified compared to the original implementation.
        This function is used mainly in the machine learning community for MIP.
        A user can only call it during the solving process, when there is an LP object. This means calling it
        from some user defined plugin on the Python side.
        An example plugin is a branching rule or an event handler, which is exclusively created to call this function.
        The user must then make certain to return the appropriate SCIP_RESULT (e.g. DIDNOTRUN)

        Parameters
        ----------
        prev_col_features : list of list or None, optional
            The list of column features previously returned by this function
        prev_edge_features : list of list or None, optional
            The list of edge features previously returned by this function
        prev_row_features : list of list or None, optional
            The list of row features previously returned by this function
        static_only : bool, optional
            Whether exclusively static features should be generated
        suppress_warnings : bool, optional
            Whether warnings should be suppressed

        Returns
        -------
        col_features : list of list
        edge_features : list of list
        row_features : list of list
        dict
            The feature mappings for the columns, edges, and rows

        """
        cdef SCIP* scip = self._scip
        cdef SCIP_VARTYPE vtype
        cdef SCIP_Real sim, prod
        cdef int col_i
        cdef int i
        cdef int j
        cdef int k

        # Check if SCIP is in the correct stage
        if SCIPgetStage(scip) != SCIP_STAGE_SOLVING:
            raise Warning("This functionality can only been called in SCIP_STAGE SOLVING. The row and column"
                          "information is then accessible")

        # Generate column features
        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        cdef int ncols = SCIPgetNLPCols(scip)

        if static_only:
            n_col_features = 5
            col_feature_map = {"continuous": 0, "binary": 1, "integer": 2, "implicit_integer": 3, "obj_coef": 4}
        else:
            n_col_features = 19
            col_feature_map = {"continuous": 0, "binary": 1, "integer": 2, "implicit_integer": 3, "obj_coef": 4,
                               "has_lb": 5, "has_ub": 6, "sol_at_lb": 7, "sol_at_ub": 8, "sol_val": 9, "sol_frac": 10,
                               "red_cost": 11, "basis_lower": 12, "basis_basic": 13, "basis_upper": 14,
                               "basis_zero": 15, "best_incumbent_val": 16, "avg_incumbent_val": 17, "age": 18}

        if prev_col_features is None:
            col_features = [[0 for i in range(n_col_features)] for j in range(ncols)]
        else:
            assert len(prev_col_features) > 0, "Previous column features is empty"
            col_features = prev_col_features
            if len(prev_col_features) != ncols:
                if not suppress_warnings:
                    raise Warning(f"The number of columns has changed. Previous column data being ignored")
                else:
                    col_features = [[0 for i in range(n_col_features)] for j in range(ncols)]
                    prev_col_features = None
            if len(prev_col_features[0]) != n_col_features:
                raise Warning(f"Dimension mismatch in provided previous features and new features:"
                              f"{len(prev_col_features[0])} != {n_col_features}")

        cdef SCIP_SOL* sol = SCIPgetBestSol(scip)
        cdef SCIP_VAR* var
        cdef SCIP_Real lb
        cdef SCIP_Real ub
        cdef SCIP_Real solval
        cdef SCIP_BASESTAT basis_status

        for i in range(ncols):
            col_i = SCIPcolGetLPPos(cols[i])
            var = SCIPcolGetVar(cols[i])

            lb = SCIPcolGetLb(cols[i])
            ub = SCIPcolGetUb(cols[i])
            solval = SCIPcolGetPrimsol(cols[i])

            # Collect the static features first (don't need to changed if previous features are passed)
            if prev_col_features is None:
                # Variable types
                vtype = SCIPvarGetType(var)
                if vtype == SCIP_VARTYPE_BINARY:
                    col_features[col_i][col_feature_map["binary"]] = 1
                elif vtype == SCIP_VARTYPE_INTEGER:
                    col_features[col_i][col_feature_map["integer"]] = 1
                elif vtype == SCIP_VARTYPE_CONTINUOUS:
                    col_features[col_i][col_feature_map["continuous"]] = 1
                elif vtype == SCIP_VARTYPE_IMPLINT:
                    col_features[col_i][col_feature_map["implicit_integer"]] = 1
                # Objective coefficient
                col_features[col_i][col_feature_map["obj_coef"]] = SCIPcolGetObj(cols[i])

            # Collect the dynamic features
            if not static_only:
                # Lower bound
                if not SCIPisInfinity(scip, abs(lb)):
                    col_features[col_i][col_feature_map["has_lb"]] = 1

                # Upper bound
                if not SCIPisInfinity(scip, abs(ub)):
                    col_features[col_i][col_feature_map["has_ub"]] = 1

                # Basis status
                basis_status = SCIPcolGetBasisStatus(cols[i])
                if basis_status == SCIP_BASESTAT_LOWER:
                    col_features[col_i][col_feature_map["basis_lower"]] = 1
                elif basis_status == SCIP_BASESTAT_BASIC:
                    col_features[col_i][col_feature_map["basis_basic"]] = 1
                elif basis_status == SCIP_BASESTAT_UPPER:
                    col_features[col_i][col_feature_map["basis_upper"]] = 1
                elif basis_status == SCIP_BASESTAT_ZERO:
                    col_features[col_i][col_feature_map["basis_zero"]] = 1

                # Reduced cost
                col_features[col_i][col_feature_map["red_cost"]] = SCIPgetColRedcost(scip, cols[i])

                # Age
                col_features[col_i][col_feature_map["age"]] = SCIPcolGetAge(cols[i])

                # LP solution value
                col_features[col_i][col_feature_map["sol_val"]] = solval
                col_features[col_i][col_feature_map["sol_frac"]] = SCIPfeasFrac(scip, solval)
                col_features[col_i][col_feature_map["sol_at_lb"]] = int(SCIPisEQ(scip, solval, lb))
                col_features[col_i][col_feature_map["sol_at_ub"]] = int(SCIPisEQ(scip, solval, ub))

                # Incumbent solution value
                if sol is NULL:
                    col_features[col_i][col_feature_map["best_incumbent_val"]] = None
                    col_features[col_i][col_feature_map["avg_incumbent_val"]] = None
                else:
                    col_features[col_i][col_feature_map["best_incumbent_val"]] = SCIPgetSolVal(scip, sol, var)
                    col_features[col_i][col_feature_map["avg_incumbent_val"]] = SCIPvarGetAvgSol(var)

        # Generate row features
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
        cdef int nrows = SCIPgetNLPRows(scip)

        if static_only:
            n_row_features = 6
            row_feature_map = {"has_lhs": 0, "has_rhs": 1, "n_non_zeros": 2, "obj_cosine": 3, "bias": 4, "norm": 5}
        else:
            n_row_features = 14
            row_feature_map = {"has_lhs": 0, "has_rhs": 1, "n_non_zeros": 2, "obj_cosine": 3, "bias": 4, "norm": 5,
                               "sol_at_lhs": 6, "sol_at_rhs": 7, "dual_sol": 8, "age": 9,
                               "basis_lower": 10, "basis_basic": 11, "basis_upper": 12, "basis_zero": 13}

        if prev_row_features is None:
            row_features = [[0 for i in range(n_row_features)] for j in range(nrows)]
        else:
            assert len(prev_row_features) > 0, "Previous row features is empty"
            row_features = prev_row_features
            if len(prev_row_features) != nrows:
                if not suppress_warnings:
                    raise Warning(f"The number of rows has changed. Previous row data being ignored")
                else:
                    row_features = [[0 for i in range(n_row_features)] for j in range(nrows)]
                    prev_row_features = None
            if len(prev_row_features[0]) != n_row_features:
                raise Warning(f"Dimension mismatch in provided previous features and new features:"
                              f"{len(prev_row_features[0])} != {n_row_features}")

        cdef SCIP_Real lhs
        cdef SCIP_Real rhs
        cdef SCIP_Real cst
        cdef int nnzrs = 0

        for i in range(nrows):

            # lhs <= activity + cst <= rhs
            lhs = SCIProwGetLhs(rows[i])
            rhs = SCIProwGetRhs(rows[i])
            cst = SCIProwGetConstant(rows[i])
            activity = SCIPgetRowLPActivity(scip, rows[i])

            if prev_row_features is None:
                # number of coefficients
                row_features[i][row_feature_map["n_non_zeros"]] = SCIProwGetNLPNonz(rows[i])
                nnzrs += row_features[i][row_feature_map["n_non_zeros"]]

                # left-hand-side
                if not SCIPisInfinity(scip, abs(lhs)):
                    row_features[i][row_feature_map["has_lhs"]] = 1

                # right-hand-side
                if not SCIPisInfinity(scip, abs(rhs)):
                    row_features[i][row_feature_map["has_rhs"]] = 1

                # bias
                row_features[i][row_feature_map["bias"]] = cst

                # Objective cosine similarity
                row_features[i][row_feature_map["obj_cosine"]] = SCIPgetRowObjParallelism(scip, rows[i])

                # L2 norm
                row_features[i][row_feature_map["norm"]] = SCIProwGetNorm(rows[i])

            if not static_only:

                # Dual solution
                row_features[i][row_feature_map["dual_sol"]] = SCIProwGetDualsol(rows[i])

                # Basis status
                basis_status = SCIProwGetBasisStatus(rows[i])
                if basis_status == SCIP_BASESTAT_LOWER:
                    row_features[i][row_feature_map["basis_lower"]] = 1
                elif basis_status == SCIP_BASESTAT_BASIC:
                    row_features[i][row_feature_map["basis_basic"]] = 1
                elif basis_status == SCIP_BASESTAT_UPPER:
                    row_features[i][row_feature_map["basis_upper"]] = 1
                elif basis_status == SCIP_BASESTAT_ZERO:
                    row_features[i][row_feature_map["basis_zero"]] = 1

                # Age
                row_features[i][row_feature_map["age"]] = SCIProwGetAge(rows[i])

                # Is tight
                row_features[i][row_feature_map["sol_at_lhs"]] = int(SCIPisEQ(scip, activity, lhs))
                row_features[i][row_feature_map["sol_at_rhs"]] = int(SCIPisEQ(scip, activity, rhs))

        # Generate edge (coefficient) features
        cdef SCIP_COL** row_cols
        cdef SCIP_Real * row_vals

        n_edge_features = 3
        edge_feature_map = {"col_idx": 0, "row_idx": 1, "coef": 2}
        if prev_edge_features is None:
            edge_features = [[0 for i in range(n_edge_features)] for j in range(nnzrs)]
            j = 0
            for i in range(nrows):
                # coefficient indexes and values
                row_cols = SCIProwGetCols(rows[i])
                row_vals = SCIProwGetVals(rows[i])
                for k in range(row_features[i][row_feature_map["n_non_zeros"]]):
                    edge_features[j][edge_feature_map["col_idx"]] = SCIPcolGetLPPos(row_cols[k])
                    edge_features[j][edge_feature_map["row_idx"]] = i
                    edge_features[j][edge_feature_map["coef"]] = row_vals[k]
                    j += 1
        else:
            assert len(prev_edge_features) > 0, "Previous edge features is empty"
            edge_features = prev_edge_features
            if len(prev_edge_features) != nnzrs:
                if not suppress_warnings:
                    raise Warning(f"The number of coefficients in the LP has changed. Previous edge data being ignored")
                else:
                    edge_features = [[0 for i in range(3)] for j in range(nnzrs)]
                    prev_edge_features = None
            if len(prev_edge_features[0]) != 3:
                raise Warning(f"Dimension mismatch in provided previous features and new features:"
                              f"{len(prev_edge_features[0])} != 3")

        return (col_features, edge_features, row_features,
                {"col": col_feature_map, "edge": edge_feature_map, "row": row_feature_map})

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

                if _is_number(relevant_value):
                    result[stat_name] = float(relevant_value)
                    if stat_name == "Solutions found" and result[stat_name] == 0:
                        break

                else: # it's a string
                    result[stat_name] = relevant_value

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
