##@file scip.pxd
#@brief holding prototype of the SCIP public functions to use them in PySCIPOpt
cdef extern from "scip/scip.h":
    # SCIP internal types
    ctypedef int SCIP_RETCODE
    cdef extern from "scip/type_retcode.h":
        SCIP_RETCODE SCIP_OKAY
        SCIP_RETCODE SCIP_ERROR
        SCIP_RETCODE SCIP_NOMEMORY
        SCIP_RETCODE SCIP_READERROR
        SCIP_RETCODE SCIP_WRITEERROR
        SCIP_RETCODE SCIP_NOFILE
        SCIP_RETCODE SCIP_FILECREATEERROR
        SCIP_RETCODE SCIP_LPERROR
        SCIP_RETCODE SCIP_NOPROBLEM
        SCIP_RETCODE SCIP_INVALIDCALL
        SCIP_RETCODE SCIP_INVALIDDATA
        SCIP_RETCODE SCIP_INVALIDRESULT
        SCIP_RETCODE SCIP_PLUGINNOTFOUND
        SCIP_RETCODE SCIP_PARAMETERUNKNOWN
        SCIP_RETCODE SCIP_PARAMETERWRONGTYPE
        SCIP_RETCODE SCIP_PARAMETERWRONGVAL
        SCIP_RETCODE SCIP_KEYALREADYEXISTING
        SCIP_RETCODE SCIP_MAXDEPTHLEVEL

    ctypedef int SCIP_VARTYPE
    cdef extern from "scip/type_var.h":
        SCIP_VARTYPE SCIP_VARTYPE_BINARY
        SCIP_VARTYPE SCIP_VARTYPE_INTEGER
        SCIP_VARTYPE SCIP_VARTYPE_IMPLINT
        SCIP_VARTYPE SCIP_VARTYPE_CONTINUOUS

    ctypedef int SCIP_OBJSENSE
    cdef extern from "scip/type_prob.h":
        SCIP_OBJSENSE SCIP_OBJSENSE_MAXIMIZE
        SCIP_OBJSENSE SCIP_OBJSENSE_MINIMIZE

    # This version is used in LPI.
    ctypedef int SCIP_OBJSEN
    cdef extern from "lpi/type_lpi.h":
        SCIP_OBJSEN SCIP_OBJSEN_MAXIMIZE
        SCIP_OBJSEN SCIP_OBJSEN_MINIMIZE

    ctypedef int SCIP_BOUNDTYPE
    cdef extern from "scip/type_lp.h":
        SCIP_BOUNDTYPE SCIP_BOUNDTYPE_LOWER
        SCIP_BOUNDTYPE SCIP_BOUNDTYPE_UPPER

    ctypedef int SCIP_RESULT
    cdef extern from "scip/type_result.h":
        SCIP_RESULT SCIP_DIDNOTRUN
        SCIP_RESULT SCIP_DELAYED
        SCIP_RESULT SCIP_DIDNOTFIND
        SCIP_RESULT SCIP_FEASIBLE
        SCIP_RESULT SCIP_INFEASIBLE
        SCIP_RESULT SCIP_UNBOUNDED
        SCIP_RESULT SCIP_CUTOFF
        SCIP_RESULT SCIP_SEPARATED
        SCIP_RESULT SCIP_NEWROUND
        SCIP_RESULT SCIP_REDUCEDDOM
        SCIP_RESULT SCIP_CONSADDED
        SCIP_RESULT SCIP_CONSCHANGED
        SCIP_RESULT SCIP_BRANCHED
        SCIP_RESULT SCIP_SOLVELP
        SCIP_RESULT SCIP_FOUNDSOL
        SCIP_RESULT SCIP_SUSPENDED
        SCIP_RESULT SCIP_SUCCESS

    ctypedef int SCIP_STATUS
    cdef extern from "scip/type_stat.h":
        SCIP_STATUS SCIP_STATUS_UNKNOWN
        SCIP_STATUS SCIP_STATUS_USERINTERRUPT
        SCIP_STATUS SCIP_STATUS_NODELIMIT
        SCIP_STATUS SCIP_STATUS_TOTALNODELIMIT
        SCIP_STATUS SCIP_STATUS_STALLNODELIMIT
        SCIP_STATUS SCIP_STATUS_TIMELIMIT
        SCIP_STATUS SCIP_STATUS_MEMLIMIT
        SCIP_STATUS SCIP_STATUS_GAPLIMIT
        SCIP_STATUS SCIP_STATUS_SOLLIMIT
        SCIP_STATUS SCIP_STATUS_BESTSOLLIMIT
        SCIP_STATUS SCIP_STATUS_RESTARTLIMIT
        SCIP_STATUS SCIP_STATUS_PRIMALLIMIT
        SCIP_STATUS SCIP_STATUS_DUALLIMIT
        SCIP_STATUS SCIP_STATUS_OPTIMAL
        SCIP_STATUS SCIP_STATUS_INFEASIBLE
        SCIP_STATUS SCIP_STATUS_UNBOUNDED
        SCIP_STATUS SCIP_STATUS_INFORUNBD

    ctypedef int SCIP_STAGE
    cdef extern from "scip/type_set.h":
        SCIP_STAGE SCIP_STAGE_INIT
        SCIP_STAGE SCIP_STAGE_PROBLEM
        SCIP_STAGE SCIP_STAGE_TRANSFORMING
        SCIP_STAGE SCIP_STAGE_TRANSFORMED
        SCIP_STAGE SCIP_STAGE_INITPRESOLVE
        SCIP_STAGE SCIP_STAGE_PRESOLVING
        SCIP_STAGE SCIP_STAGE_EXITPRESOLVE
        SCIP_STAGE SCIP_STAGE_PRESOLVED
        SCIP_STAGE SCIP_STAGE_INITSOLVE
        SCIP_STAGE SCIP_STAGE_SOLVING
        SCIP_STAGE SCIP_STAGE_SOLVED
        SCIP_STAGE SCIP_STAGE_EXITSOLVE
        SCIP_STAGE SCIP_STAGE_FREETRANS
        SCIP_STAGE SCIP_STAGE_FREE

    ctypedef int SCIP_NODETYPE
    cdef extern from "scip/type_tree.h":
        SCIP_NODETYPE SCIP_NODETYPE_FOCUSNODE
        SCIP_NODETYPE SCIP_NODETYPE_PROBINGNODE
        SCIP_NODETYPE SCIP_NODETYPE_SIBLING
        SCIP_NODETYPE SCIP_NODETYPE_CHILD
        SCIP_NODETYPE SCIP_NODETYPE_LEAF
        SCIP_NODETYPE SCIP_NODETYPE_DEADEND
        SCIP_NODETYPE SCIP_NODETYPE_JUNCTION
        SCIP_NODETYPE SCIP_NODETYPE_PSEUDOFORK
        SCIP_NODETYPE SCIP_NODETYPE_FORK
        SCIP_NODETYPE SCIP_NODETYPE_SUBROOT
        SCIP_NODETYPE SCIP_NODETYPE_REFOCUSNODE

    ctypedef int SCIP_PARAMSETTING
    cdef extern from "scip/type_paramset.h":
        SCIP_PARAMSETTING SCIP_PARAMSETTING_DEFAULT
        SCIP_PARAMSETTING SCIP_PARAMSETTING_AGGRESSIVE
        SCIP_PARAMSETTING SCIP_PARAMSETTING_FAST
        SCIP_PARAMSETTING SCIP_PARAMSETTING_OFF

    ctypedef int SCIP_PARAMTYPE
    cdef extern from "scip/type_paramset.h":
        SCIP_PARAMTYPE SCIP_PARAMTYPE_BOOL
        SCIP_PARAMTYPE SCIP_PARAMTYPE_INT
        SCIP_PARAMTYPE SCIP_PARAMTYPE_LONGINT
        SCIP_PARAMTYPE SCIP_PARAMTYPE_REAL
        SCIP_PARAMTYPE SCIP_PARAMTYPE_CHAR
        SCIP_PARAMTYPE SCIP_PARAMTYPE_STRING

    ctypedef int SCIP_PARAMEMPHASIS
    cdef extern from "scip/type_paramset.h":
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_DEFAULT
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_CPSOLVER
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_EASYCIP
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_FEASIBILITY
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_HARDLP
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_OPTIMALITY
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_COUNTER
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_PHASEFEAS
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_PHASEIMPROVE
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_PHASEPROOF
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_NUMERICS
        SCIP_PARAMEMPHASIS SCIP_PARAMEMPHASIS_BENCHMARK

    ctypedef unsigned long SCIP_PROPTIMING
    cdef extern from "scip/type_timing.h":
        SCIP_PROPTIMING SCIP_PROPTIMING_BEFORELP
        SCIP_PROPTIMING SCIP_PROPTIMING_DURINGLPLOOP
        SCIP_PROPTIMING SCIP_PROPTIMING_AFTERLPLOOP
        SCIP_PROPTIMING SCIP_PROPTIMING_AFTERLPNODE

    ctypedef unsigned long SCIP_PRESOLTIMING
    cdef extern from "scip/type_timing.h":
        SCIP_PRESOLTIMING SCIP_PRESOLTIMING_NONE
        SCIP_PRESOLTIMING SCIP_PRESOLTIMING_FAST
        SCIP_PRESOLTIMING SCIP_PRESOLTIMING_MEDIUM
        SCIP_PRESOLTIMING SCIP_PRESOLTIMING_EXHAUSTIVE

    ctypedef unsigned long SCIP_HEURTIMING
    cdef extern from "scip/type_timing.h":
        SCIP_HEURTIMING SCIP_HEURTIMING_BEFORENODE
        SCIP_HEURTIMING SCIP_HEURTIMING_DURINGLPLOOP
        SCIP_HEURTIMING SCIP_HEURTIMING_AFTERLPLOOP
        SCIP_HEURTIMING SCIP_HEURTIMING_AFTERLPNODE
        SCIP_HEURTIMING SCIP_HEURTIMING_AFTERPSEUDONODE
        SCIP_HEURTIMING SCIP_HEURTIMING_AFTERLPPLUNGE
        SCIP_HEURTIMING SCIP_HEURTIMING_AFTERPSEUDOPLUNGE
        SCIP_HEURTIMING SCIP_HEURTIMING_DURINGPRICINGLOOP
        SCIP_HEURTIMING SCIP_HEURTIMING_BEFOREPRESOL
        SCIP_HEURTIMING SCIP_HEURTIMING_DURINGPRESOLLOOP
        SCIP_HEURTIMING SCIP_HEURTIMING_AFTERPROPLOOP

    ctypedef int SCIP_EXPR
    cdef extern from "scip/type_expr.h":
        SCIP_EXPR SCIP_EXPR_VARIDX
        SCIP_EXPR SCIP_EXPR_CONST
        SCIP_EXPR SCIP_EXPR_PARAM
        SCIP_EXPR SCIP_EXPR_PLUS
        SCIP_EXPR SCIP_EXPR_MINUS
        SCIP_EXPR SCIP_EXPR_MUL
        SCIP_EXPR SCIP_EXPR_DIV
        SCIP_EXPR SCIP_EXPR_SQUARE
        SCIP_EXPR SCIP_EXPR_SQRT
        SCIP_EXPR SCIP_EXPR_REALPOWER
        SCIP_EXPR SCIP_EXPR_INTPOWER
        SCIP_EXPR SCIP_EXPR_SIGNPOWER
        SCIP_EXPR SCIP_EXPR_EXP
        SCIP_EXPR SCIP_EXPR_LOG
        SCIP_EXPR SCIP_EXPR_SIN
        SCIP_EXPR SCIP_EXPR_COS
        SCIP_EXPR SCIP_EXPR_TAN
        SCIP_EXPR SCIP_EXPR_MIN
        SCIP_EXPR SCIP_EXPR_MAX
        SCIP_EXPR SCIP_EXPR_ABS
        SCIP_EXPR SCIP_EXPR_SIGN
        SCIP_EXPR SCIP_EXPR_SUM
        SCIP_EXPR SCIP_EXPR_PRODUCT
        SCIP_EXPR SCIP_EXPR_LINEAR
        SCIP_EXPR SCIP_EXPR_QUADRATIC
        SCIP_EXPR SCIP_EXPR_POLYNOMIAL
        SCIP_EXPR SCIP_EXPR_USER
        SCIP_EXPR SCIP_EXPR_LAST

    ctypedef int SCIP_BASESTAT
    cdef extern from "lpi/type_lpi.h":
        SCIP_BASESTAT SCIP_BASESTAT_LOWER
        SCIP_BASESTAT SCIP_BASESTAT_BASIC
        SCIP_BASESTAT SCIP_BASESTAT_UPPER
        SCIP_BASESTAT SCIP_BASESTAT_ZERO


    ctypedef int SCIP_LPPARAM
    cdef extern from "lpi/type_lpi.h":
        SCIP_LPPARAM SCIP_LPPAR_FROMSCRATCH
        SCIP_LPPARAM SCIP_LPPAR_FASTMIP
        SCIP_LPPARAM SCIP_LPPAR_SCALING
        SCIP_LPPARAM SCIP_LPPAR_PRESOLVING
        SCIP_LPPARAM SCIP_LPPAR_PRICING
        SCIP_LPPARAM SCIP_LPPAR_LPINFO
        SCIP_LPPARAM SCIP_LPPAR_FEASTOL
        SCIP_LPPARAM SCIP_LPPAR_DUALFEASTOL
        SCIP_LPPARAM SCIP_LPPAR_BARRIERCONVTOL
        SCIP_LPPARAM SCIP_LPPAR_OBJLIM
        SCIP_LPPARAM SCIP_LPPAR_LPITLIM
        SCIP_LPPARAM SCIP_LPPAR_LPTILIM
        SCIP_LPPARAM SCIP_LPPAR_MARKOWITZ
        SCIP_LPPARAM SCIP_LPPAR_ROWREPSWITCH
        SCIP_LPPARAM SCIP_LPPAR_THREADS
        SCIP_LPPARAM SCIP_LPPAR_CONDITIONLIMIT
        SCIP_LPPARAM SCIP_LPPAR_TIMING
        SCIP_LPPARAM SCIP_LPPAR_RANDOMSEED
        SCIP_LPPARAM SCIP_LPPAR_POLISHING
        SCIP_LPPARAM SCIP_LPPAR_REFACTOR

    ctypedef unsigned long SCIP_EVENTTYPE
    cdef extern from "scip/type_event.h":
        SCIP_EVENTTYPE SCIP_EVENTTYPE_DISABLED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_VARADDED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_VARDELETED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_VARFIXED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_VARUNLOCKED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_OBJCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_GLBCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_GUBCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LBTIGHTENED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LBRELAXED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_UBTIGHTENED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_UBRELAXED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_GHOLEADDED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_GHOLEREMOVED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LHOLEADDED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LHOLEREMOVED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_IMPLADDED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_PRESOLVEROUND
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODEFOCUSED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODEFEASIBLE
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODEINFEASIBLE
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODEBRANCHED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODEDELETE
        SCIP_EVENTTYPE SCIP_EVENTTYPE_FIRSTLPSOLVED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LPSOLVED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_POORSOLFOUND
        SCIP_EVENTTYPE SCIP_EVENTTYPE_BESTSOLFOUND
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWADDEDSEPA
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWDELETEDSEPA
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWADDEDLP
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWDELETEDLP
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWCOEFCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWCONSTCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWSIDECHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_SYNC
        SCIP_EVENTTYPE SCIP_EVENTTYPE_GBDCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LBCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_UBCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_BOUNDTIGHTENED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_BOUNDRELAXED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_BOUNDCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_GHOLECHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LHOLECHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_HOLECHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_DOMCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_VARCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_VAREVENT
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODESOLVED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_NODEEVENT
        SCIP_EVENTTYPE SCIP_EVENTTYPE_LPEVENT
        SCIP_EVENTTYPE SCIP_EVENTTYPE_SOLFOUND
        SCIP_EVENTTYPE SCIP_EVENTTYPE_SOLEVENT
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWCHANGED
        SCIP_EVENTTYPE SCIP_EVENTTYPE_ROWEVENT

    ctypedef int SCIP_LPSOLQUALITY
    cdef extern from "lpi/type_lpi.h":
        SCIP_LPSOLQUALITY SCIP_LPSOLQUALITY_ESTIMCONDITION
        SCIP_LPSOLQUALITY SCIP_LPSOLQUALITY_EXACTCONDITION

    ctypedef int SCIP_LOCKTYPE
    cdef extern from "scip/type_var.h":
        SCIP_LOCKTYPE SCIP_LOCKTYPE_MODEL
        SCIP_LOCKTYPE SCIP_LOCKTYPE_CONFLICT

    ctypedef int SCIP_BENDERSENFOTYPE
    cdef extern from "scip/type_benders.h":
        SCIP_BENDERSENFOTYPE SCIP_BENDERSENFOTYPE_LP
        SCIP_BENDERSENFOTYPE SCIP_BENDERSENFOTYPE_RELAX
        SCIP_BENDERSENFOTYPE SCIP_BENDERSENFOTYPE_PSEUDO
        SCIP_BENDERSENFOTYPE SCIP_BENDERSENFOTYPE_CHECK

    ctypedef int SCIP_LPSOLSTAT
    cdef extern from "scip/type_lp.h":
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_NOTSOLVED
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_OPTIMAL
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_INFEASIBLE
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_UNBOUNDEDRAY
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_OBJLIMIT
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_ITERLIMIT
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_TIMELIMIT
        SCIP_LPSOLSTAT SCIP_LPSOLSTAT_ERROR

    ctypedef int SCIP_BRANCHDIR
    cdef extern from "scip/type_history.h":
        SCIP_BRANCHDIR SCIP_BRANCHDIR_DOWNWARDS
        SCIP_BRANCHDIR SCIP_BRANCHDIR_UPWARDS
        SCIP_BRANCHDIR SCIP_BRANCHDIR_FIXED
        SCIP_BRANCHDIR SCIP_BRANCHDIR_AUTO

    ctypedef int SCIP_BOUNDCHGTYPE
    cdef extern from "scip/type_var.h":
        SCIP_BOUNDCHGTYPE SCIP_BOUNDCHGTYPE_BRANCHING
        SCIP_BOUNDCHGTYPE SCIP_BOUNDCHGTYPE_CONSINFER
        SCIP_BOUNDCHGTYPE SCIP_BOUNDCHGTYPE_PROPINFER

    ctypedef int SCIP_ROWORIGINTYPE
    cdef extern from "scip/type_lp.h":
        SCIP_ROWORIGINTYPE SCIP_ROWORIGINTYPE_UNSPEC
        SCIP_ROWORIGINTYPE SCIP_ROWORIGINTYPE_CONS
        SCIP_ROWORIGINTYPE SCIP_ROWORIGINTYPE_SEPA
        SCIP_ROWORIGINTYPE SCIP_ROWORIGINTYPE_REOPT

    ctypedef int SCIP_SOLORIGIN
    cdef extern from "scip/type_sol.h":
        SCIP_SOLORIGIN SCIP_SOLORIGIN_ORIGINAL
        SCIP_SOLORIGIN SCIP_SOLORIGIN_ZERO
        SCIP_SOLORIGIN SCIP_SOLORIGIN_LPSOL
        SCIP_SOLORIGIN SCIP_SOLORIGIN_NLPSOL
        SCIP_SOLORIGIN SCIP_SOLORIGIN_RELAXSOL
        SCIP_SOLORIGIN SCIP_SOLORIGIN_PSEUDOSOL
        SCIP_SOLORIGIN SCIP_SOLORIGIN_PARTIAL
        SCIP_SOLORIGIN SCIP_SOLORIGIN_UNKNOWN

    ctypedef bint SCIP_Bool

    ctypedef long long SCIP_Longint

    ctypedef double SCIP_Real

    ctypedef struct SCIP:
        pass

    ctypedef struct SCIP_VAR:
        pass

    ctypedef struct SCIP_CONS:
        pass

    ctypedef struct SCIP_ROW:
        pass

    ctypedef struct SCIP_NLROW:
        pass

    ctypedef struct SCIP_COL:
        pass

    ctypedef struct SCIP_SOL:
        pass

    ctypedef struct FILE:
        pass

    ctypedef struct SYM_GRAPH:
        pass

    ctypedef struct SCIP_READER:
        pass

    ctypedef struct SCIP_READERDATA:
        pass

    ctypedef struct SCIP_PROBDATA:
        pass

    ctypedef struct SCIP_PRICER:
        pass

    ctypedef struct SCIP_PRICERDATA:
        pass

    ctypedef struct SCIP_PROP:
        pass

    ctypedef struct SCIP_PROPDATA:
        pass

    ctypedef struct SCIP_PROPTIMING:
        pass

    ctypedef struct SCIP_PRESOLTIMING:
        pass

    ctypedef struct SCIP_PRESOL:
        pass

    ctypedef struct SCIP_PRESOLDATA:
        pass

    ctypedef struct SCIP_HEUR:
        pass

    ctypedef struct SCIP_HEURDATA:
        pass

    ctypedef struct SCIP_RELAX:
        pass

    ctypedef struct SCIP_RELAXDATA:
        pass

    ctypedef struct SCIP_NODE:
        pass

    ctypedef struct SCIP_NODESEL:
        pass

    ctypedef struct SCIP_NODESELDATA:
        pass

    ctypedef struct SCIP_BRANCHRULE:
        pass

    ctypedef struct SCIP_BRANCHRULEDATA:
        pass

    ctypedef struct SCIP_CUTSEL:
        pass

    ctypedef struct SCIP_CUTSELDATA:
        pass

    ctypedef struct SCIP_PRESOL:
        pass

    ctypedef struct SCIP_HEURTIMING:
        pass

    ctypedef struct SCIP_SEPA:
        pass

    ctypedef struct SCIP_SEPADATA:
        pass

    ctypedef struct SCIP_CONSHDLR:
        pass

    ctypedef struct SCIP_CONSHDLRDATA:
        pass

    ctypedef struct SCIP_CONSDATA:
        pass

    ctypedef struct SCIP_VARDATA:
        pass

    ctypedef struct SCIP_EVENT:
        pass

    ctypedef struct SCIP_EVENTDATA:
        pass

    ctypedef struct SCIP_EVENTHDLR:
        pass

    ctypedef struct SCIP_EVENTHDLRDATA:
        pass

    ctypedef struct SCIP_DIVESET:
        pass

    ctypedef struct SCIP_HASHMAP:
        pass

    ctypedef struct SCIP_BDCHGIDX:
        pass

    ctypedef struct SCIP_MESSAGEHDLR:
        pass

    ctypedef struct SCIP_MESSAGEHDLRDATA:
        pass

    ctypedef struct SCIP_LPI:
        pass

    ctypedef struct BMS_BLKMEM:
        pass

    ctypedef struct SCIP_EXPR:
        pass

    ctypedef struct SCIP_EXPRHDLR:
        pass

    ctypedef struct SCIP_EXPRDATA:
        pass

    ctypedef struct SCIP_DECL_EXPR_OWNERCREATE:
        pass

    ctypedef struct SCIP_BENDERS:
        pass

    ctypedef struct SCIP_BENDERSDATA:
        pass

    ctypedef struct SCIP_BENDERSCUT:
        pass

    ctypedef struct SCIP_BENDERSCUTDATA:
        pass

    ctypedef struct SCIP_BOUNDCHG:
        pass

    ctypedef union SCIP_DOMCHG:
        pass

    ctypedef void (*messagecallback) (SCIP_MESSAGEHDLR* messagehdlr, FILE* file, const char* msg) noexcept
    ctypedef void (*errormessagecallback) (void* data, FILE* file, const char* msg)
    ctypedef SCIP_RETCODE (*messagehdlrfree) (SCIP_MESSAGEHDLR* messagehdlr)

    # General SCIP Methods
    SCIP_RETCODE SCIPcreate(SCIP** scip)
    SCIP_RETCODE SCIPfree(SCIP** scip)
    SCIP_RETCODE SCIPcopy(SCIP*                 sourcescip,
                          SCIP*                 targetscip,
                          SCIP_HASHMAP*         varmap,
                          SCIP_HASHMAP*         consmap,
                          const char*           suffix,
                          SCIP_Bool             globalcopy,
                          SCIP_Bool             enablepricing,
                          SCIP_Bool             threadsafe,
                          SCIP_Bool             passmessagehdlr,
                          SCIP_Bool*            valid)
    SCIP_RETCODE SCIPcopyOrig(SCIP*                 sourcescip,
                              SCIP*                 targetscip,
                              SCIP_HASHMAP*         varmap,
                              SCIP_HASHMAP*         consmap,
                              const char*           suffix,
                              SCIP_Bool             enablepricing,
                              SCIP_Bool             threadsafe,
                              SCIP_Bool             passmessagehdlr,
                              SCIP_Bool*            valid)
    SCIP_RETCODE SCIPcopyOrigVars(SCIP* sourcescip, SCIP* targetscip, SCIP_HASHMAP* varmap, SCIP_HASHMAP* consmap, SCIP_VAR** fixedvars, SCIP_Real* fixedvals, int nfixedvars )
    SCIP_RETCODE SCIPcopyOrigConss(SCIP* sourcescip, SCIP* targetscip, SCIP_HASHMAP* varmap, SCIP_HASHMAP* consmap, SCIP_Bool enablepricing, SCIP_Bool* valid)
    SCIP_RETCODE SCIPmessagehdlrCreate(SCIP_MESSAGEHDLR** messagehdlr,
                                       SCIP_Bool bufferedoutput,
                                       const char* filename,
                                       SCIP_Bool quiet,
                                       messagecallback,
                                       messagecallback,
                                       messagecallback,
                                       messagehdlrfree,
                                       SCIP_MESSAGEHDLRDATA* messagehdlrdata)

    SCIP_RETCODE SCIPsetMessagehdlr(SCIP* scip, SCIP_MESSAGEHDLR* messagehdlr)
    void SCIPsetMessagehdlrQuiet(SCIP* scip, SCIP_Bool quiet)
    void SCIPmessageSetErrorPrinting(errormessagecallback, void* data)
    void SCIPsetMessagehdlrLogfile(SCIP* scip, const char* filename)
    SCIP_Real SCIPversion()
    int SCIPmajorVersion()
    int SCIPminorVersion()
    int SCIPtechVersion()
    void SCIPprintVersion(SCIP* scip, FILE* outfile)
    void SCIPprintExternalCodes(SCIP* scip, FILE* outfile)
    SCIP_Real SCIPgetTotalTime(SCIP* scip)
    SCIP_Real SCIPgetSolvingTime(SCIP* scip)
    SCIP_Real SCIPgetReadingTime(SCIP* scip)
    SCIP_Real SCIPgetPresolvingTime(SCIP* scip)
    SCIP_STAGE SCIPgetStage(SCIP* scip)
    SCIP_RETCODE SCIPsetProbName(SCIP* scip, char* name)
    const char* SCIPgetProbName(SCIP* scip)

    # Diving methods
    SCIP_RETCODE SCIPstartDive(SCIP* scip)
    SCIP_RETCODE SCIPchgVarObjDive(SCIP* scip, SCIP_VAR* var, SCIP_Real newobj)
    SCIP_RETCODE SCIPchgVarLbDive(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarUbDive(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_Real SCIPgetVarLbDive(SCIP* scip, SCIP_VAR* var)
    SCIP_Real SCIPgetVarUbDive(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPsolveDiveLP(SCIP* scip, int itlim, SCIP_Bool* lperror, SCIP_Bool* cutoff)
    SCIP_RETCODE SCIPchgRowLhsDive(SCIP* scip, SCIP_ROW* row, SCIP_Real newlhs)
    SCIP_RETCODE SCIPchgRowRhsDive(SCIP* scip, SCIP_ROW* row, SCIP_Real newrhs)
    SCIP_RETCODE SCIPaddRowDive(SCIP* scip, SCIP_ROW* row)
    SCIP_RETCODE SCIPendDive(SCIP* scip)

    # Probing methods
    SCIP_RETCODE SCIPstartProbing(SCIP* scip)
    SCIP_RETCODE SCIPnewProbingNode(SCIP* scip)
    SCIP_RETCODE SCIPgetProbingDepth(SCIP* scip)
    SCIP_RETCODE SCIPbacktrackProbing(SCIP* scip, int probingdepth)
    SCIP_RETCODE SCIPchgVarObjProbing(SCIP* scip, SCIP_VAR* var, SCIP_Real newobj)
    SCIP_RETCODE SCIPchgVarUbProbing(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarLbProbing(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPsolveProbingLP(SCIP* scip, int itlim, SCIP_Bool* lperror, SCIP_Bool* cutoff)
    SCIP_RETCODE SCIPendProbing(SCIP* scip)
    SCIP_RETCODE SCIPfixVarProbing(SCIP* scip, SCIP_VAR* var, SCIP_Real fixedval)
    SCIP_Bool SCIPisObjChangedProbing(SCIP* scip)
    SCIP_Bool SCIPinProbing(SCIP* scip)
    SCIP_RETCODE SCIPapplyCutsProbing(SCIP* scip, SCIP_Bool* cutoff)
    SCIP_RETCODE SCIPpropagateProbing(SCIP* scip, int maxproprounds, SCIP_Bool* cutoff, SCIP_Longint* ndomredsfound)


    # Event Methods
    SCIP_RETCODE SCIPcatchEvent(SCIP* scip,
                                SCIP_EVENTTYPE eventtype,
                                SCIP_EVENTHDLR* eventhdlr,
                                SCIP_EVENTDATA* eventdata,
                                int* filterpos)
    SCIP_RETCODE SCIPdropEvent(SCIP* scip,
                               SCIP_EVENTTYPE eventtype,
                               SCIP_EVENTHDLR* eventhdlr,
                               SCIP_EVENTDATA* eventdata,
                               int filterpos)
    SCIP_RETCODE SCIPcatchVarEvent(SCIP* scip,
                                   SCIP_VAR* var,
                                   SCIP_EVENTTYPE eventtype,
                                   SCIP_EVENTHDLR* eventhdlr,
                                   SCIP_EVENTDATA* eventdata,
                                   int* filterpos)
    SCIP_RETCODE SCIPdropVarEvent(SCIP* scip,
                                  SCIP_VAR* var,
                                  SCIP_EVENTTYPE eventtype,
                                  SCIP_EVENTHDLR* eventhdlr,
                                  SCIP_EVENTDATA* eventdata,
                                  int filterpos)
    SCIP_RETCODE SCIPcatchRowEvent(SCIP* scip,
                                   SCIP_ROW* row,
                                   SCIP_EVENTTYPE eventtype,
                                   SCIP_EVENTHDLR* eventhdlr,
                                   SCIP_EVENTDATA* eventdata,
                                   int* filterpos)
    SCIP_RETCODE SCIPdropRowEvent(SCIP* scip,
                                  SCIP_ROW* row,
                                  SCIP_EVENTTYPE eventtype,
                                  SCIP_EVENTHDLR* eventhdlr,
                                  SCIP_EVENTDATA* eventdata,
                                  int filterpos)
    SCIP_EVENTHDLR* SCIPfindEventhdlr(SCIP* scip, const char* name)
    SCIP_EVENTTYPE SCIPeventGetType(SCIP_EVENT* event)
    SCIP_Real SCIPeventGetNewbound(SCIP_EVENT* event)
    SCIP_Real SCIPeventGetOldbound(SCIP_EVENT* event)
    SCIP_VAR* SCIPeventGetVar(SCIP_EVENT* event)
    SCIP_NODE* SCIPeventGetNode(SCIP_EVENT* event)
    SCIP_ROW* SCIPeventGetRow(SCIP_EVENT* event)
    SCIP_RETCODE SCIPinterruptSolve(SCIP* scip)
    SCIP_RETCODE SCIPrestartSolve(SCIP* scip)


    # Global Problem Methods
    SCIP_RETCODE SCIPcreateProbBasic(SCIP* scip, char* name)
    SCIP_RETCODE SCIPfreeProb(SCIP* scip)
    SCIP_RETCODE SCIPaddVar(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPdelVar(SCIP* scip, SCIP_VAR* var, SCIP_Bool* deleted)
    SCIP_RETCODE SCIPaddCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPdelCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_CONS**  SCIPgetOrigConss(SCIP* scip)
    int          SCIPgetNOrigConss(SCIP* scip)
    SCIP_CONS*   SCIPfindOrigCons(SCIP* scip, const char*)
    SCIP_CONS*   SCIPfindCons(SCIP* scip, const char*)
    SCIP_RETCODE SCIPsetObjsense(SCIP* scip, SCIP_OBJSENSE objsense)
    SCIP_OBJSENSE SCIPgetObjsense(SCIP* scip)
    SCIP_RETCODE SCIPsetObjlimit(SCIP* scip, SCIP_Real objlimit)
    SCIP_Real SCIPgetObjlimit(SCIP* scip)
    SCIP_Real SCIPgetObjNorm(SCIP* scip)
    SCIP_RETCODE SCIPaddObjoffset(SCIP* scip, SCIP_Real addval)
    SCIP_RETCODE SCIPaddOrigObjoffset(SCIP* scip, SCIP_Real addval)
    SCIP_Real SCIPgetOrigObjoffset(SCIP* scip)
    SCIP_Real SCIPgetTransObjoffset(SCIP* scip)
    SCIP_RETCODE SCIPsetPresolving(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPsetSeparating(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPsetHeuristics(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPsetRelaxation(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPwriteOrigProblem(SCIP* scip, char* filename, char* extension, SCIP_Bool genericnames)
    SCIP_RETCODE SCIPwriteTransProblem(SCIP* scip, char* filename, char* extension, SCIP_Bool genericnames)
    SCIP_RETCODE SCIPwriteLP(SCIP* scip, const char*)
    SCIP_STATUS SCIPgetStatus(SCIP* scip)
    SCIP_Real SCIPepsilon(SCIP* scip)
    SCIP_Real SCIPfeastol(SCIP* scip)
    SCIP_RETCODE SCIPsetObjIntegral(SCIP* scip)
    SCIP_Real SCIPgetLocalOrigEstimate(SCIP* scip)
    SCIP_Real SCIPgetLocalTransEstimate(SCIP* scip)

    # Solve Methods
    SCIP_RETCODE SCIPsolve(SCIP* scip)
    SCIP_RETCODE SCIPsolve(SCIP* scip) noexcept nogil
    SCIP_RETCODE SCIPsolveConcurrent(SCIP* scip)
    SCIP_RETCODE SCIPfreeTransform(SCIP* scip)
    SCIP_RETCODE SCIPpresolve(SCIP* scip)

    # Node Methods
    SCIP_NODE* SCIPgetCurrentNode(SCIP* scip)
    SCIP_NODE* SCIPnodeGetParent(SCIP_NODE* node)
    SCIP_Longint SCIPnodeGetNumber(SCIP_NODE* node)
    int SCIPnodeGetDepth(SCIP_NODE* node)
    SCIP_Real SCIPnodeGetLowerbound(SCIP_NODE* node)
    SCIP_RETCODE SCIPupdateNodeLowerbound(SCIP* scip, SCIP_NODE* node, SCIP_Real newbound)
    SCIP_Real SCIPnodeGetEstimate(SCIP_NODE* node)
    SCIP_NODETYPE SCIPnodeGetType(SCIP_NODE* node)
    SCIP_Bool SCIPnodeIsActive(SCIP_NODE* node)
    SCIP_Bool SCIPnodeIsPropagatedAgain(SCIP_NODE* node)
    SCIP_Real SCIPcalcNodeselPriority(SCIP*	scip, SCIP_VAR* var, SCIP_BRANCHDIR	branchdir, SCIP_Real targetvalue)
    SCIP_Real SCIPcalcChildEstimate(SCIP* scip, SCIP_VAR* var, SCIP_Real targetvalue)
    SCIP_RETCODE SCIPcreateChild(SCIP* scip, SCIP_NODE** node, SCIP_Real nodeselprio, SCIP_Real estimate)
    SCIP_Bool SCIPinRepropagation(SCIP* scip)
    SCIP_RETCODE SCIPaddConsNode(SCIP* scip, SCIP_NODE* node, SCIP_CONS* cons, SCIP_NODE* validnode)
    SCIP_RETCODE SCIPaddConsLocal(SCIP* scip, SCIP_CONS* cons, SCIP_NODE* validnode)
    void SCIPnodeGetParentBranchings(SCIP_NODE* node,
                                     SCIP_VAR** branchvars,
                                     SCIP_Real* branchbounds,
                                     SCIP_BOUNDTYPE* boundtypes,
                                     int* nbranchvars,
                                     int branchvarssize)
    void SCIPnodeGetAddedConss(SCIP_NODE* node, SCIP_CONS** addedconss,
                               int* naddedconss, int addedconsssize)
    void SCIPnodeGetNDomchg(SCIP_NODE* node, int* nbranchings, int* nconsprop,
                            int* nprop)
    SCIP_DOMCHG* SCIPnodeGetDomchg(SCIP_NODE* node)

    # Domain change methods
    int SCIPdomchgGetNBoundchgs(SCIP_DOMCHG* domchg)
    SCIP_BOUNDCHG* SCIPdomchgGetBoundchg(SCIP_DOMCHG* domchg, int pos)

    # Bound change methods
    SCIP_Real SCIPboundchgGetNewbound(SCIP_BOUNDCHG* boundchg)
    SCIP_VAR* SCIPboundchgGetVar(SCIP_BOUNDCHG* boundchg)
    SCIP_BOUNDCHGTYPE SCIPboundchgGetBoundchgtype(SCIP_BOUNDCHG* boundchg)
    SCIP_BOUNDTYPE SCIPboundchgGetBoundtype(SCIP_BOUNDCHG* boundchg)
    SCIP_Bool SCIPboundchgIsRedundant(SCIP_BOUNDCHG* boundchg)

    # Variable Methods
    SCIP_RETCODE SCIPcreateVarBasic(SCIP* scip,
                                    SCIP_VAR** var,
                                    char* name,
                                    SCIP_Real lb,
                                    SCIP_Real ub,
                                    SCIP_Real obj,
                                    SCIP_VARTYPE vartype)
    SCIP_RETCODE SCIPchgVarObj(SCIP* scip, SCIP_VAR* var, SCIP_Real newobj)
    SCIP_RETCODE SCIPchgVarLb(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarUb(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarLbGlobal(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarUbGlobal(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarLbNode(SCIP* scip, SCIP_NODE* node, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPchgVarUbNode(SCIP* scip, SCIP_NODE* node, SCIP_VAR* var, SCIP_Real newbound)
    SCIP_RETCODE SCIPtightenVarLb(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound,
                                  SCIP_Bool force, SCIP_Bool* infeasible, SCIP_Bool* tightened)
    SCIP_RETCODE SCIPtightenVarUb(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound,
                                  SCIP_Bool force, SCIP_Bool* infeasible, SCIP_Bool* tightened)
    SCIP_RETCODE SCIPtightenVarLbGlobal(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound,
                                        SCIP_Bool force, SCIP_Bool* infeasible, SCIP_Bool* tightened)
    SCIP_RETCODE SCIPtightenVarUbGlobal(SCIP* scip, SCIP_VAR* var, SCIP_Real newbound,
                                        SCIP_Bool force, SCIP_Bool* infeasible, SCIP_Bool* tightened)
    SCIP_RETCODE SCIPfixVar(SCIP* scip, SCIP_VAR* var, SCIP_Real fixedval, SCIP_Bool* infeasible, SCIP_Bool* fixed)
    SCIP_RETCODE SCIPdelVar(SCIP* scip, SCIP_VAR* var, SCIP_Bool* deleted)

    SCIP_RETCODE SCIPchgVarType(SCIP* scip, SCIP_VAR* var, SCIP_VARTYPE vartype, SCIP_Bool* infeasible)
    SCIP_RETCODE SCIPcaptureVar(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPaddPricedVar(SCIP* scip, SCIP_VAR* var, SCIP_Real score)
    SCIP_RETCODE SCIPreleaseVar(SCIP* scip, SCIP_VAR** var)
    SCIP_RETCODE SCIPtransformVar(SCIP* scip, SCIP_VAR* var, SCIP_VAR** transvar)
    SCIP_RETCODE SCIPgetTransformedVar(SCIP* scip, SCIP_VAR* var, SCIP_VAR** transvar)
    SCIP_RETCODE SCIPaddVarLocks(SCIP* scip, SCIP_VAR* var, int nlocksdown, int nlocksup)
    SCIP_RETCODE SCIPaddVarLocksType(SCIP* scip, SCIP_VAR* var, SCIP_LOCKTYPE locktype, int nlocksdown, int nlocksup)
    int SCIPvarGetNLocksDown(SCIP_VAR* var)
    int SCIPvarGetNLocksUp(SCIP_VAR* var)
    int SCIPvarGetNLocksDownType(SCIP_VAR* var, SCIP_LOCKTYPE locktype)
    int SCIPvarGetNLocksUpType(SCIP_VAR* var, SCIP_LOCKTYPE locktype)
    SCIP_VAR** SCIPgetVars(SCIP* scip)
    SCIP_VAR** SCIPgetOrigVars(SCIP* scip)
    const char* SCIPvarGetName(SCIP_VAR* var)
    int SCIPvarGetIndex(SCIP_VAR* var)
    int SCIPgetNVars(SCIP* scip)
    int SCIPgetNOrigVars(SCIP* scip)
    int SCIPgetNIntVars(SCIP* scip)
    int SCIPgetNBinVars(SCIP* scip)
    int SCIPgetNImplVars(SCIP* scip)
    int SCIPgetNContVars(SCIP* scip)
    SCIP_VARTYPE SCIPvarGetType(SCIP_VAR* var)
    SCIP_Bool SCIPvarIsOriginal(SCIP_VAR* var)
    SCIP_Bool SCIPvarIsTransformed(SCIP_VAR* var)
    SCIP_COL* SCIPvarGetCol(SCIP_VAR* var)
    SCIP_Bool SCIPvarIsInLP(SCIP_VAR* var)
    SCIP_Real SCIPvarGetLbOriginal(SCIP_VAR* var)
    SCIP_Real SCIPvarGetUbOriginal(SCIP_VAR* var)
    SCIP_Real SCIPvarGetLbGlobal(SCIP_VAR* var)
    SCIP_Real SCIPvarGetUbGlobal(SCIP_VAR* var)
    SCIP_Real SCIPvarGetLbLocal(SCIP_VAR* var)
    SCIP_Real SCIPvarGetUbLocal(SCIP_VAR* var)
    SCIP_Real SCIPvarGetObj(SCIP_VAR* var)
    SCIP_Real SCIPvarGetLPSol(SCIP_VAR* var)
    void SCIPvarSetData(SCIP_VAR* var, SCIP_VARDATA* vardata)
    SCIP_VARDATA* SCIPvarGetData(SCIP_VAR* var)
    SCIP_Real SCIPvarGetAvgSol(SCIP_VAR* var)
    SCIP_Real SCIPgetVarPseudocost(SCIP* scip, SCIP_VAR* var, SCIP_BRANCHDIR dir)
    SCIP_Real SCIPvarGetCutoffSum(SCIP_VAR* var, SCIP_BRANCHDIR dir)
    SCIP_Longint SCIPvarGetNBranchings(SCIP_VAR* var, SCIP_BRANCHDIR dir)
    SCIP_Bool SCIPvarMayRoundUp(SCIP_VAR* var)
    SCIP_Bool SCIPvarMayRoundDown(SCIP_VAR* var)

    # LP Methods
    SCIP_RETCODE SCIPgetLPColsData(SCIP* scip, SCIP_COL*** cols, int* ncols)
    SCIP_RETCODE SCIPgetLPRowsData(SCIP* scip, SCIP_ROW*** rows, int* nrows)
    SCIP_RETCODE SCIPgetLPBasisInd(SCIP* scip, int* basisind)
    SCIP_RETCODE SCIPgetLPBInvRow(SCIP* scip, int r, SCIP_Real* coefs, int* inds, int* ninds)
    SCIP_RETCODE SCIPgetLPBInvARow(SCIP* scip, int r, SCIP_Real* binvrow, SCIP_Real* coefs, int* inds, int* ninds)
    SCIP_RETCODE SCIPconstructLP(SCIP* scip, SCIP_Bool* cutoff)
    SCIP_Real SCIPgetLPObjval(SCIP* scip)
    SCIP_Bool SCIPisLPSolBasic(SCIP* scip)
    SCIP_LPSOLSTAT SCIPgetLPSolstat(SCIP* scip)
    int SCIPgetNLPRows(SCIP* scip)
    int SCIPgetNLPCols(SCIP* scip)
    SCIP_COL** SCIPgetLPCols(SCIP* scip)
    SCIP_ROW** SCIPgetLPRows(SCIP* scip)
    SCIP_Bool SCIPallColsInLP(SCIP* scip)

    # Cutting Plane Methods
    SCIP_RETCODE SCIPaddPoolCut(SCIP* scip, SCIP_ROW* row)
    SCIP_Real SCIPgetCutEfficacy(SCIP* scip, SCIP_SOL* sol, SCIP_ROW* cut)
    SCIP_Bool SCIPisCutEfficacious(SCIP* scip, SCIP_SOL* sol, SCIP_ROW* cut)
    SCIP_Real SCIPgetCutLPSolCutoffDistance(SCIP* scip, SCIP_SOL* sol, SCIP_ROW* cut)
    int SCIPgetNCuts(SCIP* scip)
    int SCIPgetNCutsApplied(SCIP* scip)
    SCIP_RETCODE SCIPseparateSol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool pretendroot, SCIP_Bool allowlocal, SCIP_Bool onlydelayed, SCIP_Bool* delayed, SCIP_Bool* cutoff)

    # Constraint Methods
    SCIP_RETCODE SCIPcaptureCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPreleaseCons(SCIP* scip, SCIP_CONS** cons)
    SCIP_RETCODE SCIPtransformCons(SCIP* scip, SCIP_CONS* cons, SCIP_CONS** transcons)
    SCIP_RETCODE SCIPgetTransformedCons(SCIP* scip, SCIP_CONS* cons, SCIP_CONS** transcons)
    SCIP_RETCODE SCIPgetConsVars(SCIP* scip, SCIP_CONS* cons, SCIP_VAR** vars, int varssize, SCIP_Bool* success)
    SCIP_RETCODE SCIPgetConsNVars(SCIP* scip, SCIP_CONS* cons, int* nvars, SCIP_Bool* success)
    SCIP_CONS** SCIPgetConss(SCIP* scip)
    const char* SCIPconsGetName(SCIP_CONS* cons)
    int SCIPgetNConss(SCIP* scip)
    SCIP_Bool SCIPconsIsOriginal(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsTransformed(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsInitial(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsSeparated(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsEnforced(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsChecked(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsPropagated(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsLocal(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsModifiable(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsDynamic(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsRemovable(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsStickingAtNode(SCIP_CONS* cons)
    SCIP_Bool SCIPconsIsActive(SCIP_CONS* cons)
    SCIP_CONSDATA* SCIPconsGetData(SCIP_CONS* cons)
    SCIP_CONSHDLR* SCIPconsGetHdlr(SCIP_CONS* cons)
    const char* SCIPconshdlrGetName(SCIP_CONSHDLR* conshdlr)
    SCIP_RETCODE SCIPdelConsLocal(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPdelCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPsetConsChecked(SCIP* scip, SCIP_CONS* cons, SCIP_Bool check)
    SCIP_RETCODE SCIPsetConsRemovable(SCIP* scip, SCIP_CONS* cons, SCIP_Bool removable)
    SCIP_RETCODE SCIPsetConsInitial(SCIP* scip, SCIP_CONS* cons, SCIP_Bool initial)
    SCIP_RETCODE SCIPsetConsModifiable(SCIP* scip, SCIP_CONS* cons, SCIP_Bool modifiable)
    SCIP_RETCODE SCIPsetConsEnforced(SCIP* scip, SCIP_CONS* cons, SCIP_Bool enforce)

    # Primal Solution Methods
    SCIP_SOL** SCIPgetSols(SCIP* scip)
    int SCIPgetNSols(SCIP* scip)
    int SCIPgetNSolsFound(SCIP* scip)
    int SCIPgetNLimSolsFound(SCIP* scip)
    int SCIPgetNBestSolsFound(SCIP* scip)
    SCIP_SOL* SCIPgetBestSol(SCIP* scip)
    SCIP_Real SCIPgetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var)
    SCIP_Bool SCIPsolIsOriginal(SCIP_SOL* sol)
    SCIP_RETCODE SCIPwriteVarName(SCIP* scip, FILE* outfile, SCIP_VAR* var, SCIP_Bool vartype)
    SCIP_Real SCIPgetSolOrigObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_Real SCIPgetSolTransObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_RETCODE SCIPcreateSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPcreatePartialSol(SCIP* scip, SCIP_SOL** sol,SCIP_HEUR* heur)
    SCIP_RETCODE SCIPcreateOrigSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPcreateLPSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPsetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var, SCIP_Real val)
    SCIP_RETCODE SCIPtrySolFree(SCIP* scip, SCIP_SOL** sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* stored)
    SCIP_RETCODE SCIPtrySol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* stored)
    SCIP_RETCODE SCIPfreeSol(SCIP* scip, SCIP_SOL** sol)
    SCIP_RETCODE SCIPprintBestSol(SCIP* scip, FILE* outfile, SCIP_Bool printzeros)
    SCIP_RETCODE SCIPprintBestTransSol(SCIP* scip, FILE* outfile, SCIP_Bool printzeros)
    SCIP_RETCODE SCIPprintSol(SCIP* scip, SCIP_SOL* sol, FILE* outfile, SCIP_Bool printzeros)
    SCIP_RETCODE SCIPprintTransSol(SCIP* scip, SCIP_SOL* sol, FILE* outfile, SCIP_Bool printzeros)
    SCIP_Real SCIPgetPrimalbound(SCIP* scip)
    SCIP_Real SCIPgetGap(SCIP* scip)
    int SCIPgetDepth(SCIP* scip)
    SCIP_RETCODE SCIPcutoffNode(SCIP* scip, SCIP_NODE* node)
    SCIP_Bool SCIPhasPrimalRay(SCIP* scip)
    SCIP_Real SCIPgetPrimalRayVal(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPaddSolFree(SCIP* scip, SCIP_SOL** sol, SCIP_Bool* stored)
    SCIP_RETCODE SCIPaddSol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool* stored)
    SCIP_RETCODE SCIPreadSol(SCIP* scip, const char* filename)
    SCIP_RETCODE SCIPreadSolFile(SCIP* scip, const char* filename, SCIP_SOL* sol, SCIP_Bool xml, SCIP_Bool*	partial, SCIP_Bool*	error)
    SCIP_RETCODE SCIPcheckSol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* feasible)
    SCIP_RETCODE SCIPcheckSolOrig(SCIP* scip, SCIP_SOL* sol, SCIP_Bool* feasible, SCIP_Bool printreason, SCIP_Bool completely)
    SCIP_RETCODE SCIPretransformSol(SCIP* scip, SCIP_SOL* sol)
    SCIP_RETCODE SCIPtranslateSubSol(SCIP* scip, SCIP* subscip, SCIP_SOL* subsol, SCIP_HEUR* heur, SCIP_VAR** subvars, SCIP_SOL** newsol)
    SCIP_SOLORIGIN SCIPsolGetOrigin(SCIP_SOL* sol)
    SCIP_Real SCIPgetSolTime(SCIP* scip, SCIP_SOL* sol)

    SCIP_RETCODE SCIPsetRelaxSolVal(SCIP* scip, SCIP_RELAX* relax, SCIP_VAR* var, SCIP_Real val)

    # Row Methods
    SCIP_RETCODE SCIPcreateRow(SCIP* scip, SCIP_ROW** row, const char* name, int len, SCIP_COL** cols, SCIP_Real* vals,
                               SCIP_Real lhs, SCIP_Real rhs, SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool removable)
    SCIP_RETCODE SCIPaddRow(SCIP* scip, SCIP_ROW* row, SCIP_Bool forcecut, SCIP_Bool* infeasible)
    SCIP_RETCODE SCIPcreateEmptyRowSepa(SCIP* scip, SCIP_ROW** row, SCIP_SEPA* sepa, const char* name, SCIP_Real lhs, SCIP_Real rhs, SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool removable)
    SCIP_RETCODE SCIPcreateEmptyRowUnspec(SCIP* scip, SCIP_ROW** row, const char* name, SCIP_Real lhs, SCIP_Real rhs, SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool removable)
    SCIP_Real SCIPgetRowActivity(SCIP* scip, SCIP_ROW* row)
    SCIP_Real SCIPgetRowLPActivity(SCIP* scip, SCIP_ROW* row)
    SCIP_RETCODE SCIPreleaseRow(SCIP* scip, SCIP_ROW** row)
    SCIP_RETCODE SCIPcacheRowExtensions(SCIP* scip, SCIP_ROW* row)
    SCIP_RETCODE SCIPflushRowExtensions(SCIP* scip, SCIP_ROW* row)
    SCIP_RETCODE SCIPaddVarToRow(SCIP* scip, SCIP_ROW* row, SCIP_VAR* var, SCIP_Real val)
    SCIP_RETCODE SCIPprintRow(SCIP* scip, SCIP_ROW* row, FILE* file)
    int SCIPgetRowNumIntCols(SCIP* scip, SCIP_ROW* row)
    int SCIProwGetNNonz(SCIP_ROW* row)
    SCIP_Real SCIPgetRowObjParallelism(SCIP* scip, SCIP_ROW* row)

    # Column Methods
    SCIP_Real SCIPgetColRedcost(SCIP* scip, SCIP_COL* col)

    # Dual Solution Methods
    SCIP_Real SCIPgetDualbound(SCIP* scip)
    SCIP_Real SCIPgetDualboundRoot(SCIP* scip)
    SCIP_Real SCIPgetVarRedcost(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPgetDualSolVal(SCIP* scip, SCIP_CONS* cons, SCIP_Real* dualsolval, SCIP_Bool* boundconstraint)

    # Reader plugin
    SCIP_RETCODE SCIPincludeReader(SCIP* scip,
                                   const char* name,
                                   const char* desc,
                                   const char* extension,
                                   SCIP_RETCODE (*readercopy) (SCIP* scip, SCIP_READER* reader),
                                   SCIP_RETCODE (*readerfree) (SCIP* scip, SCIP_READER* reader),
                                   SCIP_RETCODE (*readerread) (SCIP* scip, SCIP_READER* reader, const char* filename, SCIP_RESULT* result),
                                   SCIP_RETCODE (*readerwrite) (SCIP* scip, SCIP_READER* reader, FILE* file,
                                                                const char* name, SCIP_PROBDATA* probdata, SCIP_Bool transformed,
                                                                SCIP_OBJSENSE objsense, SCIP_Real objscale, SCIP_Real objoffset,
                                                                SCIP_VAR** vars, int nvars, int nbinvars, int nintvars, int nimplvars, int ncontvars,
                                                                SCIP_VAR** fixedvars, int nfixedvars, int startnvars,
                                                                SCIP_CONS** conss, int nconss, int maxnconss, int startnconss,
                                                                SCIP_Bool genericnames, SCIP_RESULT* result),
                                   SCIP_READERDATA* readerdata)
    SCIP_READER* SCIPfindReader(SCIP* scip, const char* name)
    SCIP_READERDATA* SCIPreaderGetData(SCIP_READER* reader)
    int SCIPgetNReaders(SCIP* scip)

    # Event handler plugin
    SCIP_RETCODE SCIPincludeEventhdlr(SCIP* scip,
                                      const char* name,
                                      const char* desc,
                                      SCIP_RETCODE (*eventcopy) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr),
                                      SCIP_RETCODE (*eventfree) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr),
                                      SCIP_RETCODE (*eventinit) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr),
                                      SCIP_RETCODE (*eventexit) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr),
                                      SCIP_RETCODE (*eventinitsol) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr),
                                      SCIP_RETCODE (*eventexitsol) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr),
                                      SCIP_RETCODE (*eventdelete) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENTDATA** eventdata),
                                      SCIP_RETCODE (*eventexec) (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENT* event, SCIP_EVENTDATA* eventdata),
                                      SCIP_EVENTHDLRDATA* eventhdlrdata)
    SCIP_EVENTHDLRDATA* SCIPeventhdlrGetData(SCIP_EVENTHDLR* eventhdlr)

    # Variable pricer plugin
    SCIP_RETCODE SCIPincludePricer(SCIP* scip,
                                   const char*  name,
                                   const char*  desc,
                                   int priority,
                                   SCIP_Bool delay,
                                   SCIP_RETCODE (*pricercopy) (SCIP* scip, SCIP_PRICER* pricer, SCIP_Bool* valid),
                                   SCIP_RETCODE (*pricerfree) (SCIP* scip, SCIP_PRICER* pricer),
                                   SCIP_RETCODE (*pricerinit) (SCIP* scip, SCIP_PRICER* pricer),
                                   SCIP_RETCODE (*pricerexit) (SCIP* scip, SCIP_PRICER* pricer),
                                   SCIP_RETCODE (*pricerinitsol) (SCIP* scip, SCIP_PRICER* pricer),
                                   SCIP_RETCODE (*pricerexitsol) (SCIP* scip, SCIP_PRICER* pricer),
                                   SCIP_RETCODE (*pricerredcost) (SCIP* scip, SCIP_PRICER* pricer, SCIP_Real* lowerbound, SCIP_Bool* stopearly, SCIP_RESULT* result),
                                   SCIP_RETCODE (*pricerfarkas) (SCIP* scip, SCIP_PRICER* pricer, SCIP_RESULT* result),
                                   SCIP_PRICERDATA* pricerdata)
    SCIP_PRICER* SCIPfindPricer(SCIP* scip, const char* name)
    SCIP_RETCODE SCIPactivatePricer(SCIP* scip, SCIP_PRICER* pricer)
    SCIP_RETCODE SCIPdeactivatePricer(SCIP* scip, SCIP_PRICER* pricer)
    SCIP_PRICERDATA* SCIPpricerGetData(SCIP_PRICER* pricer)

    # Constraint handler plugin
    SCIP_RETCODE SCIPincludeConshdlr(SCIP* scip,
                                     const char* name,
                                     const char* desc,
                                     int sepapriority,
                                     int enfopriority,
                                     int chckpriority,
                                     int sepafreq,
                                     int propfreq,
                                     int eagerfreq,
                                     int maxprerounds,
                                     SCIP_Bool delaysepa,
                                     SCIP_Bool delayprop,
                                     SCIP_Bool needscons,
                                     SCIP_PROPTIMING proptiming,
                                     SCIP_PRESOLTIMING presoltiming,
                                     SCIP_RETCODE (*conshdlrcopy) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_Bool* valid),
                                     SCIP_RETCODE (*consfree) (SCIP* scip, SCIP_CONSHDLR* conshdlr),
                                     SCIP_RETCODE (*consinit) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*consexit) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*consinitpre) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*consexitpre) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*consinitsol) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*consexitsol) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool restart),
                                     SCIP_RETCODE (*consdelete) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_CONSDATA** consdata),
                                     SCIP_RETCODE (*constrans) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* sourcecons, SCIP_CONS** targetcons),
                                     SCIP_RETCODE (*consinitlp) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_Bool* infeasible),
                                     SCIP_RETCODE (*conssepalp) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conssepasol) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_SOL* sol, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consenfolp) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consenforelax) (SCIP* scip, SCIP_SOL* sol, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consenfops) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_Bool objinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conscheck) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_SOL* sol, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool printreason, SCIP_Bool completely, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consprop) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, int nmarkedconss, SCIP_PROPTIMING proptiming, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conspresol) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nrounds, SCIP_PRESOLTIMING presoltiming, int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes, int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides, int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes, int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consresprop) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR* infervar, int inferinfo, SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conslock) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_LOCKTYPE locktype, int nlockspos, int nlocksneg),
                                     SCIP_RETCODE (*consactive) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons),
                                     SCIP_RETCODE (*consdeactive) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons),
                                     SCIP_RETCODE (*consenable) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons),
                                     SCIP_RETCODE (*consdisable) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons),
                                     SCIP_RETCODE (*consdelvars) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*consprint) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, FILE* file),
                                     SCIP_RETCODE (*conscopy) (SCIP* scip, SCIP_CONS** cons, const char* name, SCIP* sourcescip, SCIP_CONSHDLR* sourceconshdlr, SCIP_CONS* sourcecons, SCIP_HASHMAP* varmap, SCIP_HASHMAP* consmap, SCIP_Bool initial, SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate, SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable, SCIP_Bool stickingatnode, SCIP_Bool py_global, SCIP_Bool* valid),
                                     SCIP_RETCODE (*consparse) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** cons, const char* name, const char* str, SCIP_Bool initial, SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate, SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable, SCIP_Bool stickingatnode, SCIP_Bool* success),
                                     SCIP_RETCODE (*consgetvars) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR** vars, int varssize, SCIP_Bool* success),
                                     SCIP_RETCODE (*consgetnvars) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int* nvars, SCIP_Bool* success),
                                     SCIP_RETCODE (*consgetdivebdchgs) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_DIVESET* diveset, SCIP_SOL* sol, SCIP_Bool* success, SCIP_Bool* infeasible),
                                     SCIP_RETCODE (*consgetpermsymgraph)(SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SYM_GRAPH* graph, SCIP_Bool* success),
                                     SCIP_RETCODE (*consgetsignedpermsymgraph)(SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SYM_GRAPH* graph, SCIP_Bool* success),
                                     SCIP_CONSHDLRDATA* conshdlrdata)
    SCIP_CONSHDLRDATA* SCIPconshdlrGetData(SCIP_CONSHDLR* conshdlr)
    SCIP_CONSHDLR* SCIPfindConshdlr(SCIP* scip, const char* name)
    SCIP_RETCODE SCIPcreateCons(SCIP* scip, SCIP_CONS** cons, const char* name, SCIP_CONSHDLR* conshdlr, SCIP_CONSDATA* consdata,
                                SCIP_Bool initial, SCIP_Bool separate, SCIP_Bool enforce, SCIP_Bool check, SCIP_Bool propagate,
                                SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool dynamic, SCIP_Bool removable, SCIP_Bool stickingatnode)

    # Presolve plugin
    SCIP_RETCODE SCIPincludePresol(SCIP* scip,
                                   const char* name,
                                   const char* desc,
                                   int priority,
                                   int maxrounds,
                                   SCIP_PRESOLTIMING timing,
                                   SCIP_RETCODE (*presolcopy) (SCIP* scip, SCIP_PRESOL* presol),
                                   SCIP_RETCODE (*presolfree) (SCIP* scip, SCIP_PRESOL* presol),
                                   SCIP_RETCODE (*presolinit) (SCIP* scip, SCIP_PRESOL* presol),
                                   SCIP_RETCODE (*presolexit) (SCIP* scip, SCIP_PRESOL* presol),
                                   SCIP_RETCODE (*presolinitpre) (SCIP* scip, SCIP_PRESOL* presol),
                                   SCIP_RETCODE (*presolexitpre) (SCIP* scip, SCIP_PRESOL* presol),
                                   SCIP_RETCODE (*presolexec) (SCIP* scip, SCIP_PRESOL* presol, int nrounds, SCIP_PRESOLTIMING presoltiming, int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes, int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides, int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes, int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result),
                                   SCIP_PRESOLDATA* presoldata)
    SCIP_PRESOLDATA* SCIPpresolGetData(SCIP_PRESOL* presol)

    # Separator plugin
    SCIP_RETCODE SCIPincludeSepa(SCIP* scip,
                                 const char* name,
                                 const char* desc,
                                 int priority,
                                 int freq,
                                 SCIP_Real maxbounddist,
                                 SCIP_Bool usessubscip,
                                 SCIP_Bool delay,
                                 SCIP_RETCODE (*sepacopy) (SCIP* scip, SCIP_SEPA* sepa),
                                 SCIP_RETCODE (*sepafree) (SCIP* scip, SCIP_SEPA* sepa),
                                 SCIP_RETCODE (*sepainit) (SCIP* scip, SCIP_SEPA* sepa),
                                 SCIP_RETCODE (*sepaexit) (SCIP* scip, SCIP_SEPA* sepa),
                                 SCIP_RETCODE (*sepainitsol) (SCIP* scip, SCIP_SEPA* sepa),
                                 SCIP_RETCODE (*sepaexitsol) (SCIP* scip, SCIP_SEPA* sepa),
                                 SCIP_RETCODE (*sepaexeclp) (SCIP* scip, SCIP_SEPA* sepa, SCIP_RESULT* result, unsigned int allowlocal, int depth),
                                 SCIP_RETCODE (*sepaexecsol) (SCIP* scip, SCIP_SEPA* sepa, SCIP_SOL* sol, SCIP_RESULT* result, unsigned int allowlocal, int depth),
                                 SCIP_SEPADATA* sepadata)
    SCIP_SEPADATA* SCIPsepaGetData(SCIP_SEPA* sepa)
    SCIP_SEPA* SCIPfindSepa(SCIP* scip, const char* name)

    # Propagator plugin
    SCIP_RETCODE SCIPincludeProp(SCIP* scip,
                                 const char*  name,
                                 const char*  desc,
                                 int priority,
                                 int freq,
                                 SCIP_Bool delay,
                                 SCIP_PROPTIMING timingmask,
                                 int presolpriority,
                                 int presolmaxrounds,
                                 SCIP_PRESOLTIMING presoltiming,
                                 SCIP_RETCODE (*propcopy) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propfree) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propinit) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propexit) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propinitpre) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propexitpre) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propinitsol) (SCIP* scip, SCIP_PROP* prop),
                                 SCIP_RETCODE (*propexitsol) (SCIP* scip, SCIP_PROP* prop, SCIP_Bool restart),
                                 SCIP_RETCODE (*proppresol)  (SCIP* scip, SCIP_PROP* prop, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                                               int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                                               int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                                               int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                                               int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result),
                                 SCIP_RETCODE (*propexec) (SCIP* scip, SCIP_PROP* prop, SCIP_PROPTIMING proptiming, SCIP_RESULT* result),
                                 SCIP_RETCODE (*propresprop) (SCIP* scip, SCIP_PROP* prop, SCIP_VAR* infervar, int inferinfo,
                                                               SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result),
                                 SCIP_PROPDATA*  propdata)

    SCIP_PROPDATA* SCIPpropGetData (SCIP_PROP* prop)

    # Heuristics plugin
    SCIP_RETCODE SCIPincludeHeur(SCIP* scip,
                                 const char* name,
                                 const char* desc,
                                 char dispchar,
                                 int priority,
                                 int freq,
                                 int freqofs,
                                 int maxdepth,
                                 unsigned int timingmask,
                                 SCIP_Bool usessubscip,
                                 SCIP_RETCODE (*heurcopy) (SCIP* scip, SCIP_HEUR* heur),
                                 SCIP_RETCODE (*heurfree) (SCIP* scip, SCIP_HEUR* heur),
                                 SCIP_RETCODE (*heurinit) (SCIP* scip, SCIP_HEUR* heur),
                                 SCIP_RETCODE (*heurexit) (SCIP* scip, SCIP_HEUR* heur),
                                 SCIP_RETCODE (*heurinitsol) (SCIP* scip, SCIP_HEUR* heur),
                                 SCIP_RETCODE (*heurexitsol) (SCIP* scip, SCIP_HEUR* heur),
                                 SCIP_RETCODE (*heurexec) (SCIP* scip, SCIP_HEUR* heur, SCIP_HEURTIMING heurtiming, SCIP_Bool nodeinfeasible, SCIP_RESULT* result),
                                 SCIP_HEURDATA* heurdata)
    SCIP_HEURDATA* SCIPheurGetData(SCIP_HEUR* heur)
    SCIP_HEUR* SCIPfindHeur(SCIP* scip, const char* name)
    SCIP_HEURTIMING SCIPheurGetTimingmask(SCIP_HEUR* heur)
    void SCIPheurSetTimingmask(SCIP_HEUR* heur, SCIP_HEURTIMING timingmask)

    #Relaxation plugin
    SCIP_RETCODE SCIPincludeRelax(SCIP* scip,
		                         const char* name,
                           		 const char* desc,
		                         int priority,
		                         int freq,
		                         SCIP_RETCODE (*relaxcopy) (SCIP* scip, SCIP_RELAX* relax),
                                 SCIP_RETCODE (*relaxfree) (SCIP* scip, SCIP_RELAX* relax),
                                 SCIP_RETCODE (*relaxinit) (SCIP* scip, SCIP_RELAX* relax),
                                 SCIP_RETCODE (*relaxexit) (SCIP* scip, SCIP_RELAX* relax),
                                 SCIP_RETCODE (*relaxinitsol) (SCIP* scip, SCIP_RELAX* relax),
                                 SCIP_RETCODE (*relaxexitsol) (SCIP* scip, SCIP_RELAX* relax),
                                 SCIP_RETCODE (*relaxexec) (SCIP* scip, SCIP_RELAX* relax, SCIP_Real* lowerbound, SCIP_RESULT* result),
                                 SCIP_RELAXDATA* relaxdata)
    SCIP_RELAXDATA* SCIPrelaxGetData(SCIP_RELAX* relax)
    SCIP_RELAX* SCIPfindRelax(SCIP* scip, const char* name)

    # Node selection plugin
    SCIP_RETCODE SCIPincludeNodesel(SCIP* scip,
                                    const char* name,
                                    const char* desc,
                                    int stdpriority,
                                    int memsavepriority,
                                    SCIP_RETCODE (*nodeselcopy) (SCIP* scip, SCIP_NODESEL* nodesel),
                                    SCIP_RETCODE (*nodeselfree) (SCIP* scip, SCIP_NODESEL* nodesel),
                                    SCIP_RETCODE (*nodeselinit) (SCIP* scip, SCIP_NODESEL* nodesel),
                                    SCIP_RETCODE (*nodeselexit) (SCIP* scip, SCIP_NODESEL* nodesel),
                                    SCIP_RETCODE (*nodeselinitsol) (SCIP* scip, SCIP_NODESEL* nodesel),
                                    SCIP_RETCODE (*nodeselexitsol) (SCIP* scip, SCIP_NODESEL* nodesel),
                                    SCIP_RETCODE (*nodeselselect) (SCIP* scip, SCIP_NODESEL* nodesel, SCIP_NODE** selnode),
                                    int (*nodeselcomp) (SCIP* scip, SCIP_NODESEL* nodesel,  SCIP_NODE* node1, SCIP_NODE* node2),
                                    SCIP_NODESELDATA* nodeseldata)
    SCIP_NODESELDATA* SCIPnodeselGetData(SCIP_NODESEL* nodesel)

    # Branching rule plugin
    SCIP_RETCODE SCIPincludeBranchrule(SCIP* scip,
                                       const char* name,
                                       const char* desc,
                                       int priority,
                                       int maxdepth,
                                       SCIP_Real maxbounddist,
                                       SCIP_RETCODE (*branchrulecopy) (SCIP* scip, SCIP_BRANCHRULE* branchrule),
                                       SCIP_RETCODE (*branchrulefree) (SCIP* scip, SCIP_BRANCHRULE* branchrule),
                                       SCIP_RETCODE (*branchruleinit) (SCIP* scip, SCIP_BRANCHRULE* branchrule),
                                       SCIP_RETCODE (*branchruleexit) (SCIP* scip, SCIP_BRANCHRULE* branchrule),
                                       SCIP_RETCODE (*branchruleinitsol) (SCIP* scip, SCIP_BRANCHRULE* branchrule),
                                       SCIP_RETCODE (*branchruleexitsol) (SCIP* scip, SCIP_BRANCHRULE* branchrule),
                                       SCIP_RETCODE (*branchruleexeclp) (SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result),
                                       SCIP_RETCODE (*branchruleexecext) (SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result),
                                       SCIP_RETCODE (*branchruleexecps) (SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_Bool allowaddcons, SCIP_RESULT* result),
                                       SCIP_BRANCHRULEDATA* branchruledata)
    SCIP_BRANCHRULEDATA* SCIPbranchruleGetData(SCIP_BRANCHRULE* branchrule)
    const char* SCIPbranchruleGetName(SCIP_BRANCHRULE* branchrule)
    SCIP_BRANCHRULE* SCIPfindBranchrule(SCIP* scip, const char*  name)

    # cut selector plugin
    SCIP_RETCODE SCIPincludeCutsel(SCIP* scip,
                                   const char* name,
                                   const char* desc,
                                   int priority,
                                   SCIP_RETCODE (*cutselcopy) (SCIP* scip, SCIP_CUTSEL* cutsel),
                                   SCIP_RETCODE (*cutselfree) (SCIP* scip, SCIP_CUTSEL* cutsel),
                                   SCIP_RETCODE (*cutselinit) (SCIP* scip, SCIP_CUTSEL* cutsel),
                                   SCIP_RETCODE (*cutselexit) (SCIP* scip, SCIP_CUTSEL* cutsel),
                                   SCIP_RETCODE (*cutselinitsol) (SCIP* scip, SCIP_CUTSEL* cutsel),
                                   SCIP_RETCODE (*cutselexitsol) (SCIP* scip, SCIP_CUTSEL* cutsel),
                                   SCIP_RETCODE (*cutselselect) (SCIP* scip, SCIP_CUTSEL* cutsel, SCIP_ROW** cuts,
                                                                 int ncuts, SCIP_ROW** forcedcuts, int nforcedcuts,
                                                                 SCIP_Bool root, int maxnselectedcuts,
                                                                 int* nselectedcuts, SCIP_RESULT* result),
                                   SCIP_CUTSELDATA* cutseldata)
    SCIP_CUTSELDATA* SCIPcutselGetData(SCIP_CUTSEL* cutsel)

    # Benders' decomposition plugin
    SCIP_RETCODE SCIPincludeBenders(SCIP* scip,
                                   const char*  name,
                                   const char*  desc,
                                   int priority,
                                   SCIP_Bool cutlp,
                                   SCIP_Bool cutpseudo,
                                   SCIP_Bool cutrelax,
                                   SCIP_Bool shareaux,
                                   SCIP_RETCODE (*benderscopy) (SCIP* scip, SCIP_BENDERS* benders, SCIP_Bool threadsafe),
                                   SCIP_RETCODE (*bendersfree) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersinit) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersexit) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersinitpre) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersexitpre) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersinitsol) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersexitsol) (SCIP* scip, SCIP_BENDERS* benders),
                                   SCIP_RETCODE (*bendersgetvar) (SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR* var, SCIP_VAR** mappedvar, int probnumber),
                                   SCIP_RETCODE (*benderscreatesub) (SCIP* scip, SCIP_BENDERS* benders, int probnumber),
                                   SCIP_RETCODE (*benderspresubsolve) (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, SCIP_BENDERSENFOTYPE type, SCIP_Bool checkint, SCIP_Bool* infeasible, SCIP_Bool* auxviol, SCIP_Bool* skipsolve,  SCIP_RESULT* result),
                                   SCIP_RETCODE (*benderssolvesubconvex) (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber, SCIP_Bool onlyconvex, SCIP_Real* objective, SCIP_RESULT* result),
                                   SCIP_RETCODE (*benderssolvesub) (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber, SCIP_Real* objective, SCIP_RESULT* result),
                                   SCIP_RETCODE (*benderspostsolve) (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, SCIP_BENDERSENFOTYPE type, int* mergecands, int npriomergecands, int nmergecands, SCIP_Bool checkint, SCIP_Bool infeasible, SCIP_Bool* merged),
                                   SCIP_RETCODE (*bendersfreesub) (SCIP* scip, SCIP_BENDERS* benders, int probnumber),
                                   SCIP_BENDERSDATA* bendersdata)
    SCIP_BENDERS* SCIPfindBenders(SCIP* scip, const char* name)
    SCIP_RETCODE SCIPactivateBenders(SCIP* scip, SCIP_BENDERS* benders, int nsubproblems)
    SCIP_BENDERSDATA* SCIPbendersGetData(SCIP_BENDERS* benders)
    SCIP_RETCODE SCIPcreateBendersDefault(SCIP* scip, SCIP** subproblems, int nsubproblems)
    int SCIPbendersGetNSubproblems(SCIP_BENDERS* benders)
    SCIP_RETCODE SCIPsolveBendersSubproblems(SCIP* scip, SCIP_BENDERS* benders,
            SCIP_SOL* sol, SCIP_RESULT* result, SCIP_Bool* infeasible,
            SCIP_Bool* auxviol, SCIP_Bool checkint)
    SCIP_RETCODE SCIPsetupBendersSubproblem(SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber,
            SCIP_BENDERSENFOTYPE type)
    SCIP_RETCODE SCIPsolveBendersSubproblem(SCIP* scip, SCIP_BENDERS* benders,
            SCIP_SOL* sol, int probnumber, SCIP_Bool* infeasible,
            SCIP_Bool solvecip, SCIP_Real* objective)
    SCIP_RETCODE SCIPfreeBendersSubproblem(SCIP* scip, SCIP_BENDERS* benders, int probnumber)
    int SCIPgetNActiveBenders(SCIP* scip)
    SCIP_BENDERS** SCIPgetBenders(SCIP* scip)
    void SCIPbendersUpdateSubproblemLowerbound(SCIP_BENDERS* benders, int probnumber, SCIP_Real lowerbound)
    SCIP_RETCODE SCIPaddBendersSubproblem(SCIP* scip, SCIP_BENDERS* benders, SCIP* subproblem)
    SCIP* SCIPbendersSubproblem(SCIP_BENDERS* benders, int probnumber)
    SCIP_RETCODE SCIPgetBendersMasterVar(SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR* var, SCIP_VAR** mappedvar)
    SCIP_RETCODE SCIPgetBendersSubproblemVar(SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR* var, SCIP_VAR** mappedvar, int probnumber)
    SCIP_VAR* SCIPbendersGetAuxiliaryVar(SCIP_BENDERS* benders, int probnumber)
    SCIP_RETCODE SCIPcheckBendersSubproblemOptimality(SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber, SCIP_Bool* optimal)
    SCIP_RETCODE SCIPincludeBendersDefaultCuts(SCIP* scip, SCIP_BENDERS* benders)
    void SCIPbendersSetSubproblemIsConvex(SCIP_BENDERS* benders, int probnumber, SCIP_Bool isconvex)

    # Benders' decomposition cuts plugin
    SCIP_RETCODE SCIPincludeBenderscut(SCIP* scip,
                                      SCIP_BENDERS* benders,
                                      const char*  name,
                                      const char*  desc,
                                      int priority,
                                      SCIP_Bool islpcut,
                                      SCIP_RETCODE (*benderscutcopy) (SCIP* scip, SCIP_BENDERS* benders, SCIP_BENDERSCUT* benderscut),
                                      SCIP_RETCODE (*benderscutfree) (SCIP* scip, SCIP_BENDERSCUT* benderscut),
                                      SCIP_RETCODE (*benderscutinit) (SCIP* scip, SCIP_BENDERSCUT* benderscut),
                                      SCIP_RETCODE (*benderscutexit) (SCIP* scip, SCIP_BENDERSCUT* benderscut),
                                      SCIP_RETCODE (*benderscutinitsol) (SCIP* scip, SCIP_BENDERSCUT* benderscut),
                                      SCIP_RETCODE (*benderscutexitsol) (SCIP* scip, SCIP_BENDERSCUT* benderscut),
                                      SCIP_RETCODE (*benderscutexec) (SCIP* scip, SCIP_BENDERS* benders, SCIP_BENDERSCUT* benderscut, SCIP_SOL* sol, int probnumber, SCIP_BENDERSENFOTYPE type, SCIP_RESULT* result),
                                      SCIP_BENDERSCUTDATA* benderscutdata)
    SCIP_BENDERSCUT* SCIPfindBenderscut(SCIP_BENDERS* benders, const char* name)
    SCIP_BENDERSCUTDATA* SCIPbenderscutGetData(SCIP_BENDERSCUT* benderscut)
    SCIP_RETCODE SCIPstoreBendersCut(SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR** vars, SCIP_Real* vals, SCIP_Real lhs, SCIP_Real rhs, int nvars)
    SCIP_RETCODE SCIPapplyBendersStoredCuts(SCIP* scip, SCIP_BENDERS* benders)

    SCIP_RETCODE SCIPbranchVar(SCIP* scip,
                                SCIP_VAR* var,
                                SCIP_NODE**  downchild,
                                SCIP_NODE**  eqchild,
                                SCIP_NODE**  upchild)

    SCIP_RETCODE SCIPbranchVarVal(SCIP* scip,
                                SCIP_VAR* var,
                                SCIP_Real val,
                                SCIP_NODE** downchild,
                                SCIP_NODE**  eqchild,
                                SCIP_NODE** upchild)
    int SCIPgetNLPBranchCands(SCIP* scip)
    SCIP_RETCODE SCIPgetLPBranchCands(SCIP* scip, SCIP_VAR*** lpcands, SCIP_Real** lpcandssol,
                                      SCIP_Real** lpcandsfrac, int* nlpcands, int* npriolpcands, int* nfracimplvars)
    SCIP_RETCODE SCIPgetPseudoBranchCands(SCIP* scip, SCIP_VAR*** pseudocands, int* npseudocands, int* npriopseudocands)
    SCIP_RETCODE SCIPstartStrongbranch(SCIP* scip, SCIP_Bool enablepropogation)
    SCIP_RETCODE SCIPendStrongbranch(SCIP* scip)
    SCIP_RETCODE SCIPgetVarStrongbranchLast(SCIP* scip, SCIP_VAR* var, SCIP_Real* down, SCIP_Real* up, SCIP_Bool* downvalid, SCIP_Bool* upvalid, SCIP_Real* solval, SCIP_Real* lpobjval)
    SCIP_Longint SCIPgetVarStrongbranchNode(SCIP* scip, SCIP_VAR* var)
    SCIP_Real SCIPgetBranchScoreMultiple(SCIP* scip, SCIP_VAR* var, int nchildren, SCIP_Real* gains)
    SCIP_RETCODE SCIPgetVarStrongbranchWithPropagation(SCIP* scip, SCIP_VAR* var, SCIP_Real solval, SCIP_Real lpobjval, int itlim, int maxproprounds, SCIP_Real* down, SCIP_Real* up, SCIP_Bool* downvalid, SCIP_Bool* upvalid, SCIP_Longint* ndomredsdown, SCIP_Longint* ndomredsup, SCIP_Bool* downinf, SCIP_Bool* upinf, SCIP_Bool* downconflict, SCIP_Bool* upconflict, SCIP_Bool* lperror, SCIP_Real* newlbs, SCIP_Real* newubs)
    SCIP_RETCODE SCIPgetVarStrongbranchInt(SCIP* scip, SCIP_VAR* var, int itlim, SCIP_Bool idempotent, SCIP_Real* down, SCIP_Real* up, SCIP_Bool* downvalid, SCIP_Bool* upvalid, SCIP_Bool* downinf, SCIP_Bool* upinf, SCIP_Bool* downconflict, SCIP_Bool* upconflict, SCIP_Bool* lperror)
    SCIP_RETCODE SCIPupdateVarPseudocost(SCIP* scip, SCIP_VAR* var, SCIP_Real solvaldelta, SCIP_Real objdelta, SCIP_Real weight)
    SCIP_RETCODE SCIPgetVarStrongbranchFrac(SCIP* scip, SCIP_VAR* var, int itlim, SCIP_Bool idempotent, SCIP_Real* down, SCIP_Real* up, SCIP_Bool* downvalid, SCIP_Bool* upvalid, SCIP_Bool* downinf, SCIP_Bool* upinf, SCIP_Bool* downconflict, SCIP_Bool* upconflict, SCIP_Bool* lperror)

    # Numerical Methods
    SCIP_Real SCIPinfinity(SCIP* scip)
    SCIP_Real SCIPfrac(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPfeasFrac(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPfeasFloor(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPfeasCeil(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPfeasRound(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisZero(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasIntegral(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasZero(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasNegative(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasPositive(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisInfinity(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisLE(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisLT(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisGE(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisGT(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisEQ(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisFeasEQ(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisFeasLT(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisFeasGT(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisFeasLE(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisFeasGE(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisHugeValue(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisPositive(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisNegative(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisIntegral(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPgetTreesizeEstimation(SCIP* scip)

    # Statistic Methods
    SCIP_RETCODE SCIPprintStatistics(SCIP* scip, FILE* outfile)
    SCIP_Longint SCIPgetNNodes(SCIP* scip)
    SCIP_Longint SCIPgetNTotalNodes(SCIP* scip)
    SCIP_Longint SCIPgetNFeasibleLeaves(SCIP* scip)
    SCIP_Longint SCIPgetNInfeasibleLeaves(SCIP* scip)
    SCIP_Longint SCIPgetNLPs(SCIP* scip)
    SCIP_Longint SCIPgetNLPIterations(SCIP* scip)
    int SCIPgetNSepaRounds(SCIP* scip)

    # Parameter Functions
    SCIP_RETCODE SCIPsetBoolParam(SCIP* scip, char* name, SCIP_Bool value)
    SCIP_RETCODE SCIPsetIntParam(SCIP* scip, char* name, int value)
    SCIP_RETCODE SCIPsetLongintParam(SCIP* scip, char* name, SCIP_Longint value)
    SCIP_RETCODE SCIPsetRealParam(SCIP* scip, char* name, SCIP_Real value)
    SCIP_RETCODE SCIPsetCharParam(SCIP* scip, char* name, char value)
    SCIP_RETCODE SCIPsetStringParam(SCIP* scip, char* name, char* value)
    SCIP_RETCODE SCIPreadParams(SCIP* scip, char* file)
    SCIP_RETCODE SCIPwriteParams(SCIP* scip, char* file, SCIP_Bool comments, SCIP_Bool onlychanged)
    SCIP_RETCODE SCIPreadProb(SCIP* scip, char* file, char* extension)
    SCIP_RETCODE SCIPsetEmphasis(SCIP* scip, SCIP_PARAMEMPHASIS paramemphasis, SCIP_Bool quiet)
    SCIP_RETCODE SCIPresetParam(SCIP* scip, const char* name)
    SCIP_RETCODE SCIPresetParams(SCIP* scip)
    SCIP_PARAM* SCIPgetParam(SCIP* scip,  const char*  name)
    SCIP_PARAM** SCIPgetParams(SCIP* scip)
    int SCIPgetNParams(SCIP* scip)

    # LPI Functions
    SCIP_RETCODE SCIPgetLPI(SCIP* scip, SCIP_LPI** lpi)
    SCIP_RETCODE SCIPlpiCreate(SCIP_LPI** lpi, SCIP_MESSAGEHDLR* messagehdlr, const char* name, SCIP_OBJSEN objsen)
    SCIP_RETCODE SCIPlpiFree(SCIP_LPI** lpi)
    SCIP_RETCODE SCIPlpiWriteLP(SCIP_LPI* lpi, const char* fname)
    SCIP_RETCODE SCIPlpiReadLP(SCIP_LPI* lpi, const char* fname)
    SCIP_RETCODE SCIPlpiAddCols(SCIP_LPI* lpi, int ncols, const SCIP_Real* obj, const SCIP_Real* lb, const SCIP_Real* ub, char** colnames, int nnonz, const int* beg, const int* ind, const SCIP_Real* val)
    SCIP_RETCODE SCIPlpiDelCols(SCIP_LPI* lpi, int firstcol, int lastcol)
    SCIP_RETCODE SCIPlpiAddRows(SCIP_LPI* lpi, int nrows, const SCIP_Real* lhs, const SCIP_Real* rhs, char** rownames, int nnonz, const int* beg, const int* ind, const SCIP_Real* val)
    SCIP_RETCODE SCIPlpiDelRows(SCIP_LPI* lpi, int firstrow, int lastrow)
    SCIP_RETCODE SCIPlpiGetBounds(SCIP_LPI* lpi, int firstrow, int lastrow, SCIP_Real* lhss, SCIP_Real* rhss)
    SCIP_RETCODE SCIPlpiGetSides(SCIP_LPI* lpi, int firstcol, int lastcol, SCIP_Real* lbs, SCIP_Real* ubs)
    SCIP_RETCODE SCIPlpiChgObj(SCIP_LPI* lpi, int ncols, int* ind, SCIP_Real* obj)
    SCIP_RETCODE SCIPlpiChgCoef(SCIP_LPI* lpi, int row, int col, SCIP_Real newval)
    SCIP_RETCODE SCIPlpiChgBounds(SCIP_LPI* lpi, int nrows, const int* ind, const SCIP_Real* lhs, const SCIP_Real* rhs)
    SCIP_RETCODE SCIPlpiChgSides(SCIP_LPI* lpi, int ncols, const int* ind, const SCIP_Real* lbs, const SCIP_Real* ubs)
    SCIP_RETCODE SCIPlpiClear(SCIP_LPI* lpi)
    SCIP_RETCODE SCIPlpiGetNRows(SCIP_LPI* lpi, int* nrows)
    SCIP_RETCODE SCIPlpiGetNCols(SCIP_LPI* lpi, int* ncols)
    SCIP_RETCODE SCIPlpiSolveDual(SCIP_LPI* lpi)
    SCIP_RETCODE SCIPlpiSolvePrimal(SCIP_LPI* lpi)
    SCIP_RETCODE SCIPlpiIsOptimal(SCIP_LPI* lpi)
    SCIP_RETCODE SCIPlpiGetObjval(SCIP_LPI* lpi, SCIP_Real* objval)
    SCIP_RETCODE SCIPlpiGetSol(SCIP_LPI* lpi, SCIP_Real* objval, SCIP_Real* primsol, SCIP_Real* dualsol, SCIP_Real* activity, SCIP_Real* redcost)
    SCIP_RETCODE SCIPlpiGetIterations(SCIP_LPI* lpi, int* iterations)
    SCIP_RETCODE SCIPlpiGetPrimalRay(SCIP_LPI* lpi, SCIP_Real* ray)
    SCIP_RETCODE SCIPlpiGetDualfarkas(SCIP_LPI* lpi, SCIP_Real* dualfarkas)
    SCIP_RETCODE SCIPlpiGetBasisInd(SCIP_LPI* lpi, int* bind)
    SCIP_RETCODE SCIPlpiGetRealSolQuality(SCIP_LPI* lpi, SCIP_LPSOLQUALITY qualityindicator, SCIP_Real* quality)
    SCIP_RETCODE SCIPlpiGetIntpar(SCIP_LPI* lpi, SCIP_LPPARAM type, int* ival)
    SCIP_RETCODE SCIPlpiGetRealpar(SCIP_LPI* lpi, SCIP_LPPARAM type, SCIP_Real* dval)
    SCIP_RETCODE SCIPlpiSetIntpar(SCIP_LPI* lpi, SCIP_LPPARAM type, int ival)
    SCIP_RETCODE SCIPlpiSetRealpar(SCIP_LPI* lpi, SCIP_LPPARAM type, SCIP_Real dval)
    SCIP_Bool    SCIPlpiHasPrimalRay(SCIP_LPI* lpi)
    SCIP_Bool    SCIPlpiHasDualRay(SCIP_LPI* lpi)
    SCIP_Real    SCIPlpiInfinity(SCIP_LPI* lpi)
    SCIP_Bool    SCIPlpiIsInfinity(SCIP_LPI* lpi, SCIP_Real val)
    SCIP_Bool    SCIPlpiIsPrimalFeasible(SCIP_LPI* lpi)
    SCIP_Bool    SCIPlpiIsDualFeasible(SCIP_LPI* lpi)

    #re-optimization routines
    SCIP_RETCODE SCIPfreeReoptSolve(SCIP* scip)
    SCIP_RETCODE SCIPchgReoptObjective(SCIP* scip, SCIP_OBJSENSE objsense, SCIP_VAR** vars, SCIP_Real* coefs, int nvars)
    SCIP_RETCODE SCIPenableReoptimization(SCIP* scip, SCIP_Bool enable)

    BMS_BLKMEM* SCIPblkmem(SCIP* scip)

    # pub_misc.h
    SCIP_RETCODE SCIPhashmapCreate(SCIP_HASHMAP** hashmap, BMS_BLKMEM* blkmem, int mapsize)
    void SCIPhashmapFree(SCIP_HASHMAP** hashmap)


cdef extern from "scip/tree.h":
    int SCIPnodeGetNAddedConss(SCIP_NODE* node)

cdef extern from "scip/scipdefplugins.h":
    SCIP_RETCODE SCIPincludeDefaultPlugins(SCIP* scip)

cdef extern from "scip/bendersdefcuts.h":
    SCIP_RETCODE SCIPincludeBendersDefaultCuts(SCIP* scip, SCIP_BENDERS* benders)

cdef extern from "scip/cons_linear.h":
    SCIP_RETCODE SCIPcreateConsLinear(SCIP* scip,
                                      SCIP_CONS** cons,
                                      char* name,
                                      int nvars,
                                      SCIP_VAR** vars,
                                      SCIP_Real* vals,
                                      SCIP_Real lhs,
                                      SCIP_Real rhs,
                                      SCIP_Bool initial,
                                      SCIP_Bool separate,
                                      SCIP_Bool enforce,
                                      SCIP_Bool check,
                                      SCIP_Bool propagate,
                                      SCIP_Bool local,
                                      SCIP_Bool modifiable,
                                      SCIP_Bool dynamic,
                                      SCIP_Bool removable,
                                      SCIP_Bool stickingatnode)
    SCIP_RETCODE SCIPaddCoefLinear(SCIP* scip,
                                   SCIP_CONS* cons,
                                   SCIP_VAR* var,
                                   SCIP_Real val)

    SCIP_Real SCIPgetDualsolLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real SCIPgetDualfarkasLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPchgLhsLinear(SCIP* scip, SCIP_CONS* cons, SCIP_Real lhs)
    SCIP_RETCODE SCIPchgRhsLinear(SCIP* scip, SCIP_CONS* cons, SCIP_Real rhs)
    SCIP_Real SCIPgetLhsLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real SCIPgetRhsLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPchgCoefLinear(SCIP* scip, SCIP_CONS* cons, SCIP_VAR* var, SCIP_Real newval)
    SCIP_RETCODE SCIPdelCoefLinear(SCIP* scip, SCIP_CONS* cons, SCIP_VAR*)
    SCIP_RETCODE SCIPaddCoefLinear(SCIP* scip, SCIP_CONS* cons, SCIP_VAR*, SCIP_Real val)
    SCIP_Real SCIPgetActivityLinear(SCIP* scip, SCIP_CONS* cons, SCIP_SOL* sol)
    SCIP_VAR** SCIPgetVarsLinear(SCIP* scip, SCIP_CONS* cons)
    int SCIPgetNVarsLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real* SCIPgetValsLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_ROW* SCIPgetRowLinear(SCIP* scip, SCIP_CONS* cons)

cdef extern from "scip/cons_knapsack.h":
    SCIP_RETCODE SCIPcreateConsKnapsack(SCIP* scip,
                                      SCIP_CONS** cons,
                                      char* name,
                                      int nvars,
                                      SCIP_VAR** vars,
                                      SCIP_Longint* weights,
                                      SCIP_Longint capacity,
                                      SCIP_Bool initial,
                                      SCIP_Bool separate,
                                      SCIP_Bool enforce,
                                      SCIP_Bool check,
                                      SCIP_Bool propagate,
                                      SCIP_Bool local,
                                      SCIP_Bool modifiable,
                                      SCIP_Bool dynamic,
                                      SCIP_Bool removable,
                                      SCIP_Bool stickingatnode)
    SCIP_RETCODE SCIPaddCoefKnapsack(SCIP* scip,
                                   SCIP_CONS* cons,
                                   SCIP_VAR* var,
                                   SCIP_Longint val)

    SCIP_Real SCIPgetDualsolKnapsack(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real SCIPgetDualfarkasKnapsack(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPchgCapacityKnapsack(SCIP* scip, SCIP_CONS* cons, SCIP_Real rhs)
    SCIP_Longint SCIPgetCapacityKnapsack(SCIP* scip, SCIP_CONS* cons)
    SCIP_VAR** SCIPgetVarsKnapsack(SCIP* scip, SCIP_CONS* cons)
    int SCIPgetNVarsKnapsack(SCIP* scip, SCIP_CONS* cons)
    SCIP_Longint* SCIPgetWeightsKnapsack(SCIP* scip, SCIP_CONS* cons)

cdef extern from "scip/cons_cumulative.h":
    SCIP_RETCODE SCIPcreateConsCumulative(SCIP* scip,
                                          SCIP_CONS** cons,
                                          char* name,
                                          int nvars,
                                          SCIP_VAR** vars,
                                          int* durations,
                                          int* demands,
                                          int capacity,
                                          SCIP_Bool initial,
                                          SCIP_Bool separate,
                                          SCIP_Bool enforce,
                                          SCIP_Bool check,
                                          SCIP_Bool propagate,
                                          SCIP_Bool local,
                                          SCIP_Bool modifiable,
                                          SCIP_Bool dynamic,
                                          SCIP_Bool removable,
                                          SCIP_Bool stickingatnode)

    SCIP_RETCODE SCIPcreateConsBasicCumulative(SCIP* scip, SCIP_CONS** cons,
                                              char* name,
                                              int nvars,
                                              SCIP_VAR** vars,
                                              int* durations,
                                              int* demands,
                                              int capacity)

    SCIP_RETCODE SCIPsetHminCumulative(SCIP* scip, SCIP_CONS* cons, int hmin)
    SCIP_RETCODE SCIPsetHmaxCumulative(SCIP* scip, SCIP_CONS* cons, int hmax)

    int          SCIPgetHminCumulative(SCIP* scip, SCIP_CONS* cons)
    int          SCIPgetHmaxCumulative(SCIP* scip, SCIP_CONS* cons)

    SCIP_VAR**   SCIPgetVarsCumulative(SCIP* scip, SCIP_CONS* cons)
    int          SCIPgetNVarsCumulative(SCIP* scip, SCIP_CONS* cons)
    int*         SCIPgetDurationsCumulative(SCIP* scip, SCIP_CONS* cons)
    int*         SCIPgetDemandsCumulative(SCIP* scip, SCIP_CONS* cons)
    int          SCIPgetCapacityCumulative(SCIP* scip, SCIP_CONS* cons)


cdef extern from "scip/cons_nonlinear.h":
    SCIP_EXPR* SCIPgetExprNonlinear(SCIP_CONS* cons)
    SCIP_RETCODE SCIPcreateConsNonlinear(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         SCIP_EXPR* expr,
                                         SCIP_Real lhs,
                                         SCIP_Real rhs,
                                         SCIP_Bool initial,
                                         SCIP_Bool separate,
                                         SCIP_Bool enforce,
                                         SCIP_Bool check,
                                         SCIP_Bool propagate,
                                         SCIP_Bool local,
                                         SCIP_Bool modifiable,
                                         SCIP_Bool dynamic,
                                         SCIP_Bool removable)
    SCIP_RETCODE SCIPcreateConsQuadraticNonlinear(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         int nlinvars,
                                         SCIP_VAR** linvars,
                                         SCIP_Real* lincoefs,
                                         int nquadterms,
                                         SCIP_VAR** quadvars1,
                                         SCIP_VAR** quadvars2,
                                         SCIP_Real* quadcoeffs,
                                         SCIP_Real lhs,
                                         SCIP_Real rhs,
                                         SCIP_Bool initial,
                                         SCIP_Bool separate,
                                         SCIP_Bool enforce,
                                         SCIP_Bool check,
                                         SCIP_Bool propagate,
                                         SCIP_Bool local,
                                         SCIP_Bool modifiable,
                                         SCIP_Bool dynamic,
                                         SCIP_Bool removable)
    SCIP_RETCODE SCIPaddLinearVarNonlinear(SCIP* scip,
                                           SCIP_CONS* cons,
                                           SCIP_VAR* var,
                                           SCIP_Real coef)
    SCIP_RETCODE SCIPaddExprNonlinear(SCIP* scip,
                                      SCIP_CONS* cons,
                                      SCIP_EXPR* expr,
                                      SCIP_Real coef)
    SCIP_RETCODE SCIPchgLhsNonlinear(SCIP* scip, SCIP_CONS* cons, SCIP_Real lhs)
    SCIP_RETCODE SCIPchgRhsNonlinear(SCIP* scip, SCIP_CONS* cons, SCIP_Real rhs)
    SCIP_Real SCIPgetLhsNonlinear(SCIP_CONS* cons)
    SCIP_Real SCIPgetRhsNonlinear(SCIP_CONS* cons)
    SCIP_RETCODE SCIPcheckQuadraticNonlinear(SCIP* scip, SCIP_CONS* cons, SCIP_Bool* isquadratic)

cdef extern from "scip/cons_sos1.h":
    SCIP_RETCODE SCIPcreateConsSOS1(SCIP* scip,
                                    SCIP_CONS** cons,
                                    const char* name,
                                    int nvars,
                                    SCIP_VAR** vars,
                                    SCIP_Real* weights,
                                    SCIP_Bool initial,
                                    SCIP_Bool separate,
                                    SCIP_Bool enforce,
                                    SCIP_Bool check,
                                    SCIP_Bool propagate,
                                    SCIP_Bool local,
                                    SCIP_Bool dynamic,
                                    SCIP_Bool removable,
                                    SCIP_Bool stickingatnode)

    SCIP_RETCODE SCIPaddVarSOS1(SCIP* scip,
                                SCIP_CONS* cons,
                                SCIP_VAR* var,
                                SCIP_Real weight)

    SCIP_RETCODE SCIPappendVarSOS1(SCIP* scip,
                                   SCIP_CONS* cons,
                                   SCIP_VAR* var)


cdef extern from "scip/cons_sos2.h":
    SCIP_RETCODE SCIPcreateConsSOS2(SCIP* scip,
                                    SCIP_CONS** cons,
                                    const char* name,
                                    int nvars,
                                    SCIP_VAR** vars,
                                    SCIP_Real* weights,
                                    SCIP_Bool initial,
                                    SCIP_Bool separate,
                                    SCIP_Bool enforce,
                                    SCIP_Bool check,
                                    SCIP_Bool propagate,
                                    SCIP_Bool local,
                                    SCIP_Bool dynamic,
                                    SCIP_Bool removable,
                                    SCIP_Bool stickingatnode)

    SCIP_RETCODE SCIPaddVarSOS2(SCIP* scip,
                                SCIP_CONS* cons,
                                SCIP_VAR* var,
                                SCIP_Real weight)

    SCIP_RETCODE SCIPappendVarSOS2(SCIP* scip,
                                   SCIP_CONS* cons,
                                   SCIP_VAR* var)

cdef extern from "scip/cons_disjunction.h":
    SCIP_RETCODE SCIPcreateConsDisjunction(SCIP* scip,
                                            SCIP_CONS** cons,
                                            const char* name,
                                            int nconss,
                                            SCIP_CONS** conss,
                                            SCIP_CONS* relaxcons,
                                            SCIP_Bool initial,
                                            SCIP_Bool enforce,
                                            SCIP_Bool check,
                                            SCIP_Bool local,
                                            SCIP_Bool modifiable,
                                            SCIP_Bool dynamic)

    SCIP_RETCODE SCIPaddConsElemDisjunction(SCIP* scip,
                                            SCIP_CONS* cons,
                                            SCIP_CONS* addcons)

cdef extern from "scip/cons_and.h":
    SCIP_RETCODE SCIPcreateConsAnd(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         SCIP_VAR* resvar,
                                         int nvars,
                                         SCIP_VAR** vars,
                                         SCIP_Bool initial,
                                         SCIP_Bool separate,
                                         SCIP_Bool enforce,
                                         SCIP_Bool check,
                                         SCIP_Bool propagate,
                                         SCIP_Bool local,
                                         SCIP_Bool modifiable,
                                         SCIP_Bool dynamic,
                                         SCIP_Bool removable,
                                         SCIP_Bool stickingatnode)

cdef extern from "scip/cons_or.h":
    SCIP_RETCODE SCIPcreateConsOr(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         SCIP_VAR* resvar,
                                         int nvars,
                                         SCIP_VAR** vars,
                                         SCIP_Bool initial,
                                         SCIP_Bool separate,
                                         SCIP_Bool enforce,
                                         SCIP_Bool check,
                                         SCIP_Bool propagate,
                                         SCIP_Bool local,
                                         SCIP_Bool modifiable,
                                         SCIP_Bool dynamic,
                                         SCIP_Bool removable,
                                         SCIP_Bool stickingatnode)

cdef extern from "scip/cons_xor.h":
    SCIP_RETCODE SCIPcreateConsXor(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         SCIP_Bool rhs,
                                         int nvars,
                                         SCIP_VAR** vars,
                                         SCIP_Bool initial,
                                         SCIP_Bool separate,
                                         SCIP_Bool enforce,
                                         SCIP_Bool check,
                                         SCIP_Bool propagate,
                                         SCIP_Bool local,
                                         SCIP_Bool modifiable,
                                         SCIP_Bool dynamic,
                                         SCIP_Bool removable,
                                         SCIP_Bool stickingatnode)
cdef extern from "scip/scip_cons.h":
    SCIP_RETCODE SCIPprintCons(SCIP* scip,
                               SCIP_CONS* cons,
                               FILE* file)

cdef extern from "blockmemshell/memory.h":
    void BMScheckEmptyMemory()
    long long BMSgetMemoryUsed()

cdef extern from "scip/scip_expr.h":
    SCIP_RETCODE SCIPcreateExpr(SCIP* scip,
                                SCIP_EXPR** expr,
                                SCIP_EXPRHDLR* exprhdlr,
                                SCIP_EXPRDATA* exprdata,
                                int nchildren,
                                SCIP_EXPR** children,
                                SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                void* ownercreatedata)
    SCIP_RETCODE SCIPcreateExprMonomial(SCIP* scip,
                                        SCIP_EXPR** expr,
                                        int nfactors,
                                        SCIP_VAR** vars,
                                        SCIP_Real* exponents,
                                        SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                        void* ownercreatedata)
    SCIP_RETCODE SCIPreleaseExpr(SCIP* scip, SCIP_EXPR** expr)

cdef extern from "scip/pub_expr.h":
    void SCIPexprGetQuadraticData(SCIP_EXPR* expr,
                                  SCIP_Real* constant,
                                  int* nlinexprs,
                                  SCIP_EXPR*** linexprs,
                                  SCIP_Real** lincoefs,
                                  int* nquadexprs,
                                  int* nbilinexprs,
                                  SCIP_Real** eigenvalues,
                                  SCIP_Real** eigenvectors)
    void SCIPexprGetQuadraticQuadTerm(SCIP_EXPR* quadexpr,
                                      int termidx,
                                      SCIP_EXPR** expr,
                                      SCIP_Real* lincoef,
                                      SCIP_Real* sqrcoef,
                                      int* nadjbilin,
                                      int** adjbilin,
                                      SCIP_EXPR** sqrexpr)
    void SCIPexprGetQuadraticBilinTerm(SCIP_EXPR* expr,
                                       int termidx,
                                       SCIP_EXPR** expr1,
                                       SCIP_EXPR** expr2,
                                       SCIP_Real* coef,
                                       int* pos2,
                                       SCIP_EXPR** prodexpr)

cdef extern from "scip/expr_var.h":
   SCIP_VAR* SCIPgetVarExprVar(SCIP_EXPR* expr)
   SCIP_RETCODE SCIPcreateExprVar(SCIP* scip,
                                  SCIP_EXPR** expr,
                                  SCIP_VAR* var,
                                  SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                  void* ownercreatedata)

cdef extern from "scip/expr_varidx.h":
   SCIP_RETCODE SCIPcreateExprVaridx(SCIP* scip,
                                     SCIP_EXPR** expr,
                                     int varidx,
                                     SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                     void* ownercreatedata)

cdef extern from "scip/expr_value.h":
   SCIP_RETCODE SCIPcreateExprValue(SCIP* scip,
                                    SCIP_EXPR** expr,
                                    SCIP_Real value,
                                    SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                    void* ownercreatedata)

cdef extern from "scip/expr_sum.h":
    SCIP_RETCODE SCIPcreateExprSum(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   int nchildren,
                                   SCIP_EXPR** children,
                                   SCIP_Real* coefficients,
                                   SCIP_Real constant,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)

cdef extern from "scip/expr_abs.h":
    SCIP_RETCODE SCIPcreateExprAbs(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   SCIP_EXPR* child,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)

cdef extern from "scip/expr_exp.h":
    SCIP_RETCODE SCIPcreateExprExp(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   SCIP_EXPR* child,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)

cdef extern from "scip/expr_log.h":
    SCIP_RETCODE SCIPcreateExprLog(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   SCIP_EXPR* child,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)

cdef extern from "scip/expr_trig.h":
    SCIP_RETCODE SCIPcreateExprSin(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   SCIP_EXPR* child,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)
    SCIP_RETCODE SCIPcreateExprCos(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   SCIP_EXPR* child,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)

cdef extern from "scip/expr_product.h":
    SCIP_RETCODE SCIPcreateExprProduct(SCIP* scip,
                                       SCIP_EXPR** expr,
                                       int nchildren,
                                       SCIP_EXPR** children,
                                       SCIP_Real coefficient,
                                       SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                       void* ownercreatedata)

cdef extern from "scip/expr_pow.h":
    SCIP_RETCODE SCIPcreateExprPow(SCIP* scip,
                                   SCIP_EXPR** expr,
                                   SCIP_EXPR* child,
                                   SCIP_Real exponent,
                                   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)),
                                   void* ownercreatedata)

cdef extern from "scip/pub_nlp.h":
    SCIP_Real SCIPnlrowGetConstant(SCIP_NLROW* nlrow)
    int SCIPnlrowGetNLinearVars(SCIP_NLROW* nlrow)
    SCIP_VAR** SCIPnlrowGetLinearVars(SCIP_NLROW* nlrow)
    SCIP_Real* SCIPnlrowGetLinearCoefs(SCIP_NLROW* nlrow)
    SCIP_Real SCIPnlrowGetLhs(SCIP_NLROW* nlrow)
    SCIP_Real SCIPnlrowGetRhs(SCIP_NLROW* nlrow)
    const char* SCIPnlrowGetName(SCIP_NLROW* nlrow)
    SCIP_Real SCIPnlrowGetDualsol(SCIP_NLROW* nlrow)

cdef extern from "scip/scip_nlp.h":
    SCIP_Bool SCIPisNLPConstructed(SCIP* scip)
    SCIP_NLROW** SCIPgetNLPNlRows(SCIP* scip)
    int SCIPgetNNLPNlRows(SCIP* scip)
    SCIP_RETCODE SCIPgetNlRowSolActivity(SCIP* scip, SCIP_NLROW* nlrow, SCIP_SOL* sol, SCIP_Real* activity)
    SCIP_RETCODE SCIPgetNlRowSolFeasibility(SCIP* scip, SCIP_NLROW* nlrow, SCIP_SOL* sol, SCIP_Real* feasibility)
    SCIP_RETCODE SCIPgetNlRowActivityBounds(SCIP* scip, SCIP_NLROW* nlrow, SCIP_Real* minactivity, SCIP_Real* maxactivity)
    SCIP_RETCODE SCIPprintNlRow(SCIP* scip, SCIP_NLROW* nlrow, FILE* file)


cdef extern from "scip/cons_cardinality.h":
    SCIP_RETCODE SCIPcreateConsCardinality(SCIP* scip,
                                            SCIP_CONS** cons,
                                            const char* name,
                                            int nvars,
                                            SCIP_VAR** vars,
                                            int cardval,
                                            SCIP_VAR** indvars,
                                            SCIP_Real* weights,
                                            SCIP_Bool initial,
                                            SCIP_Bool separate,
                                            SCIP_Bool enforce,
                                            SCIP_Bool check,
                                            SCIP_Bool propagate,
                                            SCIP_Bool local,
                                            SCIP_Bool dynamic,
                                            SCIP_Bool removable,
                                            SCIP_Bool stickingatnode)

    SCIP_RETCODE SCIPaddVarCardinality(SCIP* scip,
                                       SCIP_CONS* cons,
                                       SCIP_VAR* var,
                                       SCIP_VAR* indvar,
                                       SCIP_Real weight)

    SCIP_RETCODE SCIPappendVarCardinality(SCIP* scip,
                                          SCIP_CONS* cons,
                                          SCIP_VAR* var,
                                          SCIP_VAR* indvar)

cdef extern from "scip/cons_indicator.h":
    SCIP_RETCODE SCIPcreateConsIndicator(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         SCIP_VAR* binvar,
                                         int nvars,
                                         SCIP_VAR** vars,
                                         SCIP_Real* vals,
                                         SCIP_Real rhs,
                                         SCIP_Bool initial,
                                         SCIP_Bool separate,
                                         SCIP_Bool enforce,
                                         SCIP_Bool check,
                                         SCIP_Bool propagate,
                                         SCIP_Bool local,
                                         SCIP_Bool dynamic,
                                         SCIP_Bool removable,
                                         SCIP_Bool stickingatnode)

    SCIP_RETCODE SCIPaddVarIndicator(SCIP* scip,
                                     SCIP_CONS* cons,
                                     SCIP_VAR* var,
                                     SCIP_Real val)

    SCIP_VAR* SCIPgetSlackVarIndicator(SCIP_CONS* cons)
    SCIP_CONS* SCIPgetLinearConsIndicator(SCIP_CONS* cons)

cdef extern from "scip/misc.h":
    SCIP_RETCODE SCIPhashmapCreate(SCIP_HASHMAP** hashmap, BMS_BLKMEM* blkmem, int mapsize)
    void SCIPhashmapFree(SCIP_HASHMAP** hashmap)

cdef extern from "scip/scip_copy.h":
    SCIP_RETCODE SCIPtranslateSubSol(SCIP* scip, SCIP* subscip, SCIP_SOL* subsol, SCIP_HEUR* heur, SCIP_VAR** subvars, SCIP_SOL** newsol)

cdef extern from "scip/heuristics.h":
    SCIP_RETCODE SCIPcopyLargeNeighborhoodSearch(SCIP* sourcescip, SCIP* subscip, SCIP_HASHMAP*	varmap,	const char* suffix, SCIP_VAR** fixedvars, SCIP_Real* fixedvals, int nfixedvars,	SCIP_Bool uselprows, SCIP_Bool copycuts, SCIP_Bool* success, SCIP_Bool* valid)

cdef extern from "scip/cons_countsols.h":
    SCIP_RETCODE SCIPcount(SCIP* scip)
    SCIP_RETCODE SCIPsetParamsCountsols(SCIP* scip)
    SCIP_Longint SCIPgetNCountedSols(SCIP* scip, SCIP_Bool* valid)

cdef extern from "scip/paramset.h":

    ctypedef struct SCIP_PARAM:
        pass

    const char* SCIPparamGetName(SCIP_PARAM* param)
    SCIP_PARAMTYPE SCIPparamGetType(SCIP_PARAM* param)
    SCIP_Bool SCIPparamGetBool(SCIP_PARAM* param)
    int SCIPparamGetInt(SCIP_PARAM* param)
    SCIP_Longint SCIPparamGetLongint(SCIP_PARAM* param)
    SCIP_Real SCIPparamGetReal(SCIP_PARAM* param)
    char SCIPparamGetChar(SCIP_PARAM* param)
    char* SCIPparamGetString(SCIP_PARAM* param)

cdef extern from "scip/pub_lp.h":
    # Row Methods
    const char* SCIProwGetName(SCIP_ROW* row)
    SCIP_Real SCIProwGetLhs(SCIP_ROW* row)
    SCIP_Real SCIProwGetRhs(SCIP_ROW* row)
    SCIP_Real SCIProwGetConstant(SCIP_ROW* row)
    SCIP_Real SCIProwGetDualsol(SCIP_ROW* row)
    SCIP_Real SCIProwGetDualfarkas(SCIP_ROW* row)
    int SCIProwGetLPPos(SCIP_ROW* row)
    SCIP_BASESTAT SCIProwGetBasisStatus(SCIP_ROW* row)
    SCIP_Bool SCIProwIsIntegral(SCIP_ROW* row)
    SCIP_Bool SCIProwIsLocal(SCIP_ROW* row)
    SCIP_Bool SCIProwIsModifiable(SCIP_ROW* row)
    SCIP_Bool SCIProwIsRemovable(SCIP_ROW* row)
    SCIP_Bool SCIProwIsInGlobalCutpool(SCIP_ROW* row)
    int SCIProwGetNNonz(SCIP_ROW* row)
    int SCIProwGetNLPNonz(SCIP_ROW* row)
    SCIP_COL** SCIProwGetCols(SCIP_ROW* row)
    SCIP_Real* SCIProwGetVals(SCIP_ROW* row)
    SCIP_Real SCIProwGetNorm(SCIP_ROW* row)
    SCIP_Real SCIProwGetDualsol(SCIP_ROW* row)
    SCIP_Real SCIProwGetParallelism(SCIP_ROW* row1, SCIP_ROW* row2, const char orthofunc)
    int SCIProwGetAge(SCIP_ROW* row)
    SCIP_Bool SCIProwIsRemovable(SCIP_ROW* row)
    SCIP_ROWORIGINTYPE SCIProwGetOrigintype(SCIP_ROW* row)
    SCIP_CONS* SCIProwGetOriginCons(SCIP_ROW* row)

    # Column Methods
    int SCIPcolGetLPPos(SCIP_COL* col)
    SCIP_BASESTAT SCIPcolGetBasisStatus(SCIP_COL* col)
    SCIP_Bool SCIPcolIsIntegral(SCIP_COL* col)
    SCIP_VAR* SCIPcolGetVar(SCIP_COL* col)
    SCIP_Real SCIPcolGetPrimsol(SCIP_COL* col)
    SCIP_Real SCIPcolGetLb(SCIP_COL* col)
    SCIP_Real SCIPcolGetUb(SCIP_COL* col)
    int SCIPcolGetNLPNonz(SCIP_COL* col)
    int SCIPcolGetNNonz(SCIP_COL* col)
    SCIP_ROW** SCIPcolGetRows(SCIP_COL* col)
    SCIP_Real* SCIPcolGetVals(SCIP_COL* col)
    int SCIPcolGetAge(SCIP_COL* col)
    int SCIPcolGetIndex(SCIP_COL* col)
    SCIP_Real SCIPcolGetObj(SCIP_COL* col)

cdef extern from "scip/scip_tree.h":
    SCIP_RETCODE SCIPgetOpenNodesData(SCIP* scip, SCIP_NODE*** leaves, SCIP_NODE*** children, SCIP_NODE*** siblings, int* nleaves, int* nchildren, int* nsiblings)
    SCIP_RETCODE SCIPgetChildren(SCIP* scip, SCIP_NODE*** children, int* nchildren)
    SCIP_Longint SCIPgetNChildren(SCIP* scip)
    SCIP_NODE* SCIPgetBestChild(SCIP* scip)
    SCIP_RETCODE SCIPgetSiblings(SCIP* scip, SCIP_NODE*** siblings, int* nsiblings)
    SCIP_RETCODE SCIPgetNSiblings(SCIP* scip)
    SCIP_RETCODE SCIPgetLeaves(SCIP* scip, SCIP_NODE*** leaves, int* nleaves)
    SCIP_Longint SCIPgetNLeaves(SCIP* scip)
    SCIP_NODE* SCIPgetBestSibling(SCIP* scip)
    SCIP_NODE* SCIPgetBestLeaf(SCIP* scip)
    SCIP_NODE* SCIPgetPrioChild(SCIP* scip)
    SCIP_NODE* SCIPgetPrioSibling(SCIP* scip)
    SCIP_NODE* SCIPgetBestNode(SCIP* scip)
    SCIP_NODE* SCIPgetBestboundNode(SCIP* scip)
    SCIP_RETCODE SCIPrepropagateNode(SCIP* scip, SCIP_NODE* node)

cdef extern from "scip/scip_var.h":
    SCIP_RETCODE SCIPchgVarBranchPriority(SCIP* scip, SCIP_VAR* var, int branchpriority)

    SCIP_RETCODE SCIPgetNegatedVar(SCIP* scip, SCIP_VAR* var, SCIP_VAR** negvar)

cdef extern from "tpi/tpi.h":
    int SCIPtpiGetNumThreads()

cdef class Expr:
    cdef public terms

cdef class Event:
    cdef SCIP_EVENT* event
    # can be used to store problem data
    cdef public object data
    @staticmethod
    cdef create(SCIP_EVENT* scip_event)

cdef class Column:
    cdef SCIP_COL* scip_col
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_COL* scipcol)

cdef class Row:
    cdef SCIP_ROW* scip_row
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_ROW* sciprow)

cdef class NLRow:
    cdef SCIP_NLROW* scip_nlrow
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_NLROW* scipnlrow)

cdef class Solution:
    cdef SCIP_SOL* sol
    cdef SCIP* scip
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP* scip, SCIP_SOL* scip_sol)

cdef class DomainChanges:
    cdef SCIP_DOMCHG* scip_domchg

    @staticmethod
    cdef create(SCIP_DOMCHG* scip_domchg)

cdef class BoundChange:
    cdef SCIP_BOUNDCHG* scip_boundchg

    @staticmethod
    cdef create(SCIP_BOUNDCHG* scip_boundchg)

cdef class Node:
    cdef SCIP_NODE* scip_node
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_NODE* scipnode)

cdef class Variable(Expr):
    cdef SCIP_VAR* scip_var
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_VAR* scipvar)

cdef class Constraint:
    cdef SCIP_CONS* scip_cons
    # can be used to store problem data
    cdef public object data

    @staticmethod
    cdef create(SCIP_CONS* scipcons)

cdef class Model:
    cdef SCIP* _scip
    cdef SCIP_Bool* _valid
    # store best solution to get the solution values easier
    cdef Solution _bestSol
    # can be used to store problem data
    cdef public object data
    # make Model weak referentiable
    cdef object __weakref__
    # flag to indicate whether the SCIP should be freed. It will not be freed if an empty Model was created to wrap a
    # C-API SCIP instance.
    cdef SCIP_Bool _freescip
    # map to store python variables
    cdef _modelvars
    # used to keep track of the number of event handlers generated
    cdef int _generated_event_handlers_count

    @staticmethod
    cdef create(SCIP* scip)
