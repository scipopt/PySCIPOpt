##@file scip.pxd
#@brief holding prototype of the SCIP public functions to use them in PySCIPOpt
cdef extern from "scip/scip.h":
    # SCIP internal types
    ctypedef enum SCIP_RETCODE:
        SCIP_OKAY               =   1
        SCIP_ERROR              =   0
        SCIP_NOMEMORY           =  -1
        SCIP_READERROR          =  -2
        SCIP_WRITEERROR         =  -3
        SCIP_NOFILE             =  -4
        SCIP_FILECREATEERROR    =  -5
        SCIP_LPERROR            =  -6
        SCIP_NOPROBLEM          =  -7
        SCIP_INVALIDCALL        =  -8
        SCIP_INVALIDDATA        =  -9
        SCIP_INVALIDRESULT      = -10
        SCIP_PLUGINNOTFOUND     = -11
        SCIP_PARAMETERUNKNOWN   = -12
        SCIP_PARAMETERWRONGTYPE = -13
        SCIP_PARAMETERWRONGVAL  = -14
        SCIP_KEYALREADYEXISTING = -15
        SCIP_MAXDEPTHLEVEL      = -16

    ctypedef enum SCIP_VARTYPE:
        SCIP_VARTYPE_BINARY     = 0
        SCIP_VARTYPE_INTEGER    = 1
        SCIP_VARTYPE_IMPLINT    = 2
        SCIP_VARTYPE_CONTINUOUS = 3

    ctypedef enum SCIP_OBJSENSE:
        SCIP_OBJSENSE_MAXIMIZE = -1
        SCIP_OBJSENSE_MINIMIZE =  1

    # This version is used in LPI.
    ctypedef enum SCIP_OBJSEN:
        SCIP_OBJSEN_MAXIMIZE = -1
        SCIP_OBJSEN_MINIMIZE =  1

    ctypedef enum SCIP_BOUNDTYPE:
        SCIP_BOUNDTYPE_LOWER = 0
        SCIP_BOUNDTYPE_UPPER = 1

    ctypedef enum SCIP_RESULT:
        SCIP_DIDNOTRUN   =   1
        SCIP_DELAYED     =   2
        SCIP_DIDNOTFIND  =   3
        SCIP_FEASIBLE    =   4
        SCIP_INFEASIBLE  =   5
        SCIP_UNBOUNDED   =   6
        SCIP_CUTOFF      =   7
        SCIP_SEPARATED   =   8
        SCIP_NEWROUND    =   9
        SCIP_REDUCEDDOM  =  10
        SCIP_CONSADDED   =  11
        SCIP_CONSCHANGED =  12
        SCIP_BRANCHED    =  13
        SCIP_SOLVELP     =  14
        SCIP_FOUNDSOL    =  15
        SCIP_SUSPENDED   =  16
        SCIP_SUCCESS     =  17

    ctypedef enum SCIP_STATUS:
        SCIP_STATUS_UNKNOWN        =  0
        SCIP_STATUS_USERINTERRUPT  =  1
        SCIP_STATUS_NODELIMIT      =  2
        SCIP_STATUS_TOTALNODELIMIT =  3
        SCIP_STATUS_STALLNODELIMIT =  4
        SCIP_STATUS_TIMELIMIT      =  5
        SCIP_STATUS_MEMLIMIT       =  6
        SCIP_STATUS_GAPLIMIT       =  7
        SCIP_STATUS_SOLLIMIT       =  8
        SCIP_STATUS_BESTSOLLIMIT   =  9
        SCIP_STATUS_RESTARTLIMIT   = 10
        SCIP_STATUS_OPTIMAL        = 11
        SCIP_STATUS_INFEASIBLE     = 12
        SCIP_STATUS_UNBOUNDED      = 13
        SCIP_STATUS_INFORUNBD      = 14

    ctypedef enum SCIP_STAGE:
        SCIP_STAGE_INIT         =  0
        SCIP_STAGE_PROBLEM      =  1
        SCIP_STAGE_TRANSFORMING =  2
        SCIP_STAGE_TRANSFORMED  =  3
        SCIP_STAGE_INITPRESOLVE =  4
        SCIP_STAGE_PRESOLVING   =  5
        SCIP_STAGE_EXITPRESOLVE =  6
        SCIP_STAGE_PRESOLVED    =  7
        SCIP_STAGE_INITSOLVE    =  8
        SCIP_STAGE_SOLVING      =  9
        SCIP_STAGE_SOLVED       = 10
        SCIP_STAGE_EXITSOLVE    = 11
        SCIP_STAGE_FREETRANS    = 12
        SCIP_STAGE_FREE         = 13

    ctypedef enum SCIP_NODETYPE:
        SCIP_NODETYPE_FOCUSNODE   =  0
        SCIP_NODETYPE_PROBINGNODE =  1
        SCIP_NODETYPE_SIBLING     =  2
        SCIP_NODETYPE_CHILD       =  3
        SCIP_NODETYPE_LEAF        =  4
        SCIP_NODETYPE_DEADEND     =  5
        SCIP_NODETYPE_JUNCTION    =  6
        SCIP_NODETYPE_PSEUDOFORK  =  7
        SCIP_NODETYPE_FORK        =  8
        SCIP_NODETYPE_SUBROOT     =  9
        SCIP_NODETYPE_REFOCUSNODE = 10

    ctypedef enum SCIP_PARAMSETTING:
        SCIP_PARAMSETTING_DEFAULT    = 0
        SCIP_PARAMSETTING_AGGRESSIVE = 1
        SCIP_PARAMSETTING_FAST       = 2
        SCIP_PARAMSETTING_OFF        = 3

    ctypedef enum SCIP_PARAMTYPE:
        SCIP_PARAMTYPE_BOOL    = 0
        SCIP_PARAMTYPE_INT     = 1
        SCIP_PARAMTYPE_LONGINT = 2
        SCIP_PARAMTYPE_REAL    = 3
        SCIP_PARAMTYPE_CHAR    = 4
        SCIP_PARAMTYPE_STRING  = 5

    ctypedef enum SCIP_PARAMEMPHASIS:
        SCIP_PARAMEMPHASIS_DEFAULT      = 0
        SCIP_PARAMEMPHASIS_CPSOLVER     = 1
        SCIP_PARAMEMPHASIS_EASYCIP      = 2
        SCIP_PARAMEMPHASIS_FEASIBILITY  = 3
        SCIP_PARAMEMPHASIS_HARDLP       = 4
        SCIP_PARAMEMPHASIS_OPTIMALITY   = 5
        SCIP_PARAMEMPHASIS_COUNTER      = 6
        SCIP_PARAMEMPHASIS_PHASEFEAS    = 7
        SCIP_PARAMEMPHASIS_PHASEIMPROVE = 8
        SCIP_PARAMEMPHASIS_PHASEPROOF   = 9

    ctypedef enum SCIP_PROPTIMING:
        SCIP_PROPTIMING_BEFORELP     = 0x001u
        SCIP_PROPTIMING_DURINGLPLOOP = 0x002u
        SCIP_PROPTIMING_AFTERLPLOOP  = 0x004u
        SCIP_PROPTIMING_AFTERLPNODE  = 0x008u

    ctypedef enum SCIP_PRESOLTIMING:
        SCIP_PRESOLTIMING_NONE       = 0x002u
        SCIP_PRESOLTIMING_FAST       = 0x004u
        SCIP_PRESOLTIMING_MEDIUM     = 0x008u
        SCIP_PRESOLTIMING_EXHAUSTIVE = 0x010u

    ctypedef enum SCIP_HEURTIMING:
        SCIP_HEURTIMING_BEFORENODE        = 0x001u
        SCIP_HEURTIMING_DURINGLPLOOP      = 0x002u
        SCIP_HEURTIMING_AFTERLPLOOP       = 0x004u
        SCIP_HEURTIMING_AFTERLPNODE       = 0x008u
        SCIP_HEURTIMING_AFTERPSEUDONODE   = 0x010u
        SCIP_HEURTIMING_AFTERLPPLUNGE     = 0x020u
        SCIP_HEURTIMING_AFTERPSEUDOPLUNGE = 0x040u
        SCIP_HEURTIMING_DURINGPRICINGLOOP = 0x080u
        SCIP_HEURTIMING_BEFOREPRESOL      = 0x100u
        SCIP_HEURTIMING_DURINGPRESOLLOOP  = 0x200u
        SCIP_HEURTIMING_AFTERPROPLOOP     = 0x400u

    ctypedef enum SCIP_EXPROP:
        SCIP_EXPR_VARIDX    =  1
        SCIP_EXPR_CONST     =  2
        SCIP_EXPR_PARAM     =  3
        SCIP_EXPR_PLUS      =  8
        SCIP_EXPR_MINUS     =  9
        SCIP_EXPR_MUL       = 10
        SCIP_EXPR_DIV       = 11
        SCIP_EXPR_SQUARE    = 12
        SCIP_EXPR_SQRT      = 13
        SCIP_EXPR_REALPOWER = 14
        SCIP_EXPR_INTPOWER  = 15
        SCIP_EXPR_SIGNPOWER = 16
        SCIP_EXPR_EXP       = 17
        SCIP_EXPR_LOG       = 18
        SCIP_EXPR_SIN       = 19
        SCIP_EXPR_COS       = 20
        SCIP_EXPR_TAN       = 21
        SCIP_EXPR_MIN       = 24
        SCIP_EXPR_MAX       = 25
        SCIP_EXPR_ABS       = 26
        SCIP_EXPR_SIGN      = 27
        SCIP_EXPR_SUM       = 64
        SCIP_EXPR_PRODUCT   = 65
        SCIP_EXPR_LINEAR    = 66
        SCIP_EXPR_QUADRATIC = 67
        SCIP_EXPR_POLYNOMIAL= 68
        SCIP_EXPR_USER      = 69
        SCIP_EXPR_LAST      = 70


    ctypedef enum SCIP_BASESTAT:
        SCIP_BASESTAT_LOWER = 0
        SCIP_BASESTAT_BASIC = 1
        SCIP_BASESTAT_UPPER = 2
        SCIP_BASESTAT_ZERO  = 3


    ctypedef enum SCIP_EVENTTYPE:
        SCIP_EVENTTYPE_DISABLED         = 0x00000000u
        SCIP_EVENTTYPE_VARADDED         = 0x00000001u
        SCIP_EVENTTYPE_VARDELETED       = 0x00000002u
        SCIP_EVENTTYPE_VARFIXED         = 0x00000004u
        SCIP_EVENTTYPE_VARUNLOCKED      = 0x00000008u
        SCIP_EVENTTYPE_OBJCHANGED       = 0x00000010u
        SCIP_EVENTTYPE_GLBCHANGED       = 0x00000020u
        SCIP_EVENTTYPE_GUBCHANGED       = 0x00000040u
        SCIP_EVENTTYPE_LBTIGHTENED      = 0x00000080u
        SCIP_EVENTTYPE_LBRELAXED        = 0x00000100u
        SCIP_EVENTTYPE_UBTIGHTENED      = 0x00000200u
        SCIP_EVENTTYPE_UBRELAXED        = 0x00000400u
        SCIP_EVENTTYPE_GHOLEADDED       = 0x00000800u
        SCIP_EVENTTYPE_GHOLEREMOVED     = 0x00001000u
        SCIP_EVENTTYPE_LHOLEADDED       = 0x00002000u
        SCIP_EVENTTYPE_LHOLEREMOVED     = 0x00004000u
        SCIP_EVENTTYPE_IMPLADDED        = 0x00008000u
        SCIP_EVENTTYPE_PRESOLVEROUND    = 0x00010000u
        SCIP_EVENTTYPE_NODEFOCUSED      = 0x00020000u
        SCIP_EVENTTYPE_NODEFEASIBLE     = 0x00040000u
        SCIP_EVENTTYPE_NODEINFEASIBLE   = 0x00080000u
        SCIP_EVENTTYPE_NODEBRANCHED     = 0x00100000u
        SCIP_EVENTTYPE_FIRSTLPSOLVED    = 0x00200000u
        SCIP_EVENTTYPE_LPSOLVED         = 0x00400000u
        SCIP_EVENTTYPE_POORSOLFOUND     = 0x00800000u
        SCIP_EVENTTYPE_BESTSOLFOUND     = 0x01000000u
        SCIP_EVENTTYPE_ROWADDEDSEPA     = 0x02000000u
        SCIP_EVENTTYPE_ROWDELETEDSEPA   = 0x04000000u
        SCIP_EVENTTYPE_ROWADDEDLP       = 0x08000000u
        SCIP_EVENTTYPE_ROWDELETEDLP     = 0x10000000u
        SCIP_EVENTTYPE_ROWCOEFCHANGED   = 0x20000000u
        SCIP_EVENTTYPE_ROWCONSTCHANGED  = 0x40000000u
        SCIP_EVENTTYPE_ROWSIDECHANGED   = 0x80000000u
        SCIP_EVENTTYPE_SYNC             = 0x100000000u

        SCIP_EVENTTYPE_LPEVENT          = SCIP_EVENTTYPE_FIRSTLPSOLVED | SCIP_EVENTTYPE_LPSOLVED

    ctypedef enum SCIP_LPSOLQUALITY:
        SCIP_LPSOLQUALITY_ESTIMCONDITION = 0
        SCIP_LPSOLQUALITY_EXACTCONDITION = 1

    ctypedef enum SCIP_LOCKTYPE:
        SCIP_LOCKTYPE_MODEL    = 0
        SCIP_LOCKTYPE_CONFLICT = 1

    ctypedef enum SCIP_BENDERSENFOTYPE:
        SCIP_BENDERSENFOTYPE_LP      = 1
        SCIP_BENDERSENFOTYPE_RELAX   = 2
        SCIP_BENDERSENFOTYPE_PSEUDO  = 3
        SCIP_BENDERSENFOTYPE_CHECK   = 4

    ctypedef enum SCIP_LPSOLSTAT:
        SCIP_LPSOLSTAT_NOTSOLVED    = 0
        SCIP_LPSOLSTAT_OPTIMAL      = 1
        SCIP_LPSOLSTAT_INFEASIBLE   = 2
        SCIP_LPSOLSTAT_UNBOUNDEDRAY = 3
        SCIP_LPSOLSTAT_OBJLIMIT     = 4
        SCIP_LPSOLSTAT_ITERLIMIT    = 5
        SCIP_LPSOLSTAT_TIMELIMIT    = 6
        SCIP_LPSOLSTAT_ERROR        = 7

    ctypedef enum SCIP_BRANCHDIR:
        SCIP_BRANCHDIR_DOWNWARDS = 0
        SCIP_BRANCHDIR_UPWARDS   = 1
        SCIP_BRANCHDIR_FIXED     = 2
        SCIP_BRANCHDIR_AUTO      = 3

    ctypedef enum SCIP_BOUNDCHGTYPE:
        SCIP_BOUNDCHGTYPE_BRANCHING = 0
        SCIP_BOUNDCHGTYPE_CONSINFER = 1
        SCIP_BOUNDCHGTYPE_PROPINFER = 2

    ctypedef enum SCIP_ROWORIGINTYPE:
        SCIP_ROWORIGINTYPE_UNSPEC = 0
        SCIP_ROWORIGINTYPE_CONS   = 1
        SCIP_ROWORIGINTYPE_SEPA   = 2
        SCIP_ROWORIGINTYPE_REOPT  = 3

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

    ctypedef struct SCIP_BOUNDTYPE:
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

    ctypedef struct SCIP_EXPRTREE:
        pass

    ctypedef struct SCIP_EXPRDATA_MONOMIAL:
        pass

    ctypedef struct SCIP_BENDERS:
        pass

    ctypedef struct SCIP_BENDERSDATA:
        pass

    ctypedef struct SCIP_BENDERSCUT:
        pass

    ctypedef struct SCIP_BENDERSCUTDATA:
        pass

    ctypedef struct SCIP_QUADVAREVENTDATA:
        pass

    ctypedef struct SCIP_QUADVARTERM:
        SCIP_VAR* var
        SCIP_Real lincoef
        SCIP_Real sqrcoef
        int nadjbilin
        int adjbilinsize
        int* adjbilin
        SCIP_QUADVAREVENTDATA* eventdata

    ctypedef struct SCIP_BILINTERM:
        SCIP_VAR* var1
        SCIP_VAR* var2
        SCIP_Real coef

    ctypedef struct SCIP_QUADELEM:
        int idx1
        int idx2
        SCIP_Real coef

    ctypedef struct SCIP_BOUNDCHG:
        pass

    ctypedef union SCIP_DOMCHG:
        pass

    ctypedef void (*messagecallback) (SCIP_MESSAGEHDLR *messagehdlr, FILE *file, const char *msg)
    ctypedef void (*errormessagecallback) (void *data, FILE *file, const char *msg)
    ctypedef SCIP_RETCODE (*messagehdlrfree) (SCIP_MESSAGEHDLR *messagehdlr)

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
    SCIP_RETCODE SCIPmessagehdlrCreate(SCIP_MESSAGEHDLR **messagehdlr,
                                       SCIP_Bool bufferedoutput,
                                       const char *filename,
                                       SCIP_Bool quiet,
                                       messagecallback,
                                       messagecallback,
                                       messagecallback,
                                       messagehdlrfree,
                                       SCIP_MESSAGEHDLRDATA *messagehdlrdata)

    SCIP_RETCODE SCIPsetMessagehdlr(SCIP* scip, SCIP_MESSAGEHDLR* messagehdlr)
    void SCIPsetMessagehdlrQuiet(SCIP* scip, SCIP_Bool quiet)
    void SCIPmessageSetErrorPrinting(errormessagecallback, void* data)
    void SCIPsetMessagehdlrLogfile(SCIP* scip, const char* filename)
    SCIP_Real SCIPversion()
    void SCIPprintVersion(SCIP* scip, FILE* outfile)
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
    SCIP_VAR** SCIPgetVars(SCIP* scip)
    SCIP_VAR** SCIPgetOrigVars(SCIP* scip)
    const char* SCIPvarGetName(SCIP_VAR* var)
    int SCIPvarGetIndex(SCIP_VAR* var)
    int SCIPgetNVars(SCIP* scip)
    int SCIPgetNOrigVars(SCIP* scip)
    int SCIPgetNIntVars(SCIP* scip)
    int SCIPgetNBinVars(SCIP* scip)
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
    SCIP_Real SCIPgetVarPseudocost(SCIP* scip, SCIP_VAR *var, SCIP_BRANCHDIR dir)
    SCIP_Real SCIPvarGetCutoffSum(SCIP_VAR* var, SCIP_BRANCHDIR dir)
    SCIP_Longint SCIPvarGetNBranchings(SCIP_VAR* var, SCIP_BRANCHDIR dir)

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
    SCIP_COL** SCIPgetLPCols(SCIP *scip)
    SCIP_ROW** SCIPgetLPRows(SCIP *scip)

    # Cutting Plane Methods
    SCIP_RETCODE SCIPaddPoolCut(SCIP* scip, SCIP_ROW* row)
    SCIP_Real SCIPgetCutEfficacy(SCIP* scip, SCIP_SOL* sol, SCIP_ROW* cut)
    SCIP_Bool SCIPisCutEfficacious(SCIP* scip, SCIP_SOL* sol, SCIP_ROW* cut)
    int SCIPgetNCuts(SCIP* scip)
    int SCIPgetNCutsApplied(SCIP* scip)
    SCIP_RETCODE SCIPseparateSol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool pretendroot, SCIP_Bool allowlocal, SCIP_Bool onlydelayed, SCIP_Bool* delayed, SCIP_Bool* cutoff)

    # Constraint Methods
    SCIP_RETCODE SCIPcaptureCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPreleaseCons(SCIP* scip, SCIP_CONS** cons)
    SCIP_RETCODE SCIPtransformCons(SCIP* scip, SCIP_CONS* cons, SCIP_CONS** transcons)
    SCIP_RETCODE SCIPgetTransformedCons(SCIP* scip, SCIP_CONS* cons, SCIP_CONS** transcons)
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
    SCIP_RETCODE SCIPsetConsChecked(SCIP *scip, SCIP_CONS *cons, SCIP_Bool check)
    SCIP_RETCODE SCIPsetConsRemovable(SCIP *scip, SCIP_CONS *cons, SCIP_Bool removable)
    SCIP_RETCODE SCIPsetConsInitial(SCIP *scip, SCIP_CONS *cons, SCIP_Bool initial)
    SCIP_RETCODE SCIPsetConsEnforced(SCIP *scip, SCIP_CONS *cons, SCIP_Bool enforce)

    # Primal Solution Methods
    SCIP_SOL** SCIPgetSols(SCIP* scip)
    int SCIPgetNSols(SCIP* scip)
    int SCIPgetNSolsFound(SCIP* scip)
    int SCIPgetNLimSolsFound(SCIP* scip)
    int SCIPgetNBestSolsFound(SCIP* scip)
    SCIP_SOL* SCIPgetBestSol(SCIP* scip)
    SCIP_Real SCIPgetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var)
    SCIP_RETCODE SCIPwriteVarName(SCIP* scip, FILE* outfile, SCIP_VAR* var, SCIP_Bool vartype)
    SCIP_Real SCIPgetSolOrigObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_Real SCIPgetSolTransObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_RETCODE SCIPcreateSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPcreatePartialSol(SCIP* scip, SCIP_SOL** sol,SCIP_HEUR* heur)
    SCIP_RETCODE SCIPsetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var, SCIP_Real val)
    SCIP_RETCODE SCIPtrySolFree(SCIP* scip, SCIP_SOL** sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* stored)
    SCIP_RETCODE SCIPtrySol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* stored)
    SCIP_RETCODE SCIPfreeSol(SCIP* scip, SCIP_SOL** sol)
    SCIP_RETCODE SCIPprintBestSol(SCIP* scip, FILE* outfile, SCIP_Bool printzeros)
    SCIP_RETCODE SCIPprintSol(SCIP* scip, SCIP_SOL* sol, FILE* outfile, SCIP_Bool printzeros)
    SCIP_Real SCIPgetPrimalbound(SCIP* scip)
    SCIP_Real SCIPgetGap(SCIP* scip)
    int SCIPgetDepth(SCIP* scip)
    SCIP_RETCODE SCIPaddSolFree(SCIP* scip, SCIP_SOL** sol, SCIP_Bool* stored)
    SCIP_RETCODE SCIPaddSol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool* stored)
    SCIP_RETCODE SCIPreadSol(SCIP* scip, const char* filename)
    SCIP_RETCODE SCIPreadSolFile(SCIP* scip, const char* filename, SCIP_SOL* sol, SCIP_Bool xml, SCIP_Bool*	partial, SCIP_Bool*	error)
    SCIP_RETCODE SCIPcheckSol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* feasible)
    SCIP_RETCODE SCIPcheckSolOrig(SCIP* scip, SCIP_SOL* sol, SCIP_Bool* feasible, SCIP_Bool printreason, SCIP_Bool completely)

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
    SCIP_EVENTHDLR* SCIPfindEventhdlr(SCIP* scip, const char* name)
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
                                 SCIP_RETCODE (*sepaexeclp) (SCIP* scip, SCIP_SEPA* sepa, SCIP_RESULT* result, unsigned int allowlocal),
                                 SCIP_RETCODE (*sepaexecsol) (SCIP* scip, SCIP_SEPA* sepa, SCIP_SOL* sol, SCIP_RESULT* result, unsigned int allowlocal),
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


    # Numerical Methods
    SCIP_Real SCIPinfinity(SCIP* scip)
    SCIP_Real SCIPfrac(SCIP* scip, SCIP_Real val)
    SCIP_Real SCIPfeasFrac(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisZero(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasIntegral(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasZero(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisFeasNegative(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisInfinity(SCIP* scip, SCIP_Real val)
    SCIP_Bool SCIPisLE(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisLT(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisGE(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisGT(SCIP* scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisEQ(SCIP *scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisFeasEQ(SCIP *scip, SCIP_Real val1, SCIP_Real val2)
    SCIP_Bool SCIPisHugeValue(SCIP *scip, SCIP_Real val)
    SCIP_Bool SCIPisPositive(SCIP *scip, SCIP_Real val)
    SCIP_Bool SCIPisNegative(SCIP *scip, SCIP_Real val)
    SCIP_Bool SCIPisIntegral(SCIP *scip, SCIP_Real val)

    # Statistic Methods
    SCIP_RETCODE SCIPprintStatistics(SCIP* scip, FILE* outfile)
    SCIP_Longint SCIPgetNNodes(SCIP* scip)
    SCIP_Longint SCIPgetNTotalNodes(SCIP* scip)
    SCIP_Longint SCIPgetNFeasibleLeaves(SCIP* scip)
    SCIP_Longint SCIPgetNInfeasibleLeaves(SCIP* scip)
    SCIP_Longint SCIPgetNLPs(SCIP* scip)
    SCIP_Longint SCIPgetNLPIterations(SCIP* scip)

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
    SCIP_RETCODE SCIPlpiGetObjval(SCIP_LPI* lpi, SCIP_Real* objval)
    SCIP_RETCODE SCIPlpiGetSol(SCIP_LPI* lpi, SCIP_Real* objval, SCIP_Real* primsol, SCIP_Real* dualsol, SCIP_Real* activity, SCIP_Real* redcost)
    SCIP_RETCODE SCIPlpiGetIterations(SCIP_LPI* lpi, int* iterations)
    SCIP_RETCODE SCIPlpiGetPrimalRay(SCIP_LPI* lpi, SCIP_Real* ray)
    SCIP_RETCODE SCIPlpiGetDualfarkas(SCIP_LPI* lpi, SCIP_Real* dualfarkas)
    SCIP_RETCODE SCIPlpiGetBasisInd(SCIP_LPI* lpi, int* bind)
    SCIP_RETCODE SCIPlpiGetRealSolQuality(SCIP_LPI* lpi, SCIP_LPSOLQUALITY qualityindicator, SCIP_Real* quality)
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
    SCIP_Real SCIPgetActivityLinear(SCIP* scip, SCIP_CONS* cons, SCIP_SOL* sol)
    SCIP_VAR** SCIPgetVarsLinear(SCIP* scip, SCIP_CONS* cons)
    int SCIPgetNVarsLinear(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real* SCIPgetValsLinear(SCIP* scip, SCIP_CONS* cons)

cdef extern from "scip/cons_quadratic.h":
    SCIP_RETCODE SCIPcreateConsQuadratic(SCIP* scip,
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
    SCIP_RETCODE SCIPaddLinearVarQuadratic(SCIP* scip,
                                           SCIP_CONS* cons,
                                           SCIP_VAR* var,
                                           SCIP_Real coef)
    SCIP_RETCODE SCIPaddBilinTermQuadratic(SCIP* scip,
                                           SCIP_CONS* cons,
                                           SCIP_VAR* var1,
                                           SCIP_VAR* var2,
                                           SCIP_Real coef)
    SCIP_RETCODE SCIPchgLhsQuadratic(SCIP* scip, SCIP_CONS* cons, SCIP_Real lhs)
    SCIP_RETCODE SCIPchgRhsQuadratic(SCIP* scip, SCIP_CONS* cons, SCIP_Real rhs)
    SCIP_Real SCIPgetLhsQuadratic(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real SCIPgetRhsQuadratic(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPgetActivityQuadratic(SCIP* scip, SCIP_CONS* cons, SCIP_SOL* sol, SCIP_Real* activity)
    SCIP_BILINTERM* SCIPgetBilinTermsQuadratic(SCIP* scip, SCIP_CONS* cons)
    int SCIPgetNBilinTermsQuadratic(SCIP* scip, SCIP_CONS* cons)
    SCIP_QUADVARTERM* SCIPgetQuadVarTermsQuadratic(SCIP* scip, SCIP_CONS* cons)
    int SCIPgetNQuadVarTermsQuadratic(SCIP* scip, SCIP_CONS* cons)
    SCIP_VAR** SCIPgetLinearVarsQuadratic(SCIP* scip, SCIP_CONS* cons)
    SCIP_Real* SCIPgetCoefsLinearVarsQuadratic(SCIP* scip, SCIP_CONS* cons)
    int SCIPgetNLinearVarsQuadratic(SCIP* scip, SCIP_CONS* cons)

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

cdef extern from "blockmemshell/memory.h":
    void BMScheckEmptyMemory()
    long long BMSgetMemoryUsed()

cdef extern from "nlpi/pub_expr.h":
    SCIP_RETCODE SCIPexprCreate(BMS_BLKMEM* blkmem,
                                SCIP_EXPR** expr,
                                SCIP_EXPROP op,
                                ...)
    SCIP_RETCODE SCIPexprCreateMonomial(BMS_BLKMEM* blkmem,
                                        SCIP_EXPRDATA_MONOMIAL** monomial,
                                        SCIP_Real coef,
                                        int nfactors,
                                        int* childidxs,
                                        SCIP_Real* exponents)
    SCIP_RETCODE SCIPexprCreatePolynomial(BMS_BLKMEM* blkmem,
                                          SCIP_EXPR** expr,
                                          int nchildren,
                                          SCIP_EXPR** children,
                                          int nmonomials,
                                          SCIP_EXPRDATA_MONOMIAL** monomials,
                                          SCIP_Real constant,
                                          SCIP_Bool copymonomials)
    SCIP_RETCODE SCIPexprtreeCreate(BMS_BLKMEM* blkmem,
                                    SCIP_EXPRTREE** tree,
                                    SCIP_EXPR* root,
                                    int nvars,
                                    int nparams,
                                    SCIP_Real* params)
    SCIP_RETCODE SCIPexprtreeFree(SCIP_EXPRTREE** tree)

cdef extern from "scip/pub_nlp.h":
    SCIP_RETCODE SCIPexprtreeSetVars(SCIP_EXPRTREE* tree,
                                     int nvars,
                                     SCIP_VAR** vars)


    SCIP_Real SCIPnlrowGetConstant(SCIP_NLROW* nlrow)
    int SCIPnlrowGetNLinearVars(SCIP_NLROW* nlrow)
    SCIP_VAR** SCIPnlrowGetLinearVars(SCIP_NLROW* nlrow)
    SCIP_Real* SCIPnlrowGetLinearCoefs(SCIP_NLROW* nlrow)
    void SCIPnlrowGetQuadData(SCIP_NLROW* nlrow,
                              int* nquadvars,
                              SCIP_VAR*** quadvars,
                              int* nquadelems,
                              SCIP_QUADELEM** quadelems)
    SCIP_EXPRTREE* SCIPnlrowGetExprtree(SCIP_NLROW* nlrow)
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

cdef extern from "scip/cons_nonlinear.h":
    SCIP_RETCODE SCIPcreateConsNonlinear(SCIP* scip,
                                         SCIP_CONS** cons,
                                         const char* name,
                                         int nlinvars,
                                         SCIP_VAR** linvars,
                                         SCIP_Real* lincoefs,
                                         int nexprtrees,
                                         SCIP_EXPRTREE** exprtrees,
                                         SCIP_Real* nonlincoefs,
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
    int SCIProwGetLPPos(SCIP_ROW* row)
    SCIP_BASESTAT SCIProwGetBasisStatus(SCIP_ROW* row)
    SCIP_Bool SCIProwIsIntegral(SCIP_ROW* row)
    SCIP_Bool SCIProwIsLocal(SCIP_ROW* row)
    SCIP_Bool SCIProwIsModifiable(SCIP_ROW* row)
    SCIP_Bool SCIProwIsRemovable(SCIP_ROW* row)
    int SCIProwGetNNonz(SCIP_ROW* row)
    int SCIProwGetNLPNonz(SCIP_ROW* row)
    SCIP_COL** SCIProwGetCols(SCIP_ROW* row)
    SCIP_Real* SCIProwGetVals(SCIP_ROW* row)
    SCIP_Real SCIProwGetNorm(SCIP_ROW* row)
    SCIP_Real SCIProwGetDualsol(SCIP_ROW* row)
    int SCIProwGetAge(SCIP_ROW* row)
    SCIP_Bool SCIProwIsRemovable(SCIP_ROW* row)
    SCIP_ROWORIGINTYPE SCIProwGetOrigintype(SCIP_ROW* row)

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
    int SCIPcolGetIndex(SCIP_COL* col)
    SCIP_Real SCIPcolGetObj(SCIP_COL *col)

cdef extern from "scip/scip_tree.h":
    SCIP_RETCODE SCIPgetOpenNodesData(SCIP* scip, SCIP_NODE*** leaves, SCIP_NODE*** children, SCIP_NODE*** siblings, int* nleaves, int* nchildren, int* nsiblings)
    SCIP_Longint SCIPgetNLeaves(SCIP* scip)
    SCIP_Longint SCIPgetNChildren(SCIP* scip)
    SCIP_Longint SCIPgetNSiblings(SCIP* scip)
    SCIP_NODE* SCIPgetBestChild(SCIP* scip)
    SCIP_NODE* SCIPgetBestSibling(SCIP* scip)
    SCIP_NODE* SCIPgetBestLeaf(SCIP* scip)
    SCIP_NODE* SCIPgetBestNode(SCIP* scip)
    SCIP_NODE* SCIPgetBestboundNode(SCIP* scip)
    SCIP_RETCODE SCIPrepropagateNode(SCIP* scip, SCIP_NODE* node)

cdef extern from "scip/scip_var.h":
    SCIP_RETCODE SCIPchgVarBranchPriority(SCIP* scip, SCIP_VAR* var, int branchpriority)

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

    @staticmethod
    cdef create(SCIP* scip)
