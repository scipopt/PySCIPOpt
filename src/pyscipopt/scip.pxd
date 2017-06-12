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

    ctypedef enum SCIP_PARAMSETTING:
        SCIP_PARAMSETTING_DEFAULT    = 0
        SCIP_PARAMSETTING_AGGRESSIVE = 1
        SCIP_PARAMSETTING_FAST       = 2
        SCIP_PARAMSETTING_OFF        = 3

    ctypedef enum SCIP_PARAMEMPHASIS:
        SCIP_PARAMEMPHASIS_DEFAULT      = 0
        SCIP_PARAMEMPHASIS_CPSOLVER     = 1
        SCIP_PARAMEMPHASIS_EASYCIP      = 2
        SCIP_PARAMEMPHASIS_FEASIBILITY  = 3
        SCIP_PARAMEMPHASIS_HARDLP       = 4
        SCIP_PARAMEMPHASIS_OPTIMALITY   = 5
        SCIP_PARAMEMPHASIS_COUNTER      = 6
        #SCIP_PARAMEMPHASIS_PHASEFEAS    = 7
        #SCIP_PARAMEMPHASIS_PHASEIMPROVE = 8
        #SCIP_PARAMEMPHASIS_PHASEPROOF   = 9

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

    # General SCIP Methods
    SCIP_RETCODE SCIPcreate(SCIP** scip)
    SCIP_RETCODE SCIPfree(SCIP** scip)
    void SCIPsetMessagehdlrQuiet(SCIP* scip, SCIP_Bool quiet)
    void SCIPprintVersion(SCIP* scip, FILE* outfile)
    SCIP_Real SCIPgetTotalTime(SCIP* scip)
    SCIP_Real SCIPgetSolvingTime(SCIP* scip)
    SCIP_Real SCIPgetReadingTime(SCIP* scip)
    SCIP_Real SCIPgetPresolvingTime(SCIP* scip)
    SCIP_STAGE SCIPgetStage(SCIP* scip)

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
    SCIP_RETCODE SCIPsetPresolving(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPsetSeparating(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPsetHeuristics(SCIP* scip, SCIP_PARAMSETTING paramsetting, SCIP_Bool quiet)
    SCIP_RETCODE SCIPwriteOrigProblem(SCIP* scip, char* filename, char* extension, SCIP_Bool genericnames)
    SCIP_STATUS SCIPgetStatus(SCIP* scip)
    SCIP_Real SCIPepsilon(SCIP* scip)
    SCIP_Real SCIPfeastol(SCIP* scip)

    # Solve Methods
    SCIP_RETCODE SCIPsolve(SCIP* scip)
    SCIP_RETCODE SCIPfreeTransform(SCIP* scip)

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
    SCIP_RETCODE SCIPchgVarType(SCIP* scip, SCIP_VAR* var, SCIP_VARTYPE vartype, SCIP_Bool* infeasible)
    SCIP_RETCODE SCIPcaptureVar(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPaddPricedVar(SCIP* scip, SCIP_VAR* var, SCIP_Real score)
    SCIP_RETCODE SCIPreleaseVar(SCIP* scip, SCIP_VAR** var)
    SCIP_RETCODE SCIPtransformVar(SCIP* scip, SCIP_VAR* var, SCIP_VAR** transvar)
    SCIP_RETCODE SCIPaddVarLocks(SCIP* scip, SCIP_VAR* var, int nlocksdown, int nlocksup)
    SCIP_VAR** SCIPgetVars(SCIP* scip)
    SCIP_VAR** SCIPgetOrigVars(SCIP* scip)
    const char* SCIPvarGetName(SCIP_VAR* var)
    int SCIPgetNVars(SCIP* scip)
    int SCIPgetNOrigVars(SCIP* scip)
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
    SCIP_CONSDATA* SCIPconsGetData(SCIP_CONS* cons)
    SCIP_CONSHDLR* SCIPconsGetHdlr(SCIP_CONS* cons)
    const char* SCIPconshdlrGetName(SCIP_CONSHDLR* conshdlr)
    SCIP_RETCODE SCIPdelConsLocal(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPdelCons(SCIP* scip, SCIP_CONS* cons)

    # Primal Solution Methods
    SCIP_SOL** SCIPgetSols(SCIP* scip)
    int SCIPgetNSols(SCIP* scip)
    SCIP_SOL* SCIPgetBestSol(SCIP* scip)
    SCIP_Real SCIPgetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var)
    SCIP_RETCODE SCIPwriteVarName(SCIP* scip, FILE* outfile, SCIP_VAR* var, SCIP_Bool vartype)
    SCIP_Real SCIPgetSolOrigObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_Real SCIPgetSolTransObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_RETCODE SCIPcreateSol(SCIP* scip, SCIP_SOL** sol, SCIP_HEUR* heur)
    SCIP_RETCODE SCIPsetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var, SCIP_Real val)
    SCIP_RETCODE SCIPtrySolFree(SCIP* scip, SCIP_SOL** sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* stored)
    SCIP_RETCODE SCIPtrySol(SCIP* scip, SCIP_SOL* sol, SCIP_Bool printreason, SCIP_Bool completely, SCIP_Bool checkbounds, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool* stored)
    SCIP_RETCODE SCIPfreeSol(SCIP* scip, SCIP_SOL** sol)
    SCIP_RETCODE SCIPprintBestSol(SCIP* scip, FILE* outfile, SCIP_Bool printzeros)
    SCIP_Real SCIPgetPrimalbound(SCIP* scip)
    SCIP_Real SCIPgetGap(SCIP* scip)

    # Row Methods
    SCIP_RETCODE SCIPcreateRow(SCIP* scip, SCIP_ROW** row, const char* name, int len, SCIP_COL** cols, SCIP_Real* vals,
                               SCIP_Real lhs, SCIP_Real rhs, SCIP_Bool local, SCIP_Bool modifiable, SCIP_Bool removable)
    SCIP_RETCODE SCIPaddCut(SCIP* scip, SCIP_SOL* sol, SCIP_ROW* row, SCIP_Bool forcecut, SCIP_Bool* infeasible)

    # Dual Solution Methods
    SCIP_Real SCIPgetDualbound(SCIP* scip)
    SCIP_Real SCIPgetDualboundRoot(SCIP* scip)
    SCIP_Real SCIPgetVarRedcost(SCIP* scip, SCIP_VAR* var)

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
                                     SCIP_RETCODE (*consenfolp) (SCIP* scip, SCIP_SOL* sol, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consenfops) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_Bool objinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conscheck) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_SOL* sol, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool printreason, SCIP_Bool completely, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consprop) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, int nmarkedconss, SCIP_PROPTIMING proptiming, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conspresol) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nrounds, SCIP_PRESOLTIMING presoltiming, int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes, int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides, int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes, int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consresprop) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, SCIP_VAR* infervar, int inferinfo, SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conslock) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS* cons, int nlockspos, int nlocksneg),
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
                                 SCIP_RETCODE (*sepaexeclp) (SCIP* scip, SCIP_SEPA* sepa, SCIP_RESULT* result),
                                 SCIP_RETCODE (*sepaexecsol) (SCIP* scip, SCIP_SEPA* sepa, SCIP_SOL* sol, SCIP_RESULT* result),
                                 SCIP_SEPADATA* sepadata)
    SCIP_SEPADATA* SCIPsepaGetData(SCIP_SEPA* sepa)

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

    # Numerical Methods
    SCIP_Real SCIPinfinity(SCIP* scip)

    # Statistic Methods
    SCIP_RETCODE SCIPprintStatistics(SCIP* scip, FILE* outfile)
    SCIP_Longint SCIPgetNNodes(SCIP* scip)

    # Parameter Functions
    SCIP_RETCODE SCIPsetBoolParam(SCIP* scip, char* name, SCIP_Bool value)
    SCIP_RETCODE SCIPsetIntParam(SCIP* scip, char* name, int value)
    SCIP_RETCODE SCIPsetLongintParam(SCIP* scip, char* name, SCIP_Longint value)
    SCIP_RETCODE SCIPsetRealParam(SCIP* scip, char* name, SCIP_Real value)
    SCIP_RETCODE SCIPsetCharParam(SCIP* scip, char* name, char value)
    SCIP_RETCODE SCIPsetStringParam(SCIP* scip, char* name, char* value)
    SCIP_RETCODE SCIPreadParams(SCIP* scip, char* file)
    SCIP_RETCODE SCIPreadProb(SCIP* scip, char* file, char* extension)
    SCIP_RETCODE SCIPsetEmphasis(SCIP* scip, SCIP_PARAMEMPHASIS paramemphasis, SCIP_Bool quiet)

    # LPI Functions
    SCIP_RETCODE SCIPlpiCreate(SCIP_LPI** lpi, SCIP_MESSAGEHDLR* messagehdlr, const char* name, SCIP_OBJSENSE objsen)
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
    SCIP_Bool    SCIPlpiHasPrimalRay(SCIP_LPI* lpi)
    SCIP_Bool    SCIPlpiHasDualRay(SCIP_LPI* lpi)
    SCIP_Real    SCIPlpiInfinity(SCIP_LPI* lpi)
    SCIP_Bool    SCIPlpiIsInfinity(SCIP_LPI* lpi, SCIP_Real val)
    SCIP_Bool    SCIPlpiIsPrimalFeasible(SCIP_LPI* lpi)
    SCIP_Bool    SCIPlpiIsDualFeasible(SCIP_LPI* lpi)

    BMS_BLKMEM* SCIPblkmem(SCIP* scip)

cdef extern from "scip/scipdefplugins.h":
    SCIP_RETCODE SCIPincludeDefaultPlugins(SCIP* scip)

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
