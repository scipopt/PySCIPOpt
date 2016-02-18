# Copyright (C) 2012-2013 Robert Schwarz
#   see file 'LICENSE' for details.

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

    ctypedef enum SCIP_PARAMSETTING:
        SCIP_PARAMSETTING_DEFAULT    = 0
        SCIP_PARAMSETTING_AGGRESSIVE = 1
        SCIP_PARAMSETTING_FAST       = 2
        SCIP_PARAMSETTING_OFF        = 3

    ctypedef enum SCIP_PROPTIMING:
        SCIP_PROPTIMING_BEFORELP     = 0x001u
        SCIP_PROPTIMING_DURINGLPLOOP = 0x002u
        SCIP_PROPTIMING_AFTERLPLOOP  = 0x004u
        SCIP_PROPTIMING_AFTERLPNODE  = 0x008u

    ctypedef enum SCIP_PRESOLTIMING:
        SCIP_PRESOLTIMING_NONE       = 0x000u
        SCIP_PRESOLTIMING_FAST       = 0x002u
        SCIP_PRESOLTIMING_MEDIUM     = 0x004u
        SCIP_PRESOLTIMING_EXHAUSTIVE = 0x008u

    ctypedef bint SCIP_Bool

    ctypedef long SCIP_Longint

    ctypedef float SCIP_Real

    ctypedef struct SCIP:
        pass

    ctypedef struct SCIP_VAR:
        pass

    ctypedef struct SCIP_CONS:
        pass

    ctypedef struct SCIP_ROW:
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

    # General SCIP Methods
    SCIP_RETCODE SCIPcreate(SCIP** scip)
    SCIP_RETCODE SCIPfree(SCIP** scip)
    void SCIPsetMessagehdlrQuiet(SCIP* scip, SCIP_Bool quiet)
    void SCIPprintVersion(SCIP* scip, FILE* outfile)

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
    SCIP_RETCODE SCIPwriteOrigProblem(SCIP* scip, char* filename, char* extension, SCIP_Bool genericnames)
    SCIP_STATUS SCIPgetStatus(SCIP* scip)

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
    SCIP_RETCODE SCIPcaptureVar(SCIP* scip, SCIP_VAR* var)
    SCIP_RETCODE SCIPaddPricedVar(SCIP* scip, SCIP_VAR* var, SCIP_Real score)
    SCIP_RETCODE SCIPreleaseVar(SCIP* scip, SCIP_VAR** var)
    SCIP_RETCODE SCIPtransformVar(SCIP* scip, SCIP_VAR* var, SCIP_VAR** transvar)
    SCIP_VAR** SCIPgetVars(SCIP* scip)
    const char* SCIPvarGetName(SCIP_VAR* var)
    int SCIPgetNVars(SCIP* scip)

    # Constraint Methods
    SCIP_RETCODE SCIPcaptureCons(SCIP* scip, SCIP_CONS* cons)
    SCIP_RETCODE SCIPreleaseCons(SCIP* scip, SCIP_CONS** cons)
    SCIP_RETCODE SCIPtransformCons(SCIP* scip, SCIP_CONS* cons, SCIP_CONS** transcons)
    SCIP_CONS** SCIPgetConss(SCIP* scip)
    const char* SCIPconsGetName(SCIP_CONS* cons)
    int SCIPgetNConss(SCIP* scip)

    # Primal Solution Methods
    SCIP_SOL** SCIPgetSols(SCIP* scip)
    int SCIPgetNSols(SCIP* scip)
    SCIP_SOL* SCIPgetBestSol(SCIP* scip)
    SCIP_Real SCIPgetSolVal(SCIP* scip, SCIP_SOL* sol, SCIP_VAR* var)
    SCIP_RETCODE SCIPwriteVarName(SCIP* scip, FILE* outfile, SCIP_VAR* var, SCIP_Bool vartype)
    SCIP_Real SCIPgetSolOrigObj(SCIP* scip, SCIP_SOL* sol)
    SCIP_Real SCIPgetSolTransObj(SCIP* scip, SCIP_SOL* sol)

    # Dual Solution Methods
    SCIP_Real SCIPgetDualbound(SCIP* scip)

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
                                     SCIP_RETCODE (*consinitlp) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss),
                                     SCIP_RETCODE (*conssepalp) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conssepasol) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_SOL* sol, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consenfolp) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*consenfops) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, int nusefulconss, SCIP_Bool solinfeasible, SCIP_Bool objinfeasible, SCIP_RESULT* result),
                                     SCIP_RETCODE (*conscheck) (SCIP* scip, SCIP_CONSHDLR* conshdlr, SCIP_CONS** conss, int nconss, SCIP_SOL* sol, SCIP_Bool checkintegrality, SCIP_Bool checklprows, SCIP_Bool printreason, SCIP_RESULT* result),
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

    # Numerical Methods
    SCIP_Real SCIPinfinity(SCIP* scip)

    # Statistic Methods
    SCIP_RETCODE SCIPprintStatistics(SCIP* scip, FILE* outfile)


    # Parameter Functions
    SCIP_RETCODE SCIPsetBoolParam(SCIP* scip, char* name, SCIP_Bool value)
    SCIP_RETCODE SCIPsetIntParam(SCIP* scip, char* name, int value)
    SCIP_RETCODE SCIPsetLongintParam(SCIP* scip, char* name, SCIP_Longint value)
    SCIP_RETCODE SCIPsetRealParam(SCIP* scip, char* name, SCIP_Real value)
    SCIP_RETCODE SCIPsetCharParam(SCIP* scip, char* name, char value)
    SCIP_RETCODE SCIPsetStringParam(SCIP* scip, char* name, char* value)
    SCIP_RETCODE SCIPreadParams(SCIP* scip, char* file)
    SCIP_RETCODE SCIPreadProb(SCIP* scip, char* file, char* extension)


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
