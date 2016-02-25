cdef SCIP_RETCODE PySepaCopy (SCIP* scip, SCIP_SEPA* sepa):
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaFree (SCIP* scip, SCIP_SEPA* sepa):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    PySepa.sepafree()
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaInit (SCIP* scip, SCIP_SEPA* sepa):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    PySepa.sepainit()
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaExit (SCIP* scip, SCIP_SEPA* sepa):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    PySepa.sepaexit()
    return SCIP_OKAY


cdef SCIP_RETCODE PySepaInitsol (SCIP* scip, SCIP_SEPA* sepa):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    PySepa.sepainitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaExitsol (SCIP* scip, SCIP_SEPA* sepa):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    PySepa.sepaexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaExeclp (SCIP* scip, SCIP_SEPA* sepa, SCIP_RESULT* result):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    result[0] = PySepa.sepaexeclp()
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaExecsol (SCIP* scip, SCIP_SEPA* sepa, SCIP_SOL* sol, SCIP_RESULT* result):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    solution = Solution()
    solution._solution = sol
    PySepa = <Sepa>sepadata
    result[0] = PySepa.sepaexecsol(solution)
    return SCIP_OKAY

cdef class Sepa:
    cdef public object data     # storage for the python user
    cdef public Model model

    def sepafree(self):
        pass

    def sepainit(self):
        pass

    def sepaexit(self):
        pass

    def sepainitsol(self):
        pass

    def sepaexitsol(self):
        pass

    def sepaexeclp(self):
        pass

    def sepaexecsol(self, solution):
        pass
