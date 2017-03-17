cdef class Sepa:
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
        return {}

    def sepaexecsol(self, solution):
        return {}



cdef SCIP_RETCODE PySepaCopy (SCIP* scip, SCIP_SEPA* sepa):
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaFree (SCIP* scip, SCIP_SEPA* sepa):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    PySepa = <Sepa>sepadata
    PySepa.sepafree()
    Py_DECREF(PySepa)
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
    result_dict = PySepa.sepaexeclp()
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PySepaExecsol (SCIP* scip, SCIP_SEPA* sepa, SCIP_SOL* sol, SCIP_RESULT* result):
    cdef SCIP_SEPADATA* sepadata
    sepadata = SCIPsepaGetData(sepa)
    solution = Solution()
    solution.sol = sol
    PySepa = <Sepa>sepadata
    result_dict = PySepa.sepaexecsol(solution)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY
