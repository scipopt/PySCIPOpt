cdef SCIP_RETCODE PyReaderCopy (SCIP* scip, SCIP_READER* reader):
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderFree (SCIP* scip, SCIP_READER* reader):
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    PyReader.free()
    Py_DECREF(PyReader)
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderRead (SCIP* scip, SCIP_READER* reader, const char* file, SCIP_RESULT* result):
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    result_dict = PyReader.read(file)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderWrite (SCIP* scip, SCIP_READER* reader, FILE* file,
                                 const char* name, SCIP_PROBDATA* probdata, SCIP_Bool transformed,
                                 SCIP_OBJSENSE objsense, SCIP_Real objscale, SCIP_Real objoffset,
                                 SCIP_VAR** vars, int nvars, int nbinvars, int nintvars, int nimplvars, int ncontvars,
                                 SCIP_VAR** fixedvars, int nfixedvars, int startnvars,
                                 SCIP_CONS** conss, int nconss, int maxnconss, int startnconss,
                                 SCIP_Bool genericnames, SCIP_RESULT* result):
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    # TODO this needs a proper implementation
    #result[0] = PyReader.write(file, name, probdata, transformed, objsense, objscale, objoffset,
                               #vars, nvars, nbinvars, nintvars, nimplvars, ncontvars,
                               #fixedvars, nfixedvars, startnvars, conss, nconss, maxnconss, startnconss,
                               #genericnames)
    return SCIP_OKAY

cdef class Reader:
    cdef public object data     # storage for the python user
    cdef public Model model

    def readerfree(self):
        pass

    def readerinit(self):
        pass

    def readerread(self, file):
        return {}

    def readerwrite(self, file, name, probdata, transformed, objsense, objscale, objoffset,
              vars, nvars, nbinvars, nintvars, nimplvars, ncontvars,
              fixedvars, nfixedvars, startvars, conss, nconss, maxnconss, startnconss,
              genericnames):
        return {}
