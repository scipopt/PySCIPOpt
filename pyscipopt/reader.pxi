cdef SCIP_RETCODE PyReaderCopy (SCIP* scip, SCIP_READER* reader):
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderFree (SCIP* scip, SCIP_READER* reader):
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    PyReader.free()
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderRead (SCIP* scip, SCIP_READER* reader, const char* file, SCIP_RESULT* result):
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    result[0] = PyReader.read(file)
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

    def free(self):
        pass

    def init(self):
        pass

    def read(self, file):
        return SCIP_DIDNOTRUN

    def write(self, file, name, probdata, transformed, objsense, objscale, objoffset,
              vars, nvars, nbinvars, nintvars, nimplvars, ncontvars,
              fixedvars, nfixedvars, startvars, conss, nconss, maxnconss, startnconss,
              genericnames):
        return SCIP_DIDNOTRUN
