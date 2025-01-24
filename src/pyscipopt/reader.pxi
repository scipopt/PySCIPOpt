##@file reader.pxi
#@brief Base class of the Reader Plugin
cdef class Reader:
    cdef public Model model
    cdef public str name

    def readerfree(self):
        '''calls destructor and frees memory of reader'''
        pass

    def readerread(self, filename):
        '''calls read method of reader'''
        return {}

    def readerwrite(self, file, name, transformed, objsense, objscale, objoffset, binvars, intvars,
                    implvars, contvars, fixedvars, startnvars, conss, maxnconss, startnconss, genericnames):
        '''calls write method of reader'''
        return {}


cdef SCIP_RETCODE PyReaderCopy (SCIP* scip, SCIP_READER* reader) noexcept with gil:
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderFree (SCIP* scip, SCIP_READER* reader) noexcept with gil:
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    PyReader.readerfree()
    Py_DECREF(PyReader)
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderRead (SCIP* scip, SCIP_READER* reader, const char* filename, SCIP_RESULT* result) noexcept with gil:
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    PyReader = <Reader>readerdata
    PyFilename = filename.decode('utf-8')
    result_dict = PyReader.readerread(PyFilename)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyReaderWrite (SCIP* scip, SCIP_READER* reader, FILE* file,
                                 const char* name, SCIP_PROBDATA* probdata, SCIP_Bool transformed,
                                 SCIP_OBJSENSE objsense, SCIP_Real objscale, SCIP_Real objoffset, 
                                 SCIP_VAR** vars, int nvars, int nbinvars, int nintvars, int nimplvars, int ncontvars,
                                 SCIP_VAR** fixedvars, int nfixedvars, int startnvars,
                                 SCIP_CONS** conss, int nconss, int maxnconss, int startnconss,
                                 SCIP_Bool genericnames, SCIP_RESULT* result) noexcept with gil:
    cdef SCIP_READERDATA* readerdata
    readerdata = SCIPreaderGetData(reader)
    cdef int fd = fileno(file)
    PyFile = os.fdopen(fd, "w", closefd=False)
    PyName = name.decode('utf-8')
    PyBinVars = [Variable.create(vars[i]) for i in range(nbinvars)]
    PyIntVars = [Variable.create(vars[i]) for i in range(nbinvars, nintvars)]
    PyImplVars = [Variable.create(vars[i]) for i in range(nintvars, nimplvars)]
    PyContVars = [Variable.create(vars[i]) for i in range(nimplvars, ncontvars)]
    PyFixedVars = [Variable.create(fixedvars[i]) for i in range(nfixedvars)]
    PyConss = [Constraint.create(conss[i]) for i in range(nconss)]
    PyReader = <Reader>readerdata
    result_dict = PyReader.readerwrite(PyFile, PyName, transformed, objsense, objscale, objoffset,
                                       PyBinVars, PyIntVars, PyImplVars, PyContVars, PyFixedVars, startnvars,
                                       PyConss, maxnconss, startnconss, genericnames)
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY
