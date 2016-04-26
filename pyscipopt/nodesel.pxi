cdef SCIP_RETCODE PyNodeselCopy (SCIP* scip, SCIP_NODESEL* nodesel):
    return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselFree (SCIP* scip, SCIP_NODESEL* nodesel):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    PyNodesel.free()
    Py_DECREF(PyNodesel)
    return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselInit (SCIP* scip, SCIP_NODESEL* nodesel):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    PyNodesel.init()
    return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselExit (SCIP* scip, SCIP_NODESEL* nodesel):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    PyNodesel.exit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselInitsol (SCIP* scip, SCIP_NODESEL* nodesel):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    PyNodesel.initsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselExitsol (SCIP* scip, SCIP_NODESEL* nodesel):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    PyNodesel.exitsol()
    return SCIP_OKAY


cdef SCIP_RETCODE PyNodeselSelect (SCIP* scip, SCIP_NODESEL* nodesel, SCIP_NODE** selnode):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    selnode[0] = NULL #PyNodesel.select()
    return SCIP_OKAY

cdef int PyNodeselComp (SCIP* scip, SCIP_NODESEL* nodesel, SCIP_NODE* node1, SCIP_NODE* node2):
    cdef SCIP_NODESELDATA* nodeseldata
    nodeseldata = SCIPnodeselGetData(nodesel)
    PyNodesel = <Nodesel>nodeseldata
    value = PyNodesel.select()
    return value

cdef class Nodesel:
    cdef public object data     # storage for the python user
    cdef public Model model

    def free(self):
        pass

    def init(self):
        pass

    def exit(self):
        pass

    def initsol(self):
        pass

    def exitsol(self):
        pass

    def select(self):
        # this method needs to be implemented by the user
        cdef SCIP_NODE* selnode #@todo
        selnode = NULL

        return {"selnode": 0}

    def comp(self):
        # this method needs to be implemented by the user
        return {"value": 0}
