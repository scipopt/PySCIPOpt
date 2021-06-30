##@file nodesel.pxi
#@brief Base class of the Nodesel Plugin
cdef class Nodesel:
  cdef public Model model

  def nodefree(self):
    '''frees memory of node selector'''
    pass

  def nodeinit(self):
    ''' executed after the problem is transformed. use this call to initialize node selector data.'''
    pass

  def nodeexit(self):
    '''executed before the transformed problem is freed'''
    pass

  def nodeinitsol(self):
    '''executed when the presolving is finished and the branch-and-bound process is about to begin'''
    pass

  def nodeexitsol(self):
    '''executed before the branch-and-bound process is freed'''
    pass

  def nodeselect(self):
    '''first method called in each iteration in the main solving loop. '''
    # this method needs to be implemented by the user
    return {}

  def nodecomp(self, node1, node2):
    '''
    compare two leaves of the current branching tree

    It should return the following values:

      value < 0, if node 1 comes before (is better than) node 2
      value = 0, if both nodes are equally good
      value > 0, if node 1 comes after (is worse than) node 2.
    '''
    # this method needs to be implemented by the user
    return 0


cdef SCIP_RETCODE PyNodeselCopy (SCIP* scip, SCIP_NODESEL* nodesel) with gil:
  return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselFree (SCIP* scip, SCIP_NODESEL* nodesel) with gil:
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  PyNodesel.nodefree()
  Py_DECREF(PyNodesel)
  return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselInit (SCIP* scip, SCIP_NODESEL* nodesel) with gil:
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  PyNodesel.nodeinit()
  return SCIP_OKAY


cdef SCIP_RETCODE PyNodeselExit (SCIP* scip, SCIP_NODESEL* nodesel) with gil:
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  PyNodesel.nodeexit()
  return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselInitsol (SCIP* scip, SCIP_NODESEL* nodesel) with gil:
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  PyNodesel.nodeinitsol()
  return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselExitsol (SCIP* scip, SCIP_NODESEL* nodesel) with gil:
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  PyNodesel.nodeexitsol()
  return SCIP_OKAY

cdef SCIP_RETCODE PyNodeselSelect (SCIP* scip, SCIP_NODESEL* nodesel, SCIP_NODE** selnode) with gil:
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  result_dict = PyNodesel.nodeselect()
  selected_node = <Node>(result_dict.get("selnode", None))
  selnode[0] = selected_node.scip_node
  return SCIP_OKAY

cdef int PyNodeselComp (SCIP* scip, SCIP_NODESEL* nodesel, SCIP_NODE* node1, SCIP_NODE* node2):
  cdef SCIP_NODESELDATA* nodeseldata
  nodeseldata = SCIPnodeselGetData(nodesel)
  PyNodesel = <Nodesel>nodeseldata
  n1 = Node.create(node1)
  n2 = Node.create(node2)
  result = PyNodesel.nodecomp(n1, n2) #
  return result
