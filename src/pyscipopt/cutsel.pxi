##@file cutsel.pxi
#@brief Base class of the Cutsel Plugin
cdef class Cutsel:
  cdef public Model model

  def cutselfree(self):
    '''frees memory of cut selector'''
    pass

  def cutselinit(self):
    ''' executed after the problem is transformed. use this call to initialize cut selector data.'''
    pass

  def cutselexit(self):
    '''executed before the transformed problem is freed'''
    pass

  def cutselinitsol(self):
    '''executed when the presolving is finished and the branch-and-bound process is about to begin'''
    pass

  def cutselexitsol(self):
    '''executed before the branch-and-bound process is freed'''
    pass

  def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
    '''first method called in each iteration in the main solving loop. '''
    # this method needs to be implemented by the user
    return {}


cdef SCIP_RETCODE PyCutselCopy (SCIP* scip, SCIP_CUTSEL* cutsel) noexcept with gil:
  return SCIP_OKAY

cdef SCIP_RETCODE PyCutselFree (SCIP* scip, SCIP_CUTSEL* cutsel) noexcept with gil:
  cdef SCIP_CUTSELDATA* cutseldata
  cutseldata = SCIPcutselGetData(cutsel)
  PyCutsel = <Cutsel>cutseldata
  PyCutsel.cutselfree()
  Py_DECREF(PyCutsel)
  return SCIP_OKAY

cdef SCIP_RETCODE PyCutselInit (SCIP* scip, SCIP_CUTSEL* cutsel) noexcept with gil:
  cdef SCIP_CUTSELDATA* cutseldata
  cutseldata = SCIPcutselGetData(cutsel)
  PyCutsel = <Cutsel>cutseldata
  PyCutsel.cutselinit()
  return SCIP_OKAY


cdef SCIP_RETCODE PyCutselExit (SCIP* scip, SCIP_CUTSEL* cutsel) noexcept with gil:
  cdef SCIP_CUTSELDATA* cutseldata
  cutseldata = SCIPcutselGetData(cutsel)
  PyCutsel = <Cutsel>cutseldata
  PyCutsel.cutselexit()
  return SCIP_OKAY

cdef SCIP_RETCODE PyCutselInitsol (SCIP* scip, SCIP_CUTSEL* cutsel) noexcept with gil:
  cdef SCIP_CUTSELDATA* cutseldata
  cutseldata = SCIPcutselGetData(cutsel)
  PyCutsel = <Cutsel>cutseldata
  PyCutsel.cutselinitsol()
  return SCIP_OKAY

cdef SCIP_RETCODE PyCutselExitsol (SCIP* scip, SCIP_CUTSEL* cutsel) noexcept with gil:
  cdef SCIP_CUTSELDATA* cutseldata
  cutseldata = SCIPcutselGetData(cutsel)
  PyCutsel = <Cutsel>cutseldata
  PyCutsel.cutselexitsol()
  return SCIP_OKAY

cdef SCIP_RETCODE PyCutselSelect (SCIP* scip, SCIP_CUTSEL* cutsel, SCIP_ROW** cuts, int ncuts,
                                  SCIP_ROW** forcedcuts, int nforcedcuts, SCIP_Bool root, int maxnselectedcuts,
                                  int* nselectedcuts, SCIP_RESULT* result) noexcept with gil:
  cdef SCIP_CUTSELDATA* cutseldata
  cdef SCIP_ROW* scip_row
  cutseldata = SCIPcutselGetData(cutsel)
  PyCutsel = <Cutsel>cutseldata

  # translate cuts to python
  pycuts = [Row.create(cuts[i]) for i in range(ncuts)]
  pyforcedcuts = [Row.create(forcedcuts[i]) for i in range(nforcedcuts)]
  result_dict = PyCutsel.cutselselect(pycuts, pyforcedcuts, root, maxnselectedcuts)

  # Retrieve the sorted cuts. Note that these do not need to be returned explicitly in result_dict.
  # Pycuts could have been sorted in place in cutselselect()
  pycuts = result_dict.get('cuts', pycuts)

  assert len(pycuts) == ncuts
  assert len(pyforcedcuts) == nforcedcuts

  #sort cuts
  for i,cut in enumerate(pycuts):
    cuts[i] = <SCIP_ROW *>((<Row>cut).scip_row)

  nselectedcuts[0] = result_dict.get('nselectedcuts', 0)
  result[0] = result_dict.get('result', <SCIP_RESULT>result[0])

  return SCIP_OKAY
