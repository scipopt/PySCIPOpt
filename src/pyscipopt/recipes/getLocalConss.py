from pyscipopt import Model, Constraint

def getLocalConss(model: Model, node = None) -> list[Constraint]:
    """
    Returns local constraints.

    Parameters
    ----------
    model : Model
        The model from which to retrieve the local constraints.
    node : Node, optional
        The node from which to retrieve the local constraints. If not provided, the current node is used.

    Returns
    -------
    list[Constraint]
        A list of local constraints. First entry are global constraints, second entry are all the added constraints.
    """

    if not node:
        assert model.getStageName() in ["INITPRESOLVE", "PRESOLVING", "EXITPRESOLVE", "SOLVING"], "Model cannot be called in stage %s." % model.getStageName()
        cur_node = model.getCurrentNode()
    else:
        cur_node = node

    added_conss = []
    while cur_node is not None:
        added_conss = cur_node.getAddedConss() + added_conss
        cur_node = cur_node.getParent()
    
    return [model.getConss(), added_conss]

def getNAddedConss(model: Model) -> int:
    """
    Returns the number of local constraints of a node.
    """
    return len(getLocalConss(model)[1])