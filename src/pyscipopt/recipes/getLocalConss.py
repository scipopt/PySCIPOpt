from pyscipopt import Model, Constraint

def getLocalConss(model: Model, node = None) -> list[Constraint]:
    """
    Returns the local constraints of a node.
    """

    if not node:
        assert model.getStageName() in ["INITPRESOLVE", "PRESOLVING", "EXITPRESOLVE", "SOLVING"], "Model cannot be called in stage %s." % model.getStageName()
        cur_node = model.getCurrentNode()
    else:
        cur_node = node

    local_conss = []
    while cur_node is not None:
        local_conss = cur_node.getAddedConss() + local_conss
        cur_node = cur_node.getParent()
    return local_conss

def getNLocalConss(model: Model) -> int:
    """
    Returns the number of local constraints of a node.
    """
    return len(getLocalConss(model))