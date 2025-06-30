from pyscipopt import Model

def getLocalConss(model: Model) -> list:
    """
    Returns the local constraints of a node.
    """
    assert model.getStageName() == "SOLVING", "Model must be in SOLVING stage to get local constraints."

    cur_node = model.getCurrentNode()

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