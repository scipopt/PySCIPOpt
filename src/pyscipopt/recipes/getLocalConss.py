from pyscipopt import Model, Constraint
from typing import List

def getLocalConss(model: Model, node = None) -> List[List[Constraint]]:
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

    if node is None:
        assert model.getStageName() in ["INITPRESOLVE", "PRESOLVING", "EXITPRESOLVE", "SOLVING"], "Model cannot be called in stage %s." % model.getStageName()
        cur_node = model.getCurrentNode()
    else:
        cur_node = node

    added_conss: List[Constraint] = []
    while cur_node is not None:
        added_conss = cur_node.getAddedConss() + added_conss
        cur_node = cur_node.getParent()
    
    return [model.getConss(), added_conss]

def getNLocalConss(model: Model, node = None) -> List[int]:
    """
    Returns the number of local constraints of a node.

    Parameters
    ----------
    model : Model
        The model from which to retrieve the number of local constraints.
    node : Node, optional
        The node from which to retrieve the number of local constraints. If not provided, the current node is used.

    Returns
    -------
    list[int]
        A list of the number of local constraints. First entry is the number of global constraints, second entry is the number of all the added constraints.
    """
    local_conss = getLocalConss(model, node)
    return [len(local_conss[0]), len(local_conss[1])]