"""Test for getMaxDepth() method"""
from pyscipopt import Model
from helpers.utils import random_mip_1

def test_getMaxDepth():
    m = random_mip_1(
        disable_sepa=True,
        disable_heur=True,
        disable_presolve=True,  # This ensures branching happens
        node_lim=100,  # Limit nodes so test runs quickly
        small=True  # Use smaller version for testing
    )
    
    print(f"Initial max depth: {m.getMaxDepth()}")
    assert m.getMaxDepth() == -1
    
    m.optimize()

    max_depth = m.getMaxDepth()
    nodes = m.getNNodes()

    print(f"Max depth after solving: {max_depth}")
    print(f"Number of nodes explored: {nodes}")
    print(f"Optimization status: {m.getStatus()}")


    assert max_depth >= 0, f"Expected max_depth >= 0, got {max_depth}"

    if nodes > 1:
        assert max_depth >= 1, f"Expected max_depth >= 1 with {nodes} nodes, got {max_depth}"


    assert max_depth <= nodes, f"Max depth {max_depth} shouldn't exceed nodes {nodes}"

if __name__ == "__main__":
    test_getMaxDepth()