import pytest
from pyscipopt.scip import Model, is_memory_freed
from util import is_optimized_mode

def test_not_freed():
    if is_optimized_mode():
       pytest.skip()
    m = Model()
    assert not is_memory_freed()

def test_freed():
    if is_optimized_mode():
       pytest.skip()
    m = Model()
    del m
    assert is_memory_freed()

if __name__ == "__main__":
    test_not_freed()
    test_freed()
