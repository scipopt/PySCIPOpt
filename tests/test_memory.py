from pyscipopt.scip import Model, is_memory_freed

def test_not_freed():
    m = Model()
    assert not is_memory_freed()

def test_freed():
    m = Model()
    del m
    assert is_memory_freed()

if __name__ == "__main__":
    test_not_freed()
    test_freed()
