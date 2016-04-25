from pyscipopt.scip import Model, is_memory_freed

def is_optimized_mode():
    s = Model()
    return is_memory_freed()


