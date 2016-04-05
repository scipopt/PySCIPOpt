cdef class LinExpr:
    cdef public terms

cdef class LinCons:
    cdef public expr
    cdef public lb
    cdef public ub
