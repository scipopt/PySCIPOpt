"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""

from typing import Optional, Tuple, Union
import numpy as np
try:
    # NumPy 2.x location
    from numpy.lib.array_utils import normalize_axis_tuple
except ImportError:
    # Fallback for NumPy 1.x
    from numpy.core.numeric import normalize_axis_tuple

cimport numpy as cnp

cnp.import_array()


def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, Expr)
        return False


def _matrixexpr_richcmp(self, other, op):
    def _richcmp(self, other, op):
        if op == 1: # <=
            return self.__le__(other)
        elif op == 5: # >=
            return self.__ge__(other)
        elif op == 2: # ==
            return self.__eq__(other)
        else:
            raise NotImplementedError("Can only support constraints with '<=', '>=', or '=='.")

    if _is_number(other) or isinstance(other, Expr):
        res = np.empty(self.shape, dtype=object)
        res.flat = [_richcmp(i, other, op) for i in self.flat]

    elif isinstance(other, np.ndarray):
        out = np.broadcast(self, other)
        res = np.empty(out.shape, dtype=object)
        res.flat = [_richcmp(i, j, op) for i, j in out]

    else:
        raise TypeError(f"Unsupported type {type(other)}")

    return res.view(MatrixExprCons)


class MatrixExpr(np.ndarray):

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == "__call__":
            if ufunc in {np.matmul, np.dot}:
                a, b = _ensure_array(args[0]), _ensure_array(args[1])
                if (is_num := _is_num_dt(a)) ^ _is_num_dt(b):
                    return _core_dot(a, b) if is_num else _core_dot(b.T, a.T).T

        args = tuple(_ensure_array(arg) for arg in args)
        if "out" in kwargs:
            kwargs["out"] = _ensure_array(kwargs["out"])
        res = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        return res.view(MatrixExpr) if isinstance(res, np.ndarray) else res

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        **kwargs,
    ) -> Union[Expr, MatrixExpr]:
        """
        Return the sum of the array elements over the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed. The default, axis=None, will
            sum all of the elements of the input array. If axis is negative it counts
            from the last to the first axis. If axis is a tuple of ints, a sum is
            performed on all of the axes specified in the tuple instead of a single axis
            or all the axes as before.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

        **kwargs : ignored
            Additional keyword arguments are ignored. They exist for compatibility
            with `numpy.ndarray.sum`.

        Returns
        -------
        Expr or MatrixExpr
            If the sum is performed over all axes, return an Expr, otherwise return
            a MatrixExpr.

        """
        axis: Tuple[int, ...] = normalize_axis_tuple(
            range(self.ndim) if axis is None else axis, self.ndim
        )
        if len(axis) == self.ndim:
            res = quicksum(self.flat)
            return (
                np.array([res], dtype=object).reshape([1] * self.ndim).view(MatrixExpr)
                if keepdims
                else res
            )

        keep_axes = tuple(i for i in range(self.ndim) if i not in axis)
        shape = (
            tuple(1 if i in axis else self.shape[i] for i in range(self.ndim))
            if keepdims
            else tuple(self.shape[i] for i in keep_axes)
        )
        return np.apply_along_axis(
            quicksum, -1, self.transpose(keep_axes + axis).reshape(shape + (-1,))
        ).view(MatrixExpr)

    def __le__(self, other: Union[float, int, "Expr", np.ndarray, "MatrixExpr"]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, "Expr", np.ndarray, "MatrixExpr"]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other: Union[float, int, "Expr", np.ndarray, "MatrixExpr"]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 2)

    def __add__(self, other):
        return super().__add__(other).view(MatrixExpr)
    
    def __iadd__(self, other):
        return super().__iadd__(other).view(MatrixExpr)

    def __mul__(self, other):
        return super().__mul__(other).view(MatrixExpr)

    def __truediv__(self, other):
        return super().__truediv__(other).view(MatrixExpr)
    
    def __rtruediv__(self, other):
        return super().__rtruediv__(other).view(MatrixExpr)
        
    def __pow__(self, other):
        return super().__pow__(other).view(MatrixExpr)
    
    def __sub__(self, other):
        return super().__sub__(other).view(MatrixExpr)
    
    def __radd__(self, other):
        return super().__radd__(other).view(MatrixExpr)
    
    def __rmul__(self, other):
        return super().__rmul__(other).view(MatrixExpr)
    
    def __rsub__(self, other):
        return super().__rsub__(other).view(MatrixExpr)

    def __matmul__(self, other):
        res = super().__matmul__(other)
        return res.view(MatrixExpr) if isinstance(res, np.ndarray) else res

class MatrixGenExpr(MatrixExpr):
    pass

class MatrixExprCons(np.ndarray):

    def __le__(self, other: Union[float, int, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare MatrixExprCons with '=='.")


cdef inline bool _is_num_dt(cnp.ndarray a):
    cdef char k = <char>ord(a.dtype.kind)
    return k == b"f" or k == b"i" or k == b"u" or k == b"b"


cdef inline _ensure_array(arg, bool convert_scalar = True):
    if isinstance(arg, np.ndarray):
        return arg.view(np.ndarray)
    elif isinstance(arg, (list, tuple)):
        return np.asarray(arg)
    return np.array(arg, dtype=object) if convert_scalar else arg


@np.vectorize(otypes=[object], signature="(m,n),(n,p)->(m,p)")
def _core_dot(cnp.ndarray a, cnp.ndarray x) -> np.ndarray:
    if not a.flags.c_contiguous or a.dtype != np.float64:
        a = np.ascontiguousarray(a, dtype=np.float64)

    cdef int m = a.shape[0]
    cdef int k = x.shape[1] if x.ndim > 1 else 1
    cdef cnp.ndarray[object, ndim=2] res = np.zeros((m, k), dtype=object)
    cdef cnp.ndarray row, coef
    cdef Py_ssize_t[:] nonzero
    cdef int i, j
    for i in range(m):
        row = a[i, :]
        if (nonzero := np.flatnonzero(row)).size == 0:
            continue

        coef = row[nonzero]
        for j in range(k):
            res[i, j] = quicksum(coef * x[nonzero, j])

    return res.view(MatrixExpr)
