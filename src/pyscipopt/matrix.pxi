import operator
from typing import Literal, Optional, Tuple, Union

import numpy as np

try:
    # NumPy 2.x location
    from numpy.lib.array_utils import normalize_axis_tuple
except ImportError:
    # Fallback for NumPy 1.x
    from numpy.core.numeric import normalize_axis_tuple

cimport numpy as cnp
from pyscipopt.scip cimport Expr, Solution


cnp.import_array()


class MatrixExpr(np.ndarray):

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *args,
        **kwargs,
    ):
        """
        Customizes the behavior of NumPy ufuncs for MatrixExpr.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc object that was called.

        method : {"__call__", "reduce", "reduceat", "accumulate", "outer", "at"}
            A string indicating which ufunc method was called.

        *args : tuple
            The input arguments to the ufunc.

        **kwargs : dict
            Additional keyword arguments to the ufunc.

        Returns
        -------
        Expr, MatrixExpr
            The result of the ufunc operation is wrapped back into a MatrixExpr if
            applicable.

        """
        res = NotImplemented
        # Unboxing MatrixExpr to stop __array_ufunc__ recursion
        args = tuple(_ensure_array(arg) for arg in args)
        if method == "reduce":  # Handle reduction operations, e.g., np.sum(a)
            if ufunc is np.add:
                res = _core_sum(args[0], **kwargs)

        if method == "__call__":  # Standard ufunc call, e.g., np.add(a, b)
            if ufunc in {np.matmul, np.dot}:
                res = _core_dot(args[0], args[1])
            elif ufunc is np.less_equal:
                return _vec_le(args[0], args[1]).view(MatrixExprCons)
            elif ufunc is np.greater_equal:
                return _vec_ge(args[0], args[1]).view(MatrixExprCons)
            elif ufunc is np.equal:
                return _vec_eq(args[0], args[1]).view(MatrixExprCons)
            elif ufunc in {np.less, np.greater, np.not_equal}:
                raise NotImplementedError("can only support '<=', '>=', or '=='")

        if res is NotImplemented:
            if "out" in kwargs: # Unboxing MatrixExpr to stop __array_ufunc__ recursion
                kwargs["out"] = tuple(_ensure_array(arg, False) for arg in kwargs["out"])
            res = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        return res.view(MatrixExpr) if isinstance(res, np.ndarray) else res

    def _evaluate(self, Solution sol) -> np.ndarray:
        return _vec_evaluate(self, sol).view(np.ndarray)


class MatrixExprCons(np.ndarray):

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == "__call__":
            args = tuple(_ensure_array(arg) for arg in args)
            if ufunc is np.less_equal:
                return _vec_le(args[0], args[1]).view(MatrixExprCons)
            elif ufunc is np.greater_equal:
                return _vec_ge(args[0], args[1]).view(MatrixExprCons)
        raise NotImplementedError("can only support '<=' or '>='")

    def __eq__(self, _):
        # TODO: Once numpy version >= 2.x, remove `__eq__`, as it will be handled by
        # `__array_ufunc__`.
        raise NotImplementedError("can only support '<=' or '>='")


_vec_le = np.frompyfunc(operator.le, 2, 1)
_vec_ge = np.frompyfunc(operator.ge, 2, 1)
_vec_eq = np.frompyfunc(operator.eq, 2, 1)
_vec_evaluate = np.frompyfunc(lambda expr, sol: expr._evaluate(sol), 2, 1)


cdef inline _ensure_array(arg, bint convert_scalar = True):
    if isinstance(arg, np.ndarray):
        return arg.view(np.ndarray)
    elif isinstance(arg, (list, tuple)):
        return np.asarray(arg)
    return np.array(arg, dtype=object) if convert_scalar else arg


def _core_dot(cnp.ndarray a, cnp.ndarray b) -> Union[Expr, np.ndarray]:
    """
    Perform matrix multiplication between a N-Demension constant array and a N-Demension
    `np.ndarray` of type `object` and containing `Expr` objects.

    Parameters
    ----------
    a : np.ndarray
        A constant n-d `np.ndarray` of type `np.float64`.

    b : np.ndarray
        A n-d `np.ndarray` of type `object` and containing `Expr` objects.

    Returns
    -------
    Expr or np.ndarray
        If both `a` and `b` are 1-D arrays, return an `Expr`, otherwise return a
        `np.ndarray` of type `object` and containing `Expr` objects.
    """
    cdef bint a_is_1d = a.ndim == 1
    cdef bint b_is_1d = b.ndim == 1
    cdef cnp.ndarray a_nd = a[..., np.newaxis, :] if a_is_1d else a
    cdef cnp.ndarray b_nd = b[..., :, np.newaxis] if b_is_1d else b
    cdef bint a_is_num = a_nd.dtype.kind in "fiub"

    if a_is_num ^ (b_nd.dtype.kind in "fiub"):
        res = _core_dot_nd(a_nd, b_nd) if a_is_num else _core_dot_nd(b_nd.T, a_nd.T).T
        if a_is_1d and b_is_1d:
            return res.item()
        if a_is_1d:
            return res.reshape(np.delete(res.shape, -2))
        if b_is_1d:
            return res.reshape(np.delete(res.shape, -1))
        return res
    return NotImplemented


def _core_dot_2d(cnp.ndarray a, cnp.ndarray x) -> np.ndarray:
    """
    Perform matrix multiplication between a 2-Demension constant array and a 2-Demension
    `np.ndarray` of type `object` and containing `Expr` objects.

    Parameters
    ----------
    a : np.ndarray
        A 2-D `np.ndarray` of type `np.float64`.

    x : np.ndarray
        A 2-D `np.ndarray` of type `object` and containing `Expr` objects.

    Returns
    -------
    np.ndarray
        A 2-D `np.ndarray` of type `object` and containing `Expr` objects.
    """
    if not a.flags.c_contiguous or a.dtype != np.float64:
        a = np.ascontiguousarray(a, dtype=np.float64)

    cdef const double[:, :] a_view = a
    cdef int m = a.shape[0], k = x.shape[1]
    cdef cnp.ndarray[object, ndim=2] res = np.zeros((m, k), dtype=object)
    cdef Py_ssize_t[:] nonzero
    cdef int i, j, idx

    for i in range(m):
        if (nonzero := np.flatnonzero(a_view[i, :])).size == 0:
            continue

        for j in range(k):
            res[i, j] = quicksum(a_view[i, idx] * x[idx, j] for idx in nonzero)

    return res


_core_dot_nd = np.vectorize(
    _core_dot_2d,
    otypes=[object],
    signature="(m,n),(n,p)->(m,p)",
)

def _core_sum(
    cnp.ndarray a,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    **kwargs,
) -> Union[Expr, np.ndarray]:
    """
    Return the sum of the array elements over the given axis.

    Parameters
    ----------
    a : np.ndarray
        A `np.ndarray` of type `object` and containing `Expr` objects.

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
    Expr or np.ndarray
        If the sum is performed over all axes, return an Expr, otherwise return
        a np.ndarray.

    """
    axis: Tuple[int, ...] = normalize_axis_tuple(
        range(a.ndim) if axis is None else axis, a.ndim
    )
    if len(axis) == a.ndim:
        res = quicksum(a.flat)
        return (
            np.array([res], dtype=object).reshape([1] * a.ndim)
            if keepdims
            else res
        )

    keep_axes = tuple(i for i in range(a.ndim) if i not in axis)
    shape = (
        tuple(1 if i in axis else a.shape[i] for i in range(a.ndim))
        if keepdims
        else tuple(a.shape[i] for i in keep_axes)
    )
    return np.apply_along_axis(
        quicksum, -1, a.transpose(keep_axes + axis).reshape(shape + (-1,))
    )
