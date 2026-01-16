import operator
from typing import Optional, Tuple, Union
import numpy as np
try:
    # NumPy 2.x location
    from numpy.lib.array_utils import normalize_axis_tuple
except ImportError:
    # Fallback for NumPy 1.x
    from numpy.core.numeric import normalize_axis_tuple

from pyscipopt.scip cimport Expr, quicksum


class MatrixExprLike(np.ndarray):

    __array_priority__ = 101

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        args = _ensure_array(args)
        if ufunc is np.less_equal:
            return _vec_le(*args).view(MatrixExprCons)
        elif ufunc is np.greater_equal:
            return _vec_ge(*args).view(MatrixExprCons)
        elif ufunc is np.equal:
            return _vec_eq(*args).view(MatrixExprCons)
        elif ufunc in {np.less, np.greater, np.not_equal}:
            raise NotImplementedError("can only support with '<=', '>=', or '=='")

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


class MatrixExpr(MatrixExprLike):
    ...


class MatrixExprCons(np.ndarray):

    __array_priority__ = 101

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__":
            return NotImplemented

        args = _ensure_array(args)
        if ufunc is np.less_equal:
            return _vec_le(*args).view(MatrixExprCons)
        elif ufunc is np.greater_equal:
            return _vec_ge(*args).view(MatrixExprCons)
        elif ufunc in {np.equal, np.less, np.greater, np.not_equal}:
            raise NotImplementedError("can only support with '<=' or '=='")
        return NotImplemented


_vec_le = np.frompyfunc(operator.le, 2, 1)
_vec_ge = np.frompyfunc(operator.ge, 2, 1)
_vec_eq = np.frompyfunc(operator.eq, 2, 1)


cdef inline list _ensure_array(tuple args):
    cdef list res = []
    cdef object arg
    for arg in args:
        if isinstance(arg, np.ndarray):
            res.append(arg.view(np.ndarray))
        else:
            res.append(np.array(arg, dtype=object))
    return res
