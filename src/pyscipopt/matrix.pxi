import operator
from numbers import Number
from typing import Callable, Optional, Tuple, Union
import numpy as np
try:
    # NumPy 2.x location
    from numpy.lib.array_utils import normalize_axis_tuple
except ImportError:
    # Fallback for NumPy 1.x
    from numpy.core.numeric import normalize_axis_tuple

from pyscipopt.scip cimport Expr, quicksum, Variable


class MatrixExprCons(np.ndarray):
    def __le__(self, other: Union[Number, np.ndarray]) -> MatrixExprCons:
        return _cmp(self, other, operator.le)

    def __ge__(self, other: Union[Number, np.ndarray]) -> MatrixExprCons:
        return _cmp(self, other, operator.ge)

    def __eq__(self, _):
        raise NotImplementedError("Cannot compare MatrixExprCons with '=='.")


class MatrixBase(np.ndarray):

    def __array_wrap__(self, array, context=None, return_scalar=False):
        res = super().__array_wrap__(array, context, return_scalar)
        if return_scalar and isinstance(res, np.ndarray) and res.ndim == 0:
            return res.item()
        elif isinstance(res, np.ndarray):
            if context is not None and context[0] in {
                np.less_equal,
                np.greater_equal,
                np.equal,
            }:
                return res.view(MatrixExprCons)
            return res.view(MatrixExpr)
        return res

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

    def __le__(
        self,
        other: Union[Number, Variable, Expr, np.ndarray, MatrixBase],
    ) -> MatrixExprCons:
        return _cmp(self, other, operator.le)

    def __ge__(
        self,
        other: Union[Number, Variable, Expr, np.ndarray, MatrixBase],
    ) -> MatrixExprCons:
        return _cmp(self, other, operator.ge)

    def __eq__(
        self,
        other: Union[Number, Variable, Expr, np.ndarray, MatrixBase],
    ) -> MatrixExprCons:
        return _cmp(self, other, operator.eq)


class MatrixExpr(MatrixBase):
    ...


def _cmp(
    x: Union[MatrixBase, MatrixExprCons],
    y: Union[Number, Variable, Expr, np.ndarray, MatrixBase],
    op: Callable,
) -> MatrixExprCons:
    if isinstance(y, Number) or isinstance(y, (Variable, Expr)):
        res = np.empty(x.shape, dtype=object)
        res.flat[:] = [op(i, y) for i in x.flat]
    elif isinstance(y, np.ndarray):
        out = np.broadcast(x, y)
        res = np.empty(out.shape, dtype=object)
        res.flat[:] = [op(i, j) for i, j in out]
    else:
        raise TypeError(f"Unsupported type {type(y)}")

    return res.view(MatrixExprCons)
