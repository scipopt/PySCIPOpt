"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""
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


def _matrixexpr_richcmp(self, other, op: Callable):
    if isinstance(other, Number) or isinstance(other, (Variable, Expr)):
        res = np.empty(self.shape, dtype=object)
        res.flat[:] = [op(i, other) for i in self.flat]

    elif isinstance(other, np.ndarray):
        out = np.broadcast(self, other)
        res = np.empty(out.shape, dtype=object)
        res.flat[:] = [op(i, j) for i, j in out]

    else:
        raise TypeError(f"Unsupported type {type(other)}")

    return res.view(MatrixExprCons)


class MatrixExprCons(np.ndarray):

    def __le__(self, other: Union[Number, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, operator.le)

    def __ge__(self, other: Union[Number, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, operator.ge)

    def __eq__(self, _):
        raise NotImplementedError("Cannot compare MatrixExprCons with '=='.")


class MatrixExpr(np.ndarray):

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

    def __le__(self, other: Union[Number, Expr, np.ndarray, MatrixExpr]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, operator.le)

    def __ge__(self, other: Union[Number, Expr, np.ndarray, MatrixExpr]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, operator.ge)

    def __eq__(self, other: Union[Number, Expr, np.ndarray, MatrixExpr]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, operator.eq)

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
        return super().__matmul__(other).view(MatrixExpr)
