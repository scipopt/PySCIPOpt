"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""

import numpy as np
from typing import Optional, Union


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

    def sum(
        self,
        axis: Optional[tuple[int]] = None,
        keepdims: bool = False,
        **kwargs,
    ) -> Union[Expr, MatrixExpr]:
        """
        Based on `numpy.ndarray.sum`, but returns a scalar if `axis=None`.
        This is useful for matrix expressions to compare with a matrix or a scalar.
        """

        if axis is None:
            axis = tuple(range(self.ndim))

        elif isinstance(axis, int):
            if axis < -self.ndim or axis >= self.ndim:
                raise np.exceptions.AxisError(
                    f"axis {axis} is out of bounds for array of dimension {self.ndim}"
                )
            axis = (axis,)

        elif isinstance(axis, tuple) and all(isinstance(i, int) for i in axis):
            for i in axis:
                if i < -self.ndim or i >= self.ndim:
                    raise np.exceptions.AxisError(
                        f"axis {i} is out of bounds for array of dimension {self.ndim}"
                    )

        else:
            raise TypeError("'axis' must be an int or a tuple of ints")

        if len(axis := tuple(i + self.ndim if i < 0 else i for i in axis)) == self.ndim:
            res = quicksum(self.flat)
            if keepdims:
                return (
                    np.array([res], dtype=object)
                    .reshape([1] * self.ndim)
                    .view(MatrixExpr)
                )
            return res

        keep_axes = tuple(i for i in range(self.ndim) if i not in axis)
        shape = (
            tuple(1 if i in axis else self.shape[i] for i in range(self.ndim))
            if keepdims
            else tuple(self.shape[i] for i in keep_axes)
        )
        return (
            np.fromiter(
                map(
                    quicksum,
                    self.transpose(keep_axes + axis).reshape(
                        -1, np.prod([self.shape[i] for i in axis])
                    ),
                ),
                dtype=object,
            )
            .reshape(shape)
            .view(MatrixExpr)
        )

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
        return super().__matmul__(other).view(MatrixExpr)

class MatrixGenExpr(MatrixExpr):
    pass

class MatrixExprCons(np.ndarray):

    def __le__(self, other: Union[float, int, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, np.ndarray]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other):
        raise NotImplementedError("Cannot compare MatrixExprCons with '=='.")
