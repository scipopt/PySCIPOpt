"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""

import numpy as np
from typing import Union

include "expr.pxi"


def _matrixexpr_richcmp(self, other, op):
    res = np.empty(shape, dtype=object)
    if _is_number(other) or isinstance(other, Expr):
        for idx in np.ndindex(self.shape):
            res[idx] = _expr_richcmp(self[idx], other, op)

    elif isinstance(other, np.ndarray):
        for idx in np.ndindex(self.shape):
            res[idx] = _expr_richcmp(self[idx], other[idx], op)

    else:
        raise TypeError(f"Unsupported type {type(other)}")

    return res.view(MatrixExprCons)


class MatrixExpr(np.ndarray):
    def sum(self, **kwargs):
        """
        Based on `numpy.ndarray.sum`, but returns a scalar if the result is a single value.
        This is useful for matrix expressions where the sum might reduce to a single value.
        """
        res = super().sum(**kwargs)
        return res if res.size > 1 else res.item()

    def __le__(self, other: Union[float, int, Expr, np.ndarray, 'MatrixExpr']) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, Expr, np.ndarray, 'MatrixExpr']) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other: Union[float, int, Expr, np.ndarray, 'MatrixExpr']) -> MatrixExprCons:
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
    
class MatrixGenExpr(MatrixExpr):
    pass

class MatrixExprCons(np.ndarray):

    def __le__(self, other: Union[float, int, Expr, np.ndarray, MatrixExpr]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 1)

    def __ge__(self, other: Union[float, int, Expr, np.ndarray, MatrixExpr]) -> MatrixExprCons:
        return _matrixexpr_richcmp(self, other, 5)

    def __eq__(self, other):
        raise TypeError
