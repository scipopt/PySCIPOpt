"""
# TODO Cythonize things. Improve performance.
# TODO Add tests
"""

import numpy as np
from typing import Union

def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, Expr)
        return False

class MatrixExpr(np.ndarray):
    def sum(self, **kwargs):
        return super().sum(**kwargs).item()
    
    def __le__(self, other: Union[float, int, Variable, np.ndarray, 'MatrixExpr']) -> np.ndarray:
        
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] <= other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] <= other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix.view(MatrixExprCons)

    def __ge__(self, other: Union[float, int, Variable, np.ndarray, 'MatrixExpr']) -> np.ndarray:
        
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] >= other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] >= other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix.view(MatrixExprCons)

    def __eq__(self, other: Union[float, int, Variable, np.ndarray, 'MatrixExpr']) -> np.ndarray:
        
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] == other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] == other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix.view(MatrixExprCons)
    
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

    def __le__(self, other: Union[float, int, Variable, MatrixExpr]) -> np.ndarray:
       
        if not _is_number(other) or not isinstance(other, MatrixExpr):
                raise TypeError('Ranged MatrixExprCons is not well defined!')

        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] <= other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] <= other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix.view(MatrixExprCons)

    def __ge__(self, other: Union[float, int, Variable, MatrixExpr]) -> np.ndarray:
        
        if not _is_number(other) or not isinstance(other, MatrixExpr):
                raise TypeError('Ranged MatrixExprCons is not well defined!')
                
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] >= other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] >= other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix.view(MatrixExprCons)

    def __eq__(self, other):
        raise TypeError
