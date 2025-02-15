
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

class MatrixVariable(np.ndarray):
    def sum(self, **kwargs):
        return super().sum(**kwargs).item()
    
    def __le__(self, other: Union[float, int, Variable, np.ndarray, 'MatrixVariable']) -> np.ndarray:
       
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] <= other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] <= other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix

    def __ge__(self, other: Union[float, int, Variable, np.ndarray, 'MatrixVariable']) -> np.ndarray:
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] >= other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] >= other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix

    def __eq__(self, other: Union[float, int, Variable, np.ndarray, 'MatrixVariable']) -> np.ndarray:
        expr_cons_matrix = np.empty(self.shape, dtype=object)
        if _is_number(other) or isinstance(other, Variable):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] == other
        
        elif isinstance(other, np.ndarray):
            for idx in np.ndindex(self.shape):
                expr_cons_matrix[idx] = self[idx] == other[idx]    
        else:
            raise TypeError(f"Unsupported type {type(other)}")

        return expr_cons_matrix
    
