from functools import wraps
from typing import Type

import numpy as np


def to_array(array_type: Type[np.ndarray] = np.ndarray):
    """
    Decorator to convert the input to the subclass of `numpy.ndarray` if the output is
    the instance of `numpy.ndarray`.

    Parameters
    ----------
    array_type : Type[np.ndarray], optional
        The subclass of `numpy.ndarray` to convert the output to.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            if isinstance(res, np.ndarray):
                return res.view(array_type)
            return res

        return wrapper

    return decorator
