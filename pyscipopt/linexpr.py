CONST = ()

def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, LinExpr)
        return False

class LinExpr(object):
    '''Linear expressions of variables with operator overloading.'''

    def __init__(self, terms=None):
        '''terms is a dict of variables to coefficients.

        The empty tuple is used as key for the constant term.'''
        self.terms = {} if terms is None else terms

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self.terms.get(key, 0.0)

    def __add__(self, other):
        terms = self.terms.copy()
        if isinstance(other, LinExpr):
            # merge the terms by component-wise addition
            for v,c in other.terms.items():
                terms[v] = terms.get(v, 0.0) + c
        elif _is_number(other):
            c = float(other)
            terms[CONST] = terms.get(CONST, 0.0) + c
        else:
            raise NotImplementedError
        return LinExpr(terms)

    def __mul__(self, other):
        if _is_number(other):
            f = float(other)
            return LinExpr({v:f*c for v,c in self.terms.items()})
        else:
            raise NotImplementedError

    def __neg__(self):
        return LinExpr({v:-c for v,c in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return -1.0 * self + other
