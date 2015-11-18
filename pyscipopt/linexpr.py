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

    def __le__(self, other):
        '''turn it into a constraint'''
        if isinstance(other, LinExpr):
            return (self - other) <= 0.0
        elif _is_number(other):
            return LinCons(self, ub=float(other))

    def __ge__(self, other):
        '''turn it into a constraint'''
        if isinstance(other, LinExpr):
            return (self - other) >= 0.0
        elif _is_number(other):
            return LinCons(self, lb=float(other))

    def __repr__(self):
        return 'LinExpr(%s)' % repr(self.terms)


class LinCons(object):
    '''Constraints with a linear expressions and lower/upper bounds.'''

    def __init__(self, expr, lb=None, ub=None):
        self.expr = expr
        self.lb = lb
        self.ub = ub
        assert not (lb is None and ub is None)
        self._normalize()

    def _normalize(self):
        '''move constant terms in expression to bounds'''
        c = self.expr[CONST]
        if not self.lb is None:
            self.lb -= c
        if not self.ub is None:
            self.ub -= c
        self.expr -= c

    def __repr__(self):
        return 'LinCons(%s, %s, %s)' % (self.expr, self.lb, self.ub)
