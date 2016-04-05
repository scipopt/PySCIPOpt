CONST = ()

def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, LinExpr)
        return False


def _expr_richcmp(self, other, op):
    if op == 1: # <=
        if isinstance(other, LinExpr):
            return (self - other) <= 0.0
        elif _is_number(other):
            return LinCons(self, ub=float(other))
        else:
            raise NotImplementedError
    elif op == 5: # >=
        if isinstance(other, LinExpr):
            return (self - other) >= 0.0
        elif _is_number(other):
            return LinCons(self, lb=float(other))
        else:
            raise NotImplementedError
    elif op == 2: # ==
        if isinstance(other, LinExpr):
            return (self - other) == 0.0
        elif _is_number(other):
            return LinCons(self, lb=float(other), ub=float(other))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


cdef class LinExpr:
    '''Linear expressions of variables with operator overloading.'''
    cdef public terms

    def __init__(self, terms=None):
        '''terms is a dict of variables to coefficients.

        The empty tuple is used as key for the constant term.'''
        self.terms = {} if terms is None else terms

        if len(self.terms) == 0:
            self.terms[()] = 0.0

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self.terms.get(key, 0.0)

    def __add__(self, other):
        left = self
        right = other

        if _is_number(self):
            assert isinstance(other, LinExpr)
            left,right = right,left
        terms = left.terms.copy()

        if isinstance(right, LinExpr):
            # merge the terms by component-wise addition
            for v,c in right.terms.items():
                terms[v] = terms.get(v, 0.0) + c
        elif _is_number(right):
            c = float(right)
            terms[CONST] = terms.get(CONST, 0.0) + c
        else:
            raise NotImplementedError
        return LinExpr(terms)

    def __iadd__(self, other):
        if isinstance(other, LinExpr):
            for v,c in other.terms.items():
                self.terms[v] = self.terms.get(v, 0.0) + c
        elif _is_number(other):
            c = float(other)
            self.terms[CONST] = self.terms.get(CONST, 0.0) + c
        else:
            raise NotImplementedError
        return self

    def __mul__(self, other):
        if _is_number(other):
            f = float(other)
            return LinExpr({v:f*c for v,c in self.terms.items()})
        elif _is_number(self):
            f = float(self)
            return LinExpr({v:f*c for v,c in other.terms.items()})
        elif isinstance(other, LinExpr):
            terms = {}
            for v1, c1 in self.terms.items():
                for v2, c2 in other.terms.items():
                    v = tuple(sorted(v1 + v2))
                    terms[v] = terms.get(v, 0.0) + c1 * c2
            return LinExpr(terms)
        else:
            raise NotImplementedError

    def __pow__(self, other, modulo):
        if float(other).is_integer() and other >= 0:
            exp = int(other)
        else:
            raise NotImplementedError

        res = 1
        for _ in range(exp):
            res *= self
        return res

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

    def __richcmp__(self, other, op):
        '''turn it into a constraint'''
        return _expr_richcmp(self, other, op)

    def __repr__(self):
        return 'LinExpr(%s)' % repr(self.terms)

    def degree(self):
        '''computes highest degree of terms'''
        return max(len(v) for v in self.terms)


cdef class LinCons:
    '''Constraints with a linear expressions and lower/upper bounds.'''
    cdef public expr
    cdef public lb
    cdef public ub

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
        assert self.expr[CONST] == 0.0

    def __richcmp__(self, other, op):
        '''turn it into a constraint'''
        if op == 1: # <=
           if not self.ub is None:
               raise TypeError('LinCons already has upper bound')
           assert self.ub is None
           assert not self.lb is None

           if not _is_number(other):
               raise TypeError('Ranged LinCons is not well defined!')

           return LinCons(self.expr, lb=self.lb, ub=float(other))
        elif op == 5: # >=
           if not self.lb is None:
               raise TypeError('LinCons already has lower bound')
           assert self.lb is None
           assert not self.ub is None

           if not _is_number(other):
               raise TypeError('Ranged LinCons is not well defined!')

           return LinCons(self.expr, lb=float(other), ub=self.ub)
        else:
            raise TypeError

    def __repr__(self):
        return 'LinCons(%s, %s, %s)' % (self.expr, self.lb, self.ub)

    def __nonzero__(self):
        '''Make sure that equality of expressions is not asserted with =='''
        raise TypeError("Can't evaluate constraints as booleans.")
