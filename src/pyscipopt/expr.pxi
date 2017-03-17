
def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, Expr)
        return False


def _expr_richcmp(self, other, op):
    if op == 1: # <=
        if isinstance(other, Expr):
            return (self - other) <= 0.0
        elif _is_number(other):
            return ExprCons(self, rhs=float(other))
        else:
            raise NotImplementedError
    elif op == 5: # >=
        if isinstance(other, Expr):
            return (self - other) >= 0.0
        elif _is_number(other):
            return ExprCons(self, lhs=float(other))
        else:
            raise NotImplementedError
    elif op == 2: # ==
        if isinstance(other, Expr):
            return (self - other) == 0.0
        elif _is_number(other):
            return ExprCons(self, lhs=float(other), rhs=float(other))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


class Term:
    '''This is a monomial term'''

    __slots__ = ('vartuple', 'ptrtuple', 'hashval')

    def __init__(self, *vartuple):
        self.vartuple = tuple(sorted(vartuple, key=lambda v: v.ptr()))
        self.ptrtuple = tuple(v.ptr() for v in self.vartuple)
        self.hashval = sum(self.ptrtuple)

    def __getitem__(self, idx):
        return self.vartuple[idx]

    def __hash__(self):
        return self.hashval

    def __eq__(self, other):
        return self.ptrtuple == other.ptrtuple

    def __len__(self):
        return len(self.vartuple)

    def __add__(self, other):
        both = self.vartuple + other.vartuple
        return Term(*both)

    def __repr__(self):
        return 'Term(%s)' % ', '.join([str(v) for v in self.vartuple])

CONST = Term()

cdef class Expr:
    '''Polynomial expressions of variables with operator overloading.'''
    cdef public terms

    def __init__(self, terms=None):
        '''terms is a dict of variables to coefficients.

        CONST is used as key for the constant term.'''
        self.terms = {} if terms is None else terms

        if len(self.terms) == 0:
            self.terms[CONST] = 0.0

    def __getitem__(self, key):
        if not isinstance(key, Term):
            key = Term(key)
        return self.terms.get(key, 0.0)

    def __add__(self, other):
        left = self
        right = other

        if _is_number(self):
            assert isinstance(other, Expr)
            left,right = right,left
        terms = left.terms.copy()

        if isinstance(right, Expr):
            # merge the terms by component-wise addition
            for v,c in right.terms.items():
                terms[v] = terms.get(v, 0.0) + c
        elif _is_number(right):
            c = float(right)
            terms[CONST] = terms.get(CONST, 0.0) + c
        else:
            raise NotImplementedError
        return Expr(terms)

    def __iadd__(self, other):
        if isinstance(other, Expr):
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
            return Expr({v:f*c for v,c in self.terms.items()})
        elif _is_number(self):
            f = float(self)
            return Expr({v:f*c for v,c in other.terms.items()})
        elif isinstance(other, Expr):
            terms = {}
            for v1, c1 in self.terms.items():
                for v2, c2 in other.terms.items():
                    v = v1 + v2
                    terms[v] = terms.get(v, 0.0) + c1 * c2
            return Expr(terms)
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
        return Expr({v:-c for v,c in self.terms.items()})

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

    def normalize(self):
        '''remove terms with coefficient of 0'''
        self.terms =  {t:c for (t,c) in self.terms.items() if c != 0.0}

    def __repr__(self):
        return 'Expr(%s)' % repr(self.terms)

    def degree(self):
        '''computes highest degree of terms'''
        return max(len(v) for v in self.terms)


cdef class ExprCons:
    '''Constraints with a polynomial expressions and lower/upper bounds.'''
    cdef public expr
    cdef public lhs
    cdef public rhs

    def __init__(self, expr, lhs=None, rhs=None):
        self.expr = expr
        self.lhs = lhs
        self.rhs = rhs
        assert not (lhs is None and rhs is None)
        self.normalize()

    def normalize(self):
        '''move constant terms in expression to bounds'''
        c = self.expr[CONST]
        self.expr -= c
        if not self.lhs is None:
            self.lhs -= c
        if not self.rhs is None:
            self.rhs -= c

        assert self.expr[CONST] == 0.0
        self.expr.normalize()

    def __richcmp__(self, other, op):
        '''turn it into a constraint'''
        if op == 1: # <=
           if not self.rhs is None:
               raise TypeError('ExprCons already has upper bound')
           assert self.rhs is None
           assert not self.lhs is None

           if not _is_number(other):
               raise TypeError('Ranged ExprCons is not well defined!')

           return ExprCons(self.expr, lhs=self.lhs, rhs=float(other))
        elif op == 5: # >=
           if not self.lhs is None:
               raise TypeError('ExprCons already has lower bound')
           assert self.lhs is None
           assert not self.rhs is None

           if not _is_number(other):
               raise TypeError('Ranged ExprCons is not well defined!')

           return ExprCons(self.expr, lhs=float(other), rhs=self.rhs)
        else:
            raise TypeError

    def __repr__(self):
        return 'ExprCons(%s, %s, %s)' % (self.expr, self.lhs, self.rhs)

    def __nonzero__(self):
        '''Make sure that equality of expressions is not asserted with =='''

        msg = """Can't evaluate constraints as booleans.

If you want to add a ranged constraint of the form
   lhs <= expression <= rhs
you have to use parenthesis to break the Python syntax for chained comparisons:
   lhs <= (expression <= rhs)
"""
        raise TypeError(msg)

def quicksum(termlist):
    '''add linear expressions and constants much faster than Python's sum
    by avoiding intermediate data structures and adding terms inplace
    '''
    result = Expr()
    for term in termlist:
        result += term
    return result
