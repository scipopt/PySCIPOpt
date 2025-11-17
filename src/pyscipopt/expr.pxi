##@file expr.pxi
import math
from typing import Optional, Type, Union

include "matrix.pxi"


def _is_number(e):
    try:
        f = float(e)
        return True
    except ValueError: # for malformed strings
        return False
    except TypeError: # for other types (Variable, Expr)
        return False


cdef class Term:
    """A monomial term consisting of one or more variables."""

    __slots__ = ("vars", "ptrs")

    def __init__(self, *vars):
        self.vars = tuple(sorted(vars, key=lambda v: v.ptr()))
        self.ptrs = tuple(v.ptr() for v in self.vars)

    def __getitem__(self, idx):
        return self.vars[idx]

    def __hash__(self):
        return self.ptrs.__hash__()

    def __eq__(self, other):
        return self.ptrs == other.ptrs

    def __len__(self):
        return len(self.vars)

    def __mul__(self, other):
        if not isinstance(other, Term):
            raise TypeError(
                f"unsupported operand type(s) for *: 'Term' and '{type(other)}'"
            )
        return Term(*self.vars, *other.vars)

    def __repr__(self):
        return f"Term({', '.join(map(str, self.vars))})"

    cdef float _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        if self.vars:
            return math.prod(SCIPgetSolVal(scip, sol, ptr) for ptr in self.ptrs)
        return 1.0  # constant term


CONST = Term()


cdef float _evaluate(dict children, SCIP* scip, SCIP_SOL* sol):
    return sum([i._evaluate(scip, sol) * j for i, j in children.items() if j != 0])


cdef class Expr:
    """Base class for mathematical expressions."""

    cdef public dict children

    def __init__(self, children: Optional[dict] = None):
        self.children = children or {}

    def __hash__(self):
        return frozenset(self.children.items()).__hash__()

    def __getitem__(self, key):
        return self.children.get(key, 0.0)

    def __iter__(self):
        return iter(self.children)

    def __next__(self):
        try:
            return next(self.children)
        except:
            raise StopIteration

    def __abs__(self):
        return _to_unary_expr(self, AbsExpr)

    def __add__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, Expr):
            return SumExpr({self: 1.0, other: 1.0})
        elif isinstance(other, MatrixExpr):
            return other.__add__(self)
        raise TypeError(
            f"unsupported operand type(s) for +: 'Expr' and '{type(other)}'"
        )

    def __mul__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, Expr):
            return ProdExpr(self, other)
        elif isinstance(other, MatrixExpr):
            return other.__mul__(self)
        raise TypeError(
            f"unsupported operand type(s) for *: 'Expr' and '{type(other)}'"
        )

    def __truediv__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, ConstExpr) and other[CONST] == 0:
            raise ZeroDivisionError("division by zero")
        if hash(self) == hash(other):
            return ConstExpr(1.0)
        return self.__mul__(other.__pow__(-1.0))

    def __rtruediv__(self, other):
        return Expr.to_const_or_var(other).__truediv__(self)

    def __pow__(self, other):
        other = Expr.to_const_or_var(other)
        if not isinstance(other, ConstExpr):
            raise TypeError("exponent must be a number")

        if other[CONST] == 0:
            return ConstExpr(1.0)
        return PowerExpr(self, other[CONST])

    def __rpow__(self, other):
        other = Expr.to_const_or_var(other)
        if not isinstance(other, ConstExpr):
            raise TypeError("base must be a number")
        if other[CONST] <= 0.0:
            raise ValueError("base must be positive")
        return exp(self * log(other[CONST]))

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __lt__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, Expr):
            if isinstance(other, ConstExpr):
                return ExprCons(self, rhs=other[CONST])
            return (self - other) <= 0
        elif isinstance(other, MatrixExpr):
            return other.__gt__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __gt__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, Expr):
            if isinstance(other, ConstExpr):
                return ExprCons(self, lhs=other[CONST])
            return (self - other) >= 0
        elif isinstance(other, MatrixExpr):
            return self.__lt__(other)
        raise TypeError(f"Unsupported type {type(other)}")

    def __ge__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, Expr):
            if isinstance(other, ConstExpr):
                return ExprCons(self, lhs=other[CONST], rhs=other[CONST])
            return (self - other) == 0
        elif isinstance(other, MatrixExpr):
            return other.__ge__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __repr__(self):
        return f"Expr({self.children})"

    @staticmethod
    def to_const_or_var(x):
        """Convert a number or variable to an expression."""

        if _is_number(x):
            return PolynomialExpr.to_subclass({CONST: x})
        elif isinstance(x, Variable):
            return PolynomialExpr.to_subclass({Term(x): 1.0})
        return x

    def to_dict(self, other: Optional[dict] = None) -> dict:
        """Merge two dictionaries by summing values of common keys"""
        other = other or {}
        if not isinstance(other, dict):
            raise TypeError("other must be a dict")

        res = self.children.copy()
        for child, coef in other.items():
            res[child] = res.get(child, 0.0) + coef

        return res

    def _normalize(self) -> Expr:
        return self


cdef class SumExpr(Expr):
    """Expression like `expression1 + expression2 + constant`."""

    def __add__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, SumExpr):
            return SumExpr(self.to_dict(other.children))
        return super().__add__(other)

    def __mul__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, ConstExpr):
            if other[CONST] == 0:
                return ConstExpr(0.0)
            return SumExpr({i: self[i] * other[CONST] for i in self if self[i] != 0})
        return super().__mul__(other)

    def degree(self):
        return float("inf")

    cdef float _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        return _evaluate(self.children, scip, sol)


class PolynomialExpr(SumExpr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict] = None):
        if children and not all(isinstance(t, Term) for t in children):
            raise TypeError("All keys must be Term instances")

        super().__init__(children)

    def __add__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            return PolynomialExpr.to_subclass(self.to_dict(other.children))
        return super().__add__(other)

    def __mul__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            children = {}
            for i in self:
                for j in other:
                    child = i * j
                    children[child] = children.get(child, 0.0) + self[i] * other[j]
            return PolynomialExpr.to_subclass(children)
        return super().__mul__(other)

    def __truediv__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, ConstExpr):
            return self.__mul__(1.0 / other[CONST])
        return super().__truediv__(other)

    def __pow__(self, other):
        other = Expr.to_const_or_var(other)
        if (
            isinstance(other, Expr)
            and isinstance(other, ConstExpr)
            and other[CONST].is_integer()
            and other[CONST] > 0
        ):
            res = 1
            for _ in range(int(other[CONST])):
                res *= self
            return res
        return super().__pow__(other)

    def degree(self):
        """Computes the highest degree of children"""

        return max(map(len, self.children)) if self.children else 0

    @classmethod
    def to_subclass(cls, children: dict):
        if len(children) == 0:
            return ConstExpr(0.0)
        elif len(children) == 1:
            if CONST in children:
                return ConstExpr(children[CONST])
            return MonomialExpr(children)
        return cls(children)

    def _normalize(self) -> Expr:
        return PolynomialExpr.to_subclass(
            {k: v for k, v in self.children.items() if v != 0.0}
        )


class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, constant: float = 0):
        super().__init__({CONST: constant})

    def __abs__(self):
        return ConstExpr(abs(self[CONST]))

    def __pow__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, ConstExpr):
            return ConstExpr(self[CONST] ** other[CONST])
        return super().__pow__(other)


class MonomialExpr(PolynomialExpr):
    """Expression like `x**3`."""

    def __init__(self, children: Optional[dict] = None):
        if children and len(children) != 1:
            raise ValueError("MonomialExpr must have exactly one child")

        super().__init__(children)

    @staticmethod
    def from_var(var: Variable, coef: float = 1.0):
        return MonomialExpr({Term(var): coef})


class FuncExpr(Expr):
    def degree(self):
        return float("inf")


cdef class ProdExpr(FuncExpr):
    """Expression like `coefficient * expression`."""

    def __init__(self, *children, coef: float = 1.0):
        super().__init__({i: 1.0 for i in children})
        self.coef = coef

    def __hash__(self):
        return (frozenset(self), self.coef).__hash__()

    def __add__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, ProdExpr) and hash(frozenset(self)) == hash(
            frozenset(other)
        ):
            return ProdExpr(*self, coef=self.coef + other.coef)
        return super().__add__(other)

    def __mul__(self, other):
        other = Expr.to_const_or_var(other)
        if isinstance(other, ConstExpr):
            if other[CONST] == 0:
                return ConstExpr(0.0)
            return ProdExpr(*self, coef=self.coef * other[CONST])
        return super().__mul__(other)

    def __repr__(self):
        return f"ProdExpr({{{tuple(self)}: {self.coef}}})"

    def _normalize(self) -> Expr:
        if self.coef == 0:
            return ConstExpr(0.0)
        return self

    cdef float _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        return self.coef * _evaluate(self.children, scip, sol)


cdef class PowerExpr(FuncExpr):
    """Expression like `pow(expression, exponent)`."""

    def __init__(self, base, expo: float = 1.0):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self):
        return (frozenset(self), self.expo).__hash__()

    def __repr__(self):
        return f"PowerExpr({tuple(self)}, {self.expo})"

    def _normalize(self) -> Expr:
        if self.expo == 0:
            return ConstExpr(1.0)
        elif self.expo == 1:
            return tuple(self)[0]
        return self

    cdef float _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        return pow(_evaluate(self.children, scip, sol), self.expo)


cdef class UnaryExpr(FuncExpr):
    """Expression like `f(expression)`."""

    def __init__(self, expr: Expr):
        super().__init__({expr: 1.0})

    def __hash__(self):
        return frozenset(self).__hash__()

    def __repr__(self):
        return f"{type(self).__name__}({tuple(self)[0]})"

    cdef float _evaluate(self, SCIP* scip, SCIP_SOL* sol):
        return self.op(_evaluate(self.children, scip, sol))


class AbsExpr(UnaryExpr):
    """Expression like `abs(expression)`."""
    op = abs


class ExpExpr(UnaryExpr):
    """Expression like `exp(expression)`."""
    op = math.exp


class LogExpr(UnaryExpr):
    """Expression like `log(expression)`."""
    op = math.log


class SqrtExpr(UnaryExpr):
    """Expression like `sqrt(expression)`."""
    op = math.sqrt


class SinExpr(UnaryExpr):
    """Expression like `sin(expression)`."""
    op = math.sin


class CosExpr(UnaryExpr):
    """Expression like `cos(expression)`."""
    op = math.cos


cdef class ExprCons:
    """Constraints with a polynomial expressions and lower/upper bounds."""

    cdef public Expr expr
    cdef public object _lhs
    cdef public object _rhs

    def __init__(self, expr, lhs=None, rhs=None):
        self.expr = expr
        self._lhs = lhs
        self._rhs = rhs
        self._normalize()

    def _normalize(self) -> Expr:
        """Move constant children in expression to bounds"""

        if self._lhs is None and self._rhs is None:
            raise ValueError(
                "Ranged ExprCons (with both lhs and rhs) doesn't supported."
            )
        if not isinstance(self.expr, Expr):
            raise TypeError("expr must be an Expr instance")

        c = self.expr[CONST]
        self.expr = (self.expr - c)._normalize()

        if self._lhs is not None:
            self._lhs -= c
        if self._rhs is not None:
            self._rhs -= c

    def __lt__(self, other):
        if not self._rhs is None:
            raise TypeError("ExprCons already has upper bound")
        if self._lhs is None:
            raise TypeError("ExprCons must have a lower bound")
        if not _is_number(other):
            raise TypeError("Ranged ExprCons is not well defined!")

        return ExprCons(self.expr, lhs=self._lhs, rhs=float(other))

    def __gt__(self, other):
        if not self._lhs is None:
            raise TypeError("ExprCons already has lower bound")
        if self._rhs is None:
            raise TypeError("ExprCons must have an upper bound")
        if not _is_number(other):
            raise TypeError("Ranged ExprCons is not well defined!")

        return ExprCons(self.expr, lhs=float(other), rhs=self._rhs)

    def __repr__(self):
        return f"ExprCons({self.expr}, {self._lhs}, {self._rhs})"

    def __bool__(self):
        """Make sure that equality of expressions is not asserted with =="""

        msg = """Can't evaluate constraints as booleans.

If you want to add a ranged constraint of the form:
    lhs <= expression <= rhs
you have to use parenthesis to break the Python syntax for chained comparisons:
    lhs <= (expression <= rhs)
"""
        raise TypeError(msg)


def quicksum(termlist):
    """add linear expressions and constants much faster than Python's sum
    by avoiding intermediate data structures and adding terms inplace
    """
    result = Expr()
    for term in termlist:
        result += term
    return result


def quickprod(termlist):
    """multiply linear expressions and constants by avoiding intermediate
    data structures and multiplying terms inplace
    """
    result = Expr() + 1
    for term in termlist:
        result *= term
    return result


def _to_unary_expr(expr: Union[Expr, MatrixExpr], cls: Type[UnaryExpr]):
    if isinstance(expr, MatrixExpr):   
        res = np.empty(shape=expr.shape, dtype=object)
        res.flat = [cls(i) for i in expr.flat]
        return res.view(MatrixExpr)
    return cls(expr)


def exp(expr: Union[Expr, MatrixExpr]):
    """returns expression with exp-function"""
    return _to_unary_expr(expr, ExpExpr)


def log(expr: Union[Expr, MatrixExpr]):
    """returns expression with log-function"""
    return _to_unary_expr(expr, LogExpr)


def sqrt(expr: Union[Expr, MatrixExpr]):
    """returns expression with sqrt-function"""
    return _to_unary_expr(expr, SqrtExpr)


def sin(expr: Union[Expr, MatrixExpr]):
    """returns expression with sin-function"""
    return _to_unary_expr(expr, SinExpr)


def cos(expr: Union[Expr, MatrixExpr]):
    """returns expression with cos-function"""
    return _to_unary_expr(expr, CosExpr)
