##@file expr.pxi
from numbers import Number
from typing import Iterator, Optional, Type, Union

import numpy as np

include "matrix.pxi"


cdef class Term:
    """A monomial term consisting of one or more variables."""

    cdef public tuple vars
    cdef readonly int _hash
    __slots__ = ("vars", "_hash")

    def __init__(self, *vars: Variable):
        if not all(isinstance(i, Variable) for i in vars):
            raise TypeError("All arguments must be Variable instances")

        self.vars = tuple(sorted(vars, key=hash))
        self._hash = hash(self.vars)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self.vars)

    def __getitem__(self, key: int) -> Variable:
        return self.vars[key]

    def __hash__(self) -> int:
        return self._hash

    def __len__(self) -> int:
        return len(self.vars)

    def __eq__(self, other) -> bool:
        return isinstance(other, Term) and self._hash == other._hash

    def __mul__(self, Term other) -> Term:
        return Term(*self.vars, *other.vars)

    def __repr__(self) -> str:
        return f"Term({', '.join(map(str, self.vars))})"

    def degree(self) -> int:
        return len(self)

    def _to_node(self, coef: float = 1, start: int = 0) -> list[tuple]:
        """Convert term to list of node for SCIP expression construction"""
        if coef == 0:
            return []
        elif self.degree() == 0:
            return [(ConstExpr, coef)]
        else:
            node = [(Variable, i) for i in self]
            if coef != 1:
                node.append((ConstExpr, coef))
            if len(node) > 1:
                node.append((ProdExpr, list(range(start, start + len(node)))))
            return node


CONST = Term()


cdef class _ExprKey:

    cdef public Expr expr
    __slots__ = ("expr",)

    def __init__(self, Expr expr):
        self.expr = expr

    def __hash__(self) -> int:
        return hash(self.expr)

    def __eq__(self, other) -> bool:
        return isinstance(other, _ExprKey) and self.expr._is_equal(other.expr)

    def __repr__(self) -> str:
        return repr(self.expr)

    def degree(self) -> float:
        return self.expr.degree()

    def _to_node(self, coef: float = 1, start: int = 0) -> list[tuple]:
        return self.expr._to_node(coef, start)

    @staticmethod
    def wrap(x):
        return _ExprKey(x) if isinstance(x, Expr) else x

    @staticmethod
    def unwrap(x):
        return x.expr if isinstance(x, _ExprKey) else x


cdef class Expr:
    """Base class for mathematical expressions."""

    cdef public dict _children
    __slots__ = ("_children",)

    def __init__(
        self,
        children: Optional[dict[Union[Term, Expr, _ExprKey], float]] = None,
    ):
        if children and not all(isinstance(i, (Term, Expr, _ExprKey)) for i in children):
            raise TypeError("All keys must be Term or Expr instances")

        self._children = {_ExprKey.wrap(k): v for k, v in (children or {}).items()}

    @property
    def children(self):
        return {_ExprKey.unwrap(k): v for k, v in self._children.items()}

    def __hash__(self) -> int:
        return (type(self), frozenset(self._children.items())).__hash__()

    def __getitem__(self, key: Union[Variable, Term, Expr]) -> float:
        if not isinstance(key, (Variable, Term, Expr, _ExprKey)):
            raise TypeError("key must be Variable, Term, or Expr")

        if isinstance(key, Variable):
            key = Term(key)
        return self._children.get(_ExprKey.wrap(key), 0.0)

    def __iter__(self) -> Iterator[Union[Term, Expr]]:
        for i in self._children:
            yield _ExprKey.unwrap(i)

    def __bool__(self):
        return bool(self._children)

    def __abs__(self) -> AbsExpr:
        return AbsExpr(self)

    def __add__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, Expr):
            if not self or Expr._is_zero(self):
                return other.copy()
            elif not other or Expr._is_zero(other):
                return self.copy()
            elif Expr._is_sum(self):
                return Expr(
                    <dict>self._to_dict(
                        other._children if Expr._is_sum(other) else {other: 1.0}
                    )
                )
            elif Expr._is_sum(other):
                return Expr(<dict>other._to_dict({self: 1.0}))
            elif self._is_equal(other):
                return self.__mul__(2.0)
            return Expr({self: 1.0, other: 1.0})

        elif isinstance(other, MatrixExpr):
            return other.__add__(self)
        raise TypeError(
            f"unsupported operand type(s) for +: 'Expr' and '{type(other)}'"
        )

    def __iadd__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_sum(self):
            if Expr._is_sum(other):
                self._to_dict(other._children, copy=False)
            else:
                self._to_dict({other: 1.0}, copy=False)
            return self
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, Expr):
            left, right = (self, other) if Expr._is_const(self) else (other, self)
            if not left or not right or Expr._is_zero(left) or Expr._is_zero(right):
                return ConstExpr(0.0)
            elif Expr._is_const(left):
                if left[CONST] == 1:
                    return right.copy()
                elif Expr._is_sum(right):
                    return Expr({
                        k: v * left[CONST] for k, v in right._children.items() if v != 0
                    })
                return Expr({right: left[CONST]})
            elif self._is_equal(other):
                return PowExpr(self, 2.0)
            return ProdExpr(self, other)

        elif isinstance(other, MatrixExpr):
            return other.__mul__(self)
        raise TypeError(
            f"unsupported operand type(s) for *: 'Expr' and '{type(other)}'"
        )

    def __imul__(self, other):
        other = Expr._from_const_or_var(other)
        if self and Expr._is_sum(self) and Expr._is_const(other) and other[CONST] != 0:
            self._children = {
                k: v * other[CONST] for k, v in self._children.items() if v != 0
            }
            return self
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_zero(other):
            raise ZeroDivisionError("division by zero")
        if self._is_equal(other):
            return ConstExpr(1.0)
        return self.__mul__(other.__pow__(-1.0))

    def __rtruediv__(self, other):
        return Expr._from_const_or_var(other).__truediv__(self)

    def __pow__(self, other):
        other = Expr._from_const_or_var(other)
        if not Expr._is_const(other):
            raise TypeError("exponent must be a number")
        if Expr._is_zero(other):
            return ConstExpr(1.0)
        return PowExpr(self, other[CONST])

    def __rpow__(self, other):
        other = Expr._from_const_or_var(other)
        if not Expr._is_const(other):
            raise TypeError("base must be a number")
        if other[CONST] <= 0.0:
            raise ValueError("base must be positive")
        return exp(self.__mul__(log(other)))

    def __neg__(self):
        return self.__mul__(-1.0)

    def __sub__(self, other):
        other = Expr._from_const_or_var(other)
        if self._is_equal(other):
            return ConstExpr(0.0)
        return self.__add__(-other)

    def __isub__(self, other):
        other = Expr._from_const_or_var(other)
        if self._is_equal(other):
            return ConstExpr(0.0)
        return self.__iadd__(-other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __le__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, Expr):
            if Expr._is_const(self):
                return ExprCons(other, lhs=self[CONST])
            elif Expr._is_const(other):
                return ExprCons(self, rhs=other[CONST])
            return self.__add__(-other).__le__(ConstExpr(0))
        elif isinstance(other, MatrixExpr):
            return other.__ge__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __ge__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, Expr):
            if Expr._is_const(self):
                return ExprCons(other, rhs=self[CONST])
            elif Expr._is_const(other):
                return ExprCons(self, lhs=other[CONST])
            return self.__add__(-other).__ge__(ConstExpr(0.0))
        elif isinstance(other, MatrixExpr):
            return other.__le__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __eq__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, Expr):
            if Expr._is_const(self):
                return ExprCons(other, lhs=self[CONST], rhs=self[CONST])
            elif Expr._is_const(other):
                return ExprCons(self, lhs=other[CONST], rhs=other[CONST])
            return self.__add__(-other).__eq__(ConstExpr(0.0))
        elif isinstance(other, MatrixExpr):
            return other.__eq__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __repr__(self) -> str:
        return f"Expr({self._children})"

    @staticmethod
    def _from_const_or_var(x):
        """Convert a number or variable to an expression."""

        if isinstance(x, Number):
            return ConstExpr(<float>x)
        elif isinstance(x, Variable):
            return MonomialExpr._from_var(x)
        return x

    def _to_dict(
            self,
            other: dict[Union[Term, Expr, _ExprKey], float],
            copy: bool = True,
        ) -> dict[Union[Term, _ExprKey], float]:
        """Merge two dictionaries by summing values of common keys"""
        children = self._children.copy() if copy else self._children
        for child, coef in other.items():
            key = _ExprKey.wrap(child)
            children[key] = children.get(key, 0.0) + coef
        return children

    def _normalize(self) -> Expr:
        self._children = {k: v for k, v in self._children.items() if v != 0}
        return self

    def degree(self) -> float:
        return max((i.degree() for i in self._children)) if self else 0

    def copy(self) -> Expr:
        return type(self)(self._children.copy())

    def _to_node(self, coef: float = 1, start: int = 0) -> list[tuple]:
        """Convert expression to list of node for SCIP expression construction"""
        if coef == 0:
            return []

        node, index = [], []
        for k, v in self._children.items():
            if v != 0 and (child_node := k._to_node(v, start + len(node))):
                node.extend(child_node)
                index.append(start + len(node) - 1)

        if node:
            if issubclass(type(self), PolynomialExpr):
                if len(node) > 1:
                    node.append((Expr, index))
            elif isinstance(self, UnaryExpr):
                node.append((type(self), index[0]))
            else:
                if type(self) is PowExpr:
                    node.append((ConstExpr, self.expo))
                    index.append(start + len(node) - 1)
                elif type(self) is ProdExpr and self.coef != 1:
                    node.append((ConstExpr, self.coef))
                    index.append(start + len(node) - 1)
                node.append((type(self), index))

            if coef != 1:
                node.append((ConstExpr, coef))
                node.append((ProdExpr, [start + len(node) - 2, start + len(node) - 1]))

        return node

    def _fchild(self) -> Union[Term, _ExprKey]:
        return next(iter(self._children))

    def _is_equal(self, other) -> bool:
        return (
            isinstance(other, Expr)
            and (
                (Expr._is_sum(self) and Expr._is_sum(other))
                or type(self) is type(other)
            )
            and len(self._children) == len(other._children)
            and self._children == other._children
        )

    @staticmethod
    def _is_sum(expr) -> bool:
        return type(expr) is Expr or isinstance(expr, PolynomialExpr)

    @staticmethod
    def _is_const(expr):
        return (
            Expr._is_sum(expr)
            and len(expr._children) == 1
            and expr._fchild() is CONST
        )

    @staticmethod
    def _is_zero(expr):
        return  Expr._is_const(expr) and expr[CONST] == 0


class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict[Term, float]] = None):
        if children and not all(isinstance(t, Term) for t in children):
            raise TypeError("All keys must be Term instances")

        super().__init__(<dict>children)

    def __hash__(self) -> int:
        return (Expr, frozenset(self._children.items())).__hash__()

    def __add__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PolynomialExpr) and not Expr._is_zero(other):
            return PolynomialExpr._to_subclass(<dict>self._to_dict(other._children))
        return super().__add__(other)

    def __iadd__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            self._to_dict(other._children, copy=False)
            return self
        return super().__iadd__(other)

    def __mul__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PolynomialExpr) and not (
            Expr._is_const(other) and (other[CONST] == 0 or other[CONST] == 1)
        ):
            children = {}
            for i in self:
                for j in other:
                    child = i * j
                    children[child] = children.get(child, 0.0) + self[i] * other[j]
            return PolynomialExpr._to_subclass(children)
        return super().__mul__(other)

    def __truediv__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_const(other):
            return self.__mul__(1.0 / other[CONST])
        return super().__truediv__(other)

    def __pow__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_const(other) and other[CONST].is_integer() and other[CONST] > 0:
            res = ConstExpr(1.0)
            for _ in range(int(other[CONST])):
                res *= self
            return res
        return super().__pow__(other)

    @classmethod
    def _to_subclass(cls, children: dict[Term, float]) -> PolynomialExpr:
        if len(children) == 0:
            return ConstExpr(0.0)
        elif len(children) == 1:
            if CONST in children:
                return ConstExpr(children[CONST])
            return MonomialExpr(children)
        return cls(children)


class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, float constant = 0.0):
        super().__init__({CONST: constant})

    def __abs__(self) -> ConstExpr:
        return ConstExpr(abs(self[CONST]))

    def __iadd__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_const(other):
            self._children[CONST] += other[CONST]
            return self
        if isinstance(other, PolynomialExpr):
            return self.__add__(other)
        return super().__iadd__(other)

    def __pow__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_const(other):
            return ConstExpr(self[CONST] ** other[CONST])
        return super().__pow__(other)

    def copy(self) -> ConstExpr:
        return ConstExpr(self[CONST])


class MonomialExpr(PolynomialExpr):
    """Expression like `x**3`."""

    def __init__(self, children: dict[Term, float]):
        if len(children) != 1:
            raise ValueError("MonomialExpr must have exactly one child")

        super().__init__(children)

    def __iadd__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            child = self._fchild()
            if isinstance(other, MonomialExpr) and child == other._fchild():
                self._children[child] += other[child]
            else:
                self = self.__add__(other)
            return self
        return super().__iadd__(other)

    @staticmethod
    def _from_var(var: Variable, coef: float = 1.0) -> MonomialExpr:
        return MonomialExpr({Term(var): coef})


class FuncExpr(Expr):
    def __init__(self, children: Optional[dict[Union[Term, Expr, _ExprKey], float]] = None):
        if children and any((i is CONST) for i in children):
            raise ValueError("FuncExpr can't have Term without Variable as a child")

        super().__init__(children)

    def degree(self) -> float:
        return float("inf")

    def _is_child_equal(self, other) -> bool:
        return (
            type(other) is type(self)
            and len(self._children) == len(other._children)
            and self._children.keys() == other._children.keys()
        )


class ProdExpr(FuncExpr):
    """Expression like `coefficient * expression`."""

    __slots__ = ("coef",)

    def __init__(self, *children: Union[Term, Expr], float coef = 1.0):
        if len(set(children)) != len(children):
            raise ValueError("ProdExpr can't have duplicate children")

        super().__init__(dict.fromkeys(children, 1.0))
        self.coef = coef

    def __hash__(self) -> int:
        return (type(self), frozenset(self), self.coef).__hash__()

    def __add__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, ProdExpr) and self._is_child_equal(other):
            return ProdExpr(*self, coef=self.coef + other.coef)
        return super().__add__(other)

    def __iadd__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, ProdExpr) and self._is_child_equal(other):
            self.coef += other.coef
            return self
        return super().__iadd__(other)

    def __mul__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_const(other) and (other[CONST] != 0 or other[CONST] != 1):
            return ProdExpr(*self, coef=self.coef * other[CONST])
        return super().__mul__(other)

    def __imul__(self, other):
        other = Expr._from_const_or_var(other)
        if Expr._is_const(other):
            if other[CONST] == 0:
                self = ConstExpr(0.0)
            else:
                self.coef *= other[CONST]
            return self
        return super().__imul__(other)

    def __repr__(self) -> str:
        return f"ProdExpr({{{tuple(self)}: {self.coef}}})"

    def _normalize(self) -> Union[ConstExpr, ProdExpr]:
        if self.coef == 0:
            self = ConstExpr(0.0)
        return self

    def copy(self) -> ProdExpr:
        return ProdExpr(*self._children.keys(), coef=self.coef)


class PowExpr(FuncExpr):
    """Expression like `pow(expression, exponent)`."""

    __slots__ = ("expo",)

    def __init__(self, base: Union[Term, Expr, _ExprKey], float expo = 1.0):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self) -> int:
        return (type(self), frozenset(self), self.expo).__hash__()

    def __mul__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PowExpr) and self._is_child_equal(other):
            return PowExpr(self._fchild(), self.expo + other.expo)
        return super().__mul__(other)

    def __imul__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PowExpr) and self._is_child_equal(other):
            self.expo += other.expo
            return self
        return super().__imul__(other)

    def __truediv__(self, other):
        other = Expr._from_const_or_var(other)
        if (
            isinstance(other, PowExpr)
            and not self._is_equal(other)
            and self._is_child_equal(other)
        ):
            return PowExpr(self._fchild(), self.expo - other.expo)
        return super().__truediv__(other)

    def __repr__(self) -> str:
        return f"PowExpr({self._fchild()}, {self.expo})"

    def _normalize(self) -> Expr:
        if self.expo == 0:
            self = ConstExpr(1.0)
        elif self.expo == 1:
            self = _ExprKey.unwrap(self._fchild())
            if isinstance(self, Term):
                self = MonomialExpr({self: 1.0})
        return self

    def copy(self) -> PowExpr:
        return PowExpr(self._fchild(), self.expo)


class UnaryExpr(FuncExpr):
    """Expression like `f(expression)`."""

    def __init__(self, expr: Union[Number, Variable, Term, Expr, _ExprKey]):
        if isinstance(expr, Number):
            expr = ConstExpr(<float>expr)
        super().__init__({expr: 1.0})

    def __hash__(self) -> int:
        return (type(self), frozenset(self)).__hash__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._fchild()})"

    def copy(self) -> UnaryExpr:
        return type(self)(self._fchild())

    @staticmethod
    def _to_subclass(
        x: Union[Number, Variable, Term, Expr, MatrixExpr],
        cls: Type[UnaryExpr],
    ) -> Union[UnaryExpr, MatrixExpr]:
        if isinstance(x, Variable):
            x = Term(x)
        elif isinstance(x, MatrixExpr):
            res = np.empty(shape=x.shape, dtype=object)
            res.flat = [cls(Term(i) if isinstance(i, Variable) else i) for i in x.flat]
            return res.view(MatrixExpr)
        return cls(x)


class AbsExpr(UnaryExpr):
    """Expression like `abs(expression)`."""
    ...


class ExpExpr(UnaryExpr):
    """Expression like `exp(expression)`."""
    ...


class LogExpr(UnaryExpr):
    """Expression like `log(expression)`."""
    ...


class SqrtExpr(UnaryExpr):
    """Expression like `sqrt(expression)`."""
    ...


class SinExpr(UnaryExpr):
    """Expression like `sin(expression)`."""
    ...


class CosExpr(UnaryExpr):
    """Expression like `cos(expression)`."""
    ...


cdef class ExprCons:
    """Constraints with a polynomial expressions and lower/upper bounds."""

    cdef public Expr expr
    cdef public object _lhs
    cdef public object _rhs
    cdef readonly object _lhs
    cdef readonly object _rhs

    def __init__(
        self,
        Expr expr,
        lhs: Optional[float] = None,
        rhs: Optional[float] = None,
    ):
        if lhs is None and rhs is None:
            raise ValueError("ExprCons (with both lhs and rhs) doesn't supported")

        self.expr = expr
        self._lhs = lhs
        self._rhs = rhs
        self._normalize()

    def _normalize(self) -> ExprCons:
        """Move constant children in expression to bounds"""
        c = self.expr[CONST]
        self.expr = (<Expr>(self.expr - c))._normalize()
        if self._lhs is not None:
            self._lhs = <float>self._lhs - c
        if self._rhs is not None:
            self._rhs = <float>self._rhs - c
        return self

    def __le__(self, float other) -> ExprCons:
        if not self._rhs is None:
            raise TypeError("ExprCons already has upper bound")

        return ExprCons(self.expr, lhs=<float>self._lhs, rhs=float(other))

    def __ge__(self, float other) -> ExprCons:
        if not self._lhs is None:
            raise TypeError("ExprCons already has lower bound")

        return ExprCons(self.expr, lhs=float(other), rhs=<float>self._rhs)

    def __eq__(self, _) -> ExprCons:
        raise NotImplementedError("ExprCons can only support with '<=' or '>='.")

    def __repr__(self) -> str:
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


def quicksum(expressions) -> Expr:
    """add linear expressions and constants much faster than Python's sum
    by avoiding intermediate data structures and adding terms inplace
    """
    res = ConstExpr(0.0)
    for i in expressions:
        res += i
    return res


def quickprod(expressions) -> Expr:
    """multiply linear expressions and constants by avoiding intermediate
    data structures and multiplying terms inplace
    """
    res = ConstExpr(1.0)
    for i in expressions:
        res *= i
    return res


def exp(x: Union[Number, Variable, Expr, MatrixExpr]) -> Union[ExpExpr, MatrixExpr]:
    """returns expression with exp-function"""
    return UnaryExpr._to_subclass(x, ExpExpr)


def log(x: Union[Number, Variable, Expr, MatrixExpr]) -> Union[LogExpr, MatrixExpr]:
    """returns expression with log-function"""
    return UnaryExpr._to_subclass(x, LogExpr)


def sqrt(x: Union[Number, Variable, Expr, MatrixExpr]) -> Union[SqrtExpr, MatrixExpr]:
    """returns expression with sqrt-function"""
    return UnaryExpr._to_subclass(x, SqrtExpr)


def sin(x: Union[Number, Variable, Expr, MatrixExpr]) -> Union[SinExpr, MatrixExpr]:
    """returns expression with sin-function"""
    return UnaryExpr._to_subclass(x, SinExpr)


def cos(x: Union[Number, Variable, Expr, MatrixExpr]) -> Union[CosExpr, MatrixExpr]:
    """returns expression with cos-function"""
    return UnaryExpr._to_subclass(x, CosExpr)
