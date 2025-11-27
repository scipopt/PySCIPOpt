##@file expr.pxi
from collections.abc import Hashable
from numbers import Number
from typing import Optional, Type, Union

include "matrix.pxi"


class Term:
    """A monomial term consisting of one or more variables."""

    __slots__ = ("vars", "HASH")

    def __init__(self, *vars: Variable):
        self.vars = tuple(sorted(vars, key=hash))
        self.HASH = hash(self.vars)

    def __getitem__(self, idx: int) -> Variable:
        return self.vars[idx]

    def __hash__(self) -> int:
        return self.HASH

    def __eq__(self, other: Term) -> bool:
        return self.HASH == other.HASH

    def __len__(self) -> int:
        return len(self.vars)

    def __mul__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            raise TypeError(
                f"unsupported operand type(s) for *: 'Term' and '{type(other)}'"
            )
        return Term(*self.vars, *other.vars)

    def __repr__(self) -> str:
        return f"Term({', '.join(map(str, self.vars))})"

    def degree(self) -> int:
        return self.__len__()

    def _to_nodes(self, start: int = 0, coef: float = 1) -> list[tuple]:
        """Convert term to list of nodes for SCIP expression construction"""
        if coef == 0:
            return []
        elif self.degree() == 0:
            return [(ConstExpr, coef)]
        else:
            nodes = [(Term, i) for i in self.vars]
            if coef != 1:
                nodes += [(ConstExpr, coef)]
            if len(nodes) > 1:
                nodes += [(ProdExpr, list(range(start, start + len(nodes))))]
            return nodes


CONST = Term()


class Expr:
    """Base class for mathematical expressions."""

    def __init__(self, children: Optional[dict[Union[Variable, Term, Expr], float]] = None):
        children = children or {}
        if not all(isinstance(i, (Variable, Term, Expr)) for i in children):
            raise TypeError("All keys must be Variable, Term or Expr instances")

        self.children = {
            (MonomialExpr.from_var(k) if isinstance(k, Variable) else k): v
            for k, v in children.items()
        }

    def __hash__(self) -> int:
        return frozenset(self.children.items()).__hash__()

    def __getitem__(self, key: Union[Variable, Term, Expr]) -> float:
        if not isinstance(key, (Term, Expr)):
            key = Term(key)
        return self.children.get(key, 0.0)

    def __iter__(self) -> Union[Term, Expr]:
        return iter(self.children)

    def __next__(self) -> Union[Term, Expr]:
        try:
            return next(self.children)
        except:
            raise StopIteration

    def __abs__(self) -> AbsExpr:
        return UnaryExpr.from_expr(self, AbsExpr)

    @staticmethod
    def _is_sum(expr: Expr) -> bool:
        return type(expr) is Expr or isinstance(expr, PolynomialExpr)

    def __add__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, Expr):
            if not self.children:
                return other
            if Expr._is_sum(self):
                if Expr._is_sum(other):
                    return Expr(self.to_dict(other.children))
                return Expr(self.to_dict({other: 1.0}))
            elif Expr._is_sum(other):
                return Expr(other.to_dict({self: 1.0}))
            return Expr({self: 1.0, other: 1.0})
        elif isinstance(other, MatrixExpr):
            return other.__add__(self)
        raise TypeError(
            f"unsupported operand type(s) for +: 'Expr' and '{type(other)}'"
        )

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, Expr):
            if not self.children:
                return ConstExpr(0.0)
            if isinstance(other, ConstExpr):
                if other[CONST] == 0:
                    return ConstExpr(0.0)
                return Expr({i: self[i] * other[CONST] for i in self if self[i] != 0})
            return ProdExpr(self, other)
        elif isinstance(other, MatrixExpr):
            return other.__mul__(self)
        raise TypeError(
            f"unsupported operand type(s) for *: 'Expr' and '{type(other)}'"
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, ConstExpr) and other[CONST] == 0:
            raise ZeroDivisionError("division by zero")
        if isinstance(other, Hashable) and hash(self) == hash(other):
            return ConstExpr(1.0)
        return self.__mul__(other.__pow__(-1.0))

    def __rtruediv__(self, other):
        return Expr.from_const_or_var(other).__truediv__(self)

    def __pow__(self, other):
        other = Expr.from_const_or_var(other)
        if not isinstance(other, ConstExpr):
            raise TypeError("exponent must be a number")

        if other[CONST] == 0:
            return ConstExpr(1.0)
        return PowExpr(self, other[CONST])

    def __rpow__(self, other):
        other = Expr.from_const_or_var(other)
        if not isinstance(other, ConstExpr):
            raise TypeError("base must be a number")
        if other[CONST] <= 0.0:
            raise ValueError("base must be positive")
        return exp(self * log(other))

    def __neg__(self) -> Expr:
        return self.__mul__(-1.0)

    def __sub__(self, other) -> Expr:
        return self.__add__(-other)

    def __rsub__(self, other) -> Expr:
        return self.__neg__().__add__(other)

    def __le__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, Expr):
            if isinstance(other, ConstExpr):
                return ExprCons(self, rhs=other[CONST])
            return (self - other).__le__(0)
        elif isinstance(other, MatrixExpr):
            return other.__ge__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __ge__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, Expr):
            if isinstance(other, ConstExpr):
                return ExprCons(self, lhs=other[CONST])
            return (self - other).__ge__(0)
        elif isinstance(other, MatrixExpr):
            return self.__le__(other)
        raise TypeError(f"Unsupported type {type(other)}")

    def __eq__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, Expr):
            if isinstance(other, ConstExpr):
                return ExprCons(self, lhs=other[CONST], rhs=other[CONST])
            return (self - other).__eq__(0)
        elif isinstance(other, MatrixExpr):
            return other.__ge__(self)
        raise TypeError(f"Unsupported type {type(other)}")

    def __repr__(self) -> str:
        return f"Expr({self.children})"

    @staticmethod
    def from_const_or_var(x):
        """Convert a number or variable to an expression."""

        if isinstance(x, Number):
            return PolynomialExpr.to_subclass({CONST: x})
        elif isinstance(x, Variable):
            return PolynomialExpr.to_subclass({Term(x): 1.0})
        return x

    def to_dict(
        self,
        other: Optional[dict[Union[Term, Expr], float]] = None,
    ) -> dict[Union[Term, Expr], float]:
        """Merge two dictionaries by summing values of common keys"""
        other = other or {}
        if not isinstance(other, dict):
            raise TypeError("other must be a dict")

        res = self.children.copy()
        for child, coef in other.items():
            res[child] = res.get(child, 0.0) + coef

        return res

    def _remove_zero(self) -> dict:
        return {k: v for k, v in self.children.items() if v != 0}

    def _normalize(self) -> Expr:
        return Expr(self._remove_zero())

    def degree(self) -> float:
        return max((i.degree() for i in self)) if self.children else float("inf")

    def _to_nodes(self, start: int = 0, coef: float = 1) -> list[tuple]:
        """Convert expression to list of nodes for SCIP expression construction"""
        nodes, indices = [], []
        for child, c in self.children.items():
            if (child_nodes := child._to_nodes(start + len(nodes), c)):
                nodes += child_nodes
                indices += [start + len(nodes) - 1]

        if type(self) is PowExpr:
            nodes += [(ConstExpr, self.expo)]
            indices += [start + len(nodes) - 1]
        elif type(self) is ProdExpr and self.coef != 1:
            nodes += [(ConstExpr, self.coef)]
            indices += [start + len(nodes) - 1]
        return nodes + [(type(self), indices)]


class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict[Term, float]] = None):
        if children and not all(isinstance(t, Term) for t in children):
            raise TypeError("All keys must be Term instances")

        super().__init__(children)

    def __add__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            return PolynomialExpr.to_subclass(self.to_dict(other.children))
        return super().__add__(other)

    def __iadd__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            for child, coef in other.children.items():
                self.children[child] = self.children.get(child, 0.0) + coef
            return self
        return super().__iadd__(other)

    def __mul__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, PolynomialExpr):
            children = {}
            for i in self:
                for j in other:
                    child = i * j
                    children[child] = children.get(child, 0.0) + self[i] * other[j]
            return PolynomialExpr.to_subclass(children)
        return super().__mul__(other)

    def __truediv__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, ConstExpr):
            return self.__mul__(1.0 / other[CONST])
        return super().__truediv__(other)

    def __pow__(self, other):
        other = Expr.from_const_or_var(other)
        if (
            isinstance(other, ConstExpr)
            and other[CONST].is_integer()
            and other[CONST] > 0
        ):
            res = 1
            for _ in range(int(other[CONST])):
                res *= self
            return res
        return super().__pow__(other)

    @classmethod
    def to_subclass(cls, children: dict[Term, float]) -> PolynomialExpr:
        if len(children) == 0:
            return ConstExpr(0.0)
        elif len(children) == 1:
            if CONST in children:
                return ConstExpr(children[CONST])
            return MonomialExpr(children)
        return cls(children)

    def _normalize(self) -> PolynomialExpr:
        return PolynomialExpr(self._remove_zero())

    def _to_nodes(self, start: int = 0, coef: float = 1) -> list[tuple]:
        """Convert expression to list of nodes for SCIP expression construction"""
        nodes = []
        for child, c in self.children.items():
            nodes += child._to_nodes(start + len(nodes), c)

        if len(nodes) > 1:
            return nodes + [(Expr, list(range(start, start + len(nodes))))]
        return nodes


class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, constant: float = 0.0):
        super().__init__({CONST: constant})

    def __abs__(self) -> ConstExpr:
        return ConstExpr(abs(self[CONST]))

    def __pow__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, ConstExpr):
            return ConstExpr(self[CONST] ** other[CONST])
        return super().__pow__(other)


class MonomialExpr(PolynomialExpr):
    """Expression like `x**3`."""

    def __init__(self, children: dict[Term, float]):
        if len(children) != 1:
            raise ValueError("MonomialExpr must have exactly one child")

        super().__init__(children)

    @staticmethod
    def from_var(var: Variable, coef: float = 1.0) -> MonomialExpr:
        return MonomialExpr({Term(var): coef})


class FuncExpr(Expr):
    def __init__(
        self,
        children: Optional[dict[Union[Variable, Term, Expr], float]] = None,
    ):
        if children and any((i is CONST) for i in children):
            raise ValueError("FuncExpr can't have Term without Variable as a child")
        super().__init__(children)

    def degree(self) -> float:
        return float("inf")


class ProdExpr(FuncExpr):
    """Expression like `coefficient * expression`."""

    def __init__(self, *children: Expr, coef: float = 1.0):
        if len(set(children)) != len(children):
            raise ValueError("ProdExpr can't have duplicate children")
        super().__init__({i: 1.0 for i in children})
        self.coef = coef

    def __hash__(self) -> int:
        return (frozenset(self), self.coef).__hash__()

    def __add__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, ProdExpr) and hash(self) == hash(other):
            return ProdExpr(*self, coef=self.coef + other.coef)
        return super().__add__(other)

    def __mul__(self, other):
        other = Expr.from_const_or_var(other)
        if isinstance(other, ConstExpr):
            if other[CONST] == 0:
                return ConstExpr(0.0)
            return ProdExpr(*self, coef=self.coef * other[CONST])
        return super().__mul__(other)

    def __repr__(self) -> str:
        return f"ProdExpr({{{tuple(self)}: {self.coef}}})"

    def _normalize(self) -> Union[ConstExpr, ProdExpr]:
        if self.coef == 0:
            return ConstExpr(0.0)
        return self


class PowExpr(FuncExpr):
    """Expression like `pow(expression, exponent)`."""

    def __init__(self, base: Union[Variable, Term, Expr], expo: float = 1.0):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self) -> int:
        return (frozenset(self), self.expo).__hash__()

    def __repr__(self) -> str:
        return f"PowExpr({tuple(self)}, {self.expo})"

    def _normalize(self) -> Expr:
        if self.expo == 0:
            return ConstExpr(1.0)
        elif self.expo == 1:
            return tuple(self)[0]
        return self


class UnaryExpr(FuncExpr):
    """Expression like `f(expression)`."""

    def __init__(self, expr: Union[Number, Variable, Term, Expr]):
        if isinstance(expr, Number):
            expr = ConstExpr(expr)
        super().__init__({expr: 1.0})

    def __hash__(self) -> int:
        return frozenset(self).__hash__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({tuple(self)[0]})"

    @staticmethod
    def from_expr(expr: Union[Expr, MatrixExpr], cls: Type[UnaryExpr]) -> UnaryExpr:
        if isinstance(expr, MatrixExpr):
            res = np.empty(shape=expr.shape, dtype=object)
            res.flat = [cls(i) for i in expr.flat]
            return res.view(MatrixExpr)
        return cls(expr)

    def _to_nodes(self, start: int = 0, coef: float = 1) -> list[tuple]:
        """Convert expression to list of nodes for SCIP expression construction"""
        nodes = []
        for child, c in self.children.items():
            nodes += child._to_nodes(start + len(nodes), c)

        return nodes + [(type(self), start + len(nodes) - 1)]


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


class ExprCons:
    """Constraints with a polynomial expressions and lower/upper bounds."""

    def __init__(self, expr: Expr, lhs: Optional[float] = None, rhs: Optional[float] = None):
        if not isinstance(expr, Expr):
            raise TypeError("expr must be an Expr instance")
        if lhs is None and rhs is None:
            raise ValueError(
                "Ranged ExprCons (with both lhs and rhs) doesn't supported"
            )
        self.expr = expr
        self._lhs = lhs
        self._rhs = rhs
        self._normalize()

    def _normalize(self):
        """Move constant children in expression to bounds"""
        c = self.expr[CONST]
        self.expr = (self.expr - c)._normalize()
        if self._lhs is not None:
            self._lhs -= c
        if self._rhs is not None:
            self._rhs -= c

    def __le__(self, other) -> ExprCons:
        if not self._rhs is None:
            raise TypeError("ExprCons already has upper bound")
        if self._lhs is None:
            raise TypeError("ExprCons must have a lower bound")
        if not isinstance(other, Number):
            raise TypeError("Ranged ExprCons is not well defined!")

        return ExprCons(self.expr, lhs=self._lhs, rhs=float(other))

    def __ge__(self, other) -> ExprCons:
        if not self._lhs is None:
            raise TypeError("ExprCons already has lower bound")
        if self._rhs is None:
            raise TypeError("ExprCons must have an upper bound")
        if not isinstance(other, Number):
            raise TypeError("Ranged ExprCons is not well defined!")

        return ExprCons(self.expr, lhs=float(other), rhs=self._rhs)

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


def exp(expr: Union[Expr, MatrixExpr]) -> ExpExpr:
    """returns expression with exp-function"""
    return UnaryExpr.from_expr(expr, ExpExpr)


def log(expr: Union[Expr, MatrixExpr]) -> LogExpr:
    """returns expression with log-function"""
    return UnaryExpr.from_expr(expr, LogExpr)


def sqrt(expr: Union[Expr, MatrixExpr]) -> SqrtExpr:
    """returns expression with sqrt-function"""
    return UnaryExpr.from_expr(expr, SqrtExpr)


def sin(expr: Union[Expr, MatrixExpr]) -> SinExpr:
    """returns expression with sin-function"""
    return UnaryExpr.from_expr(expr, SinExpr)


def cos(expr: Union[Expr, MatrixExpr]) -> CosExpr:
    """returns expression with cos-function"""
    return UnaryExpr.from_expr(expr, CosExpr)
