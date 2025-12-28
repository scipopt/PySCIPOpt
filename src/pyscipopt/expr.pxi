##@file expr.pxi
from numbers import Number
from typing import Iterator, Optional, Type, Union

from pyscipopt._decorator import to_array

cimport cython
from cpython.object cimport Py_LE, Py_EQ, Py_GE

include "matrix.pxi"


cdef class Term:
    """A monomial term consisting of one or more variables."""

    cdef readonly tuple vars
    cdef int _hash
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
        return isinstance(other, Term) and hash(self) == hash(other)

    def __mul__(self, Term other) -> Term:
        return Term(*self.vars, *other.vars)

    def __repr__(self) -> str:
        return f"Term({', '.join(map(str, self.vars))})"

    cpdef int degree(self):
        return len(self)

    cpdef list[tuple] _to_node(self, float coef = 1, int start = 0):
        """Convert term to list of node for SCIP expression construction"""
        cdef list[tuple] node
        if coef == 0:
            node = []
        elif self.degree() == 0:
            node = [(ConstExpr, coef)]
        else:
            node = [(Variable, i) for i in self]
            if coef != 1:
                node.append((ConstExpr, coef))
            if len(node) > 1:
                node.append((ProdExpr, list(range(start, start + len(node)))))
        return node


CONST = Term()


cdef class _ExprKey:

    cdef readonly Expr expr
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

    cdef readonly dict _children
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
        return frozenset(self._children.items()).__hash__()

    def __getitem__(self, key: Union[Variable, Term, Expr, _ExprKey]) -> float:
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
            if Expr._is_zero(self):
                return other.copy()
            elif Expr._is_zero(other):
                return self.copy()
            elif Expr._is_sum(self):
                return Expr(self._to_dict(other))
            elif Expr._is_sum(other):
                return Expr(other._to_dict(other))
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
        if Expr._is_zero(other):
            return self
        elif Expr._is_sum(self):
            self._to_dict(other, copy=False)
            if isinstance(self, PolynomialExpr) and isinstance(other, PolynomialExpr):
                return self._to_polynomial(PolynomialExpr)
            return self._to_polynomial(Expr)
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, Expr):
            left, right = (self, other) if Expr._is_const(self) else (other, self)
            if Expr._is_zero(left) or Expr._is_zero(right):
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
            if isinstance(self, PolynomialExpr) and isinstance(other, PolynomialExpr):
                return self._to_polynomial(PolynomialExpr)
            return self._to_polynomial(Expr)
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

    def __richcmp__(
            self,
            other: Union[Number, Variable, Expr, MatrixExpr],
            int op,
        ) -> Union[ExprCons, MatrixExprCons]:
        other = Expr._from_const_or_var(other)
        if not isinstance(other, (Expr, MatrixExpr)):
            raise TypeError(f"Unsupported type {type(other)}")

        if op == Py_LE:
            if isinstance(other, Expr):
                if Expr._is_const(self):
                    return ExprCons(other, lhs=self[CONST])
                elif Expr._is_const(other):
                    return ExprCons(self, rhs=other[CONST])
                return ExprCons(<Expr>self.__add__(-other), lhs=0.0)
            return other >= self

        elif op == Py_GE:
            if isinstance(other, Expr):
                if Expr._is_const(self):
                    return ExprCons(other, rhs=self[CONST])
                elif Expr._is_const(other):
                    return ExprCons(self, lhs=other[CONST])
                return ExprCons(<Expr>self.__add__(-other), rhs=0.0)
            return other <= self

        elif op == Py_EQ:
            if isinstance(other, Expr):
                if Expr._is_const(self):
                    return ExprCons(other, lhs=self[CONST], rhs=self[CONST])
                elif Expr._is_const(other):
                    return ExprCons(self, lhs=other[CONST], rhs=other[CONST])
                return ExprCons(<Expr>self.__add__(-other), lhs=0.0, rhs=0.0)
            return other == self

        raise NotImplementedError("Expr can only support with '<=', '>=', or '=='.")

    def __repr__(self) -> str:
        return f"Expr({self._children})"

    def _normalize(self) -> Expr:
        self._children = {k: v for k, v in self._children.items() if v != 0}
        return self

    def degree(self) -> float:
        return max((i.degree() for i in self._children)) if self else 0

    def copy(self) -> Expr:
        return type(self)(self._children.copy())

    @staticmethod
    def _from_const_or_var(x):
        """Convert a number or variable to an expression."""

        if isinstance(x, Number):
            return ConstExpr(<float>x)
        elif isinstance(x, Variable):
            return PolynomialExpr._from_var(x)
        return x

    cdef dict _to_dict(self, Expr other, bool copy = True):
        cdef dict children = self._children.copy() if copy else self._children
        cdef object child
        cdef float coef
        for child, coef in (
            other._children if Expr._is_sum(other) else {other: 1.0}
        ).items():
            key = _ExprKey.wrap(child)
            children[key] = children.get(key, 0.0) + coef
        return children

    cpdef list[tuple] _to_node(self, float coef = 1, int start = 0):
        """Convert expression to list of node for SCIP expression construction"""
        cdef list[tuple] node = []
        cdef list[tuple] child_node
        cdef list[int] index = []
        cdef object k
        cdef float v

        if coef == 0:
            return node

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
        return isinstance(expr, Expr) and (
            not expr or (Expr._is_const(expr) and expr[CONST] == 0)
        )

    cdef Expr _to_polynomial(self, cls: Type[Expr]):
        cdef Expr res = (
            ConstExpr.__new__(ConstExpr) if Expr._is_const(self) else cls.__new__(cls)
        )
        (<Expr>res)._children = self._children
        return res


cdef class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict[Term, float]] = None):
        if children and not all(isinstance(t, Term) for t in children):
            raise TypeError("All keys must be Term instances")

        super().__init__(<dict>children)

    def __add__(self, other):
        other = Expr._from_const_or_var(other)
        if isinstance(other, PolynomialExpr) and not Expr._is_zero(other):
            return PolynomialExpr._to_subclass(self._to_dict(other))
        return super().__add__(other)

    def __mul__(self, other):
        cdef dict[Term, float] children
        cdef Term k1, k2, child
        cdef float v1, v2
        other = Expr._from_const_or_var(other)
        if self and isinstance(other, PolynomialExpr) and not (
            Expr._is_const(other) and (other[CONST] == 0 or other[CONST] == 1)
        ):
            children = {}
            for k1, v1 in self._children.items():
                for k2, v2 in other._children.items():
                    child = k1 * k2
                    children[child] = children.get(child, 0.0) + v1 * v2
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

    @staticmethod
    def _from_var(var: Variable, float coef = 1.0) -> PolynomialExpr:
        return PolynomialExpr({Term(var): coef})

    @classmethod
    def _to_subclass(cls, children: dict[Term, float]) -> PolynomialExpr:
        if len(children) == 0:
            return ConstExpr(0.0)
        elif len(children) == 1 and CONST in children:
            return ConstExpr(children[CONST])
        return cls(children)


cdef class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, float constant = 0.0):
        super().__init__({CONST: constant})

    def __abs__(self) -> ConstExpr:
        return ConstExpr(abs(self[CONST]))

    def __neg__(self) -> ConstExpr:
        return ConstExpr(-self[CONST])

    def __pow__(self, float other) -> ConstExpr:
        return ConstExpr(self[CONST] ** other)

    def copy(self) -> ConstExpr:
        return ConstExpr(self[CONST])


cdef class FuncExpr(Expr):

    def __init__(self, children: Optional[dict[Union[Term, Expr, _ExprKey], float]] = None):
        if children and any((i is CONST) for i in children):
            raise ValueError("FuncExpr can't have Term without Variable as a child")

        super().__init__(children)

    cpdef float degree(self):
        return float("inf")

    def _is_child_equal(self, other) -> bool:
        return (
            type(other) is type(self)
            and len(self._children) == len(other._children)
            and self._children.keys() == other._children.keys()
        )


cdef class ProdExpr(FuncExpr):
    """Expression like `coefficient * expression`."""

    cdef readonly float coef
    __slots__ = ("coef",)

    def __init__(self, *children: Union[Term, Expr], float coef = 1.0):
        if len(set(children)) != len(children):
            raise ValueError("ProdExpr can't have duplicate children")

        super().__init__(dict.fromkeys(children, 1.0))
        self.coef = coef

    def __hash__(self) -> int:
        return (frozenset(self), self.coef).__hash__()

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


cdef class PowExpr(FuncExpr):
    """Expression like `pow(expression, exponent)`."""

    cdef readonly float expo
    __slots__ = ("expo",)

    def __init__(self, base: Union[Term, Expr, _ExprKey], float expo = 1.0):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self) -> int:
        return (frozenset(self), self.expo).__hash__()

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
                self = PolynomialExpr({self: 1.0})
        return self

    def copy(self) -> PowExpr:
        return PowExpr(self._fchild(), self.expo)


cdef class UnaryExpr(FuncExpr):
    """Expression like `f(expression)`."""

    def __init__(self, expr: Union[Number, Variable, Term, Expr, _ExprKey]):
        if isinstance(expr, Number):
            expr = ConstExpr(<float>expr)
        elif isinstance(expr, Variable):
            expr = Term(expr)
        super().__init__({expr: 1.0})

    def __hash__(self) -> int:
        return frozenset(self).__hash__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._fchild()})"

    def copy(self) -> UnaryExpr:
        return type(self)(self._fchild())


cdef class AbsExpr(UnaryExpr):
    """Expression like `abs(expression)`."""
    ...


cdef class ExpExpr(UnaryExpr):
    """Expression like `exp(expression)`."""
    ...


cdef class LogExpr(UnaryExpr):
    """Expression like `log(expression)`."""
    ...


cdef class SqrtExpr(UnaryExpr):
    """Expression like `sqrt(expression)`."""
    ...


cdef class SinExpr(UnaryExpr):
    """Expression like `sin(expression)`."""
    ...


cdef class CosExpr(UnaryExpr):
    """Expression like `cos(expression)`."""
    ...


cdef class ExprCons:
    """Constraints with a polynomial expressions and lower/upper bounds."""

    cdef readonly Expr expr
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

    def __richcmp__(self, float other, int op) -> ExprCons:
        if op == Py_LE:
            if self._rhs is not None:
                raise TypeError("ExprCons already has upper bound")
            return ExprCons(self.expr, lhs=<float>self._lhs, rhs=other)

        elif op == Py_GE:
            if self._lhs is not None:
                raise TypeError("ExprCons already has lower bound")
            return ExprCons(self.expr, lhs=other, rhs=<float>self._rhs)

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


cpdef Expr quicksum(expressions: Iterator[Expr]):
    """
    Use inplace addition to sum a list of expressions quickly, avoiding intermediate
    data structures created by Python's built-in sum function.

    Parameters
    ----------
    expressions : Iterator[Expr]
        An iterator of expressions to be summed.

    Returns
    -------
    Expr
        The sum of the input expressions.
    """
    cdef Expr res = ConstExpr(0.0)
    cdef Expr i
    for i in expressions:
        res += i
    return res


cpdef Expr quickprod(expressions: Iterator[Expr]):
    """
    Use inplace multiplication to multiply a list of expressions quickly, avoiding
    intermediate data structures created by Python's built-in prod function.

    Parameters
    ----------
    expressions : Iterator[Expr]
        An iterator of expressions to be multiplied.

    Returns
    -------
    Expr
        The product of the input expressions.
    """
    cdef Expr res = ConstExpr(1.0)
    cdef Expr i
    for i in expressions:
        res *= i
    return res




@cython.ufunc
cdef UnaryExpr _vec_to_unary(object x, type cls):
    return <UnaryExpr>cls(x)


@to_array(MatrixExpr)
def exp(
    x: Union[Number, Variable, Term, Expr, MatrixExpr],
) -> Union[ExpExpr, MatrixExpr]:
    """
    exp(x)

    Parameters
    ----------
    x : Number, Variable, Term, Expr, MatrixExpr

    Returns
    -------
    ExpExpr or MatrixExpr
    """
    return <ExpExpr>_vec_to_unary(x, ExpExpr)


@to_array(MatrixExpr)
def log(
    x: Union[Number, Variable, Term, Expr, MatrixExpr],
) -> Union[LogExpr, MatrixExpr]:
    """
    log(x)

    Parameters
    ----------
    x : Number, Variable, Term, Expr, MatrixExpr

    Returns
    -------
    LogExpr or MatrixExpr
    """
    return <LogExpr>_vec_to_unary(x, LogExpr)


@to_array(MatrixExpr)
def sqrt(
    x: Union[Number, Variable, Term, Expr, MatrixExpr],
) -> Union[SqrtExpr, MatrixExpr]:
    """
    sqrt(x)

    Parameters
    ----------
    x : Number, Variable, Term, Expr, MatrixExpr

    Returns
    -------
    SqrtExpr or MatrixExpr
    """
    return <SqrtExpr>_vec_to_unary(x, SqrtExpr)


@to_array(MatrixExpr)
def sin(
    x: Union[Number, Variable, Term, Expr, MatrixExpr],
) -> Union[SinExpr, MatrixExpr]:
    """
    sin(x)

    Parameters
    ----------
    x : Number, Variable, Term, Expr, MatrixExpr

    Returns
    -------
    SinExpr or MatrixExpr
    """
    return _vec_to_unary(x, SinExpr)


@to_array(MatrixExpr)
def cos(
    x: Union[Number, Variable, Term, Expr, MatrixExpr],
) -> Union[CosExpr, MatrixExpr]:
    """
    cos(x)

    Parameters
    ----------
    x : Number, Variable, Term, Expr, MatrixExpr

    Returns
    -------
    CosExpr or MatrixExpr
    """
    return _vec_to_unary(x, CosExpr)
