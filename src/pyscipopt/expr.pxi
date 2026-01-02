##@file expr.pxi
import operator
from numbers import Number
from typing import Iterator, Optional, Type, Union

import numpy as np

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

    def __eq__(self, other) -> bool:
        return isinstance(other, Term) and hash(self) == hash(other)

    def __mul__(self, Term other) -> Term:
        return Term(*self.vars, *other.vars)

    def __repr__(self) -> str:
        return f"Term({self[0]})" if self.degree() == 1 else f"Term{self.vars}"

    cpdef int degree(self):
        return len(self.vars)

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


cdef inline _wrap(x):
    return _ExprKey(x) if isinstance(x, Expr) else x


cdef inline _unwrap(x):
    return x.expr if isinstance(x, _ExprKey) else x


cdef class UnaryOperator:

    def __abs__(self) -> AbsExpr:
        return AbsExpr(self)

    def exp(self) -> ExpExpr:
        return ExpExpr(self)
    
    def log(self) -> LogExpr:
        return LogExpr(self)
    
    def sqrt(self) -> SqrtExpr:
        return SqrtExpr(self)

    def sin(self) -> SinExpr:
        return SinExpr(self)

    def cos(self) -> CosExpr:
        return CosExpr(self)


cdef inline Expr _copy(expr, cls: Type[Expr], bool copy = False):
    cdef Expr res = (
        ConstExpr.__new__(ConstExpr) if Expr._is_const(expr) else cls.__new__(cls)
    )
    (<Expr>res)._children = expr._children if not copy else expr._children.copy()
    if cls is ProdExpr:
        (<ProdExpr>res).coef = (<ProdExpr>expr).coef
    elif cls is PowExpr:
        (<PowExpr>res).expo = (<PowExpr>expr).expo
    return res


cdef class Expr(UnaryOperator):
    """Base class for mathematical expressions."""

    cdef readonly dict _children
    __slots__ = ("_children",)
    __array_priority__ = 100

    def __init__(
        self,
        children: Optional[dict[Union[Term, Expr, _ExprKey], float]] = None,
    ):
        if children and not all(isinstance(i, (Term, Expr, _ExprKey)) for i in children):
            raise TypeError("All keys must be Term or Expr instances")

        self._children = {_wrap(k): v for k, v in (children or {}).items()}

    @property
    def children(self):
        return {_unwrap(k): v for k, v in self.items()}

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__":
            return NotImplemented

        if (handler := UFUNC_DISPATCH.get(ufunc)) is not None:
            return handler(*args, **kwargs)
        return NotImplemented

    def __hash__(self) -> int:
        return frozenset(self.items()).__hash__()

    def __getitem__(self, key: Union[Variable, Term, Expr, _ExprKey]) -> float:
        if not isinstance(key, (Variable, Term, Expr, _ExprKey)):
            raise TypeError("key must be Variable, Term, or Expr")

        if isinstance(key, Variable):
            key = Term(key)
        return self._children.get(_wrap(key), 0.0)

    def __iter__(self) -> Iterator[Union[Term, Expr]]:
        for i in self._children:
            yield _unwrap(i)

    def __bool__(self) -> bool:
        return bool(self._children)

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_zero(self):
            return _other.copy()
        elif Expr._is_zero(_other):
            return self.copy()
        elif Expr._is_sum(self):
            return Expr(self._to_dict(_other))
        elif Expr._is_sum(_other):
            return Expr(_other._to_dict(self))
        elif self._is_equal(_other):
            return self * 2.0
        return Expr({_wrap(self): 1.0, _wrap(_other): 1.0})

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_zero(_other):
            return self
        elif Expr._is_sum(self) and Expr._is_sum(_other):
            self._to_dict(_other, copy=False)
            if isinstance(self, PolynomialExpr) and isinstance(_other, PolynomialExpr):
                return _copy(self, PolynomialExpr)
            return _copy(self, Expr)
        return self + _other

    def __radd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return self + other

    def __sub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if self._is_equal(_other):
            return ConstExpr(0.0)
        return self + (-_other)

    def __isub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if self._is_equal(_other):
            return ConstExpr(0.0)
        return self + (-_other)

    def __rsub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return (-self) + other

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_zero(self) or Expr._is_zero(_other):
            return ConstExpr(0.0)
        elif Expr._is_const(self):
            if self[CONST] == 1:
                return _other.copy()
            elif Expr._is_sum(_other):
                return Expr({k: v * self[CONST] for k, v in _other.items() if v != 0})
            return Expr({_other: self[CONST]})
        elif Expr._is_const(_other):
            if _other[CONST] == 1:
                return self.copy()
            elif Expr._is_sum(self):
                return Expr({k: v * _other[CONST] for k, v in self.items() if v != 0})
            return Expr({self: _other[CONST]})
        elif self._is_equal(_other):
            return PowExpr(self, 2.0)
        return ProdExpr(self, _other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if self and Expr._is_sum(self) and Expr._is_const(_other) and _other[CONST] != 0:
            self._children = {k: v * _other[CONST] for k, v in self.items() if v != 0}
            return _copy(
                self, PolynomialExpr if isinstance(self, PolynomialExpr) else Expr
            )
        return self * _other

    def __rmul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return self * other

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_zero(_other):
            raise ZeroDivisionError("division by zero")
        if self._is_equal(_other):
            return ConstExpr(1.0)
        return self * (_other ** ConstExpr(-1.0))

    def __rtruediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return Expr._from_other(other) / self

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if not Expr._is_const(_other):
            raise TypeError("exponent must be a number")
        return ConstExpr(1.0) if Expr._is_zero(_other) else PowExpr(self, _other[CONST])

    def __rpow__(self, other: Union[Number, Expr]) -> ExpExpr:
        cdef Expr _other = Expr._from_other(other)
        if _other[CONST] <= 0.0:
            raise ValueError("base must be positive")
        return ExpExpr(self * LogExpr(_other))

    def __neg__(self) -> Expr:
        return self * ConstExpr(-1.0)

    cdef ExprCons _cmp(self, other: Union[Number, Variable, Expr], int op):
        cdef Expr _other = Expr._from_other(other)
        if op == Py_LE:
            if Expr._is_const(_other):
                return ExprCons(self, rhs=_other[CONST])
            return ExprCons(self - _other, rhs=0.0)
        elif op == Py_GE:
            if Expr._is_const(_other):
                return ExprCons(self, lhs=_other[CONST])
            return ExprCons(self - _other, lhs=0.0)
        elif op == Py_EQ:
            if Expr._is_const(_other):
                return ExprCons(self, lhs=_other[CONST], rhs=_other[CONST])
            return ExprCons(self - _other, lhs=0.0, rhs=0.0)
        raise NotImplementedError("Expr can only support with '<=', '>=', or '=='.")

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    cdef Expr copy(self):
        return _copy(self, type(self), copy=True)

    def __repr__(self) -> str:
        return f"Expr({self._children})"

    def degree(self) -> float:
        return max((i.degree() for i in self)) if self else 0

    def items(self):
        return self._children.items()

    def _normalize(self) -> Expr:
        self._children = {k: v for k, v in self.items() if v != 0}
        return self

    @staticmethod
    cdef Expr _from_other(x: Union[Number, Variable, Expr]):
        """Convert a number or variable to an expression."""
        if isinstance(x, Number):
            return ConstExpr(<float>x)
        elif isinstance(x, Variable):
            return PolynomialExpr({Term(x): 1.0})
        elif isinstance(x, Expr):
            return x
        return NotImplemented

    cdef dict _to_dict(self, Expr other, bool copy = True):
        cdef dict children = self._children.copy() if copy else self._children
        cdef object child
        cdef float coef
        for child, coef in (other if Expr._is_sum(other) else {other: 1.0}).items():
            key = _wrap(child)
            children[key] = children.get(key, 0.0) + coef
        return children

    cpdef list[tuple] _to_node(self, float coef = 1, int start = 0):
        """Convert expression to list of node for SCIP expression construction"""
        cdef list[tuple] node = []
        cdef list[tuple] c_node
        cdef list[int] index = []
        cdef object k
        cdef float v

        if coef == 0:
            return node

        for k, v in self.items():
            if v != 0 and (c_node := _unwrap(k)._to_node(v, start + len(node))):
                node.extend(c_node)
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

    cdef _fchild(self):
        return next(iter(self._children))

    cdef bool _is_equal(self, object other):
        return (
            isinstance(other, Expr)
            and len(self._children) == len(other._children)
            and (
                (Expr._is_sum(self) and Expr._is_sum(other))
                or (
                    type(self) is type(other)
                    and (
                        (type(self) is ProdExpr and self.coef == (<ProdExpr>other).coef)
                        or (type(self) is PowExpr and self.expo == (<PowExpr>other).expo)
                        or isinstance(self, UnaryExpr)
                    )
                )
            )
            and self._children == other._children
        )

    @staticmethod
    cdef bool _is_sum(expr):
        return type(expr) is Expr or isinstance(expr, PolynomialExpr)

    @staticmethod
    cdef bool _is_const(expr):
        return isinstance(expr, ConstExpr) or (
            Expr._is_sum(expr)
            and len(expr._children) == 1
            and (<Expr>expr)._fchild() == CONST
        )

    @staticmethod
    cdef bool _is_zero(expr):
        return isinstance(expr, Expr) and (
            not expr or (Expr._is_const(expr) and expr[CONST] == 0)
        )

    @staticmethod
    cdef bool _is_term(expr):
        return (
            Expr._is_sum(expr)
            and len(expr._children) == 1
            and isinstance((<Expr>expr)._fchild(), Term)
            and (<Expr>expr)[(<Expr>expr)._fchild()] == 1
        )


cdef class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict[Term, float]] = None):
        if children and not all(isinstance(t, Term) for t in children):
            raise TypeError("All keys must be Term instances")

        super().__init__(<dict>children)

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if isinstance(_other, PolynomialExpr) and not Expr._is_zero(_other):
            return _copy(PolynomialExpr(self._to_dict(_other)), PolynomialExpr)
        return super().__add__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        cdef dict[Term, float] children
        cdef Term k1, k2, child
        cdef float v1, v2
        if self and isinstance(_other, PolynomialExpr) and other and not (
            Expr._is_const(_other) and (_other[CONST] == 0 or _other[CONST] == 1)
        ):
            children = {}
            for k1, v1 in self.items():
                for k2, v2 in _other.items():
                    child = k1 * k2
                    children[child] = children.get(child, 0.0) + v1 * v2
            return _copy(PolynomialExpr(children), PolynomialExpr)
        return super().__mul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_const(_other):
            return self.__mul__(1.0 / _other[CONST])
        return super().__truediv__(_other)

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_const(_other) and _other[CONST].is_integer() and _other[CONST] > 0:
            res = ConstExpr(1.0)
            for _ in range(int(_other[CONST])):
                res *= self
            return res
        return super().__pow__(_other)


cdef class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, float constant = 0.0):
        super().__init__({CONST: constant})

    def __abs__(self) -> ConstExpr:
        return ConstExpr(abs(self[CONST]))

    def __neg__(self) -> ConstExpr:
        return ConstExpr(-self[CONST])

    def __pow__(self, other: Union[Number, Expr]) -> ConstExpr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_const(_other):
            return ConstExpr(self[CONST] ** _other[CONST])
        return <ConstExpr>super().__pow__(_other)


cdef class FuncExpr(Expr):

    cpdef float degree(self):
        return float("inf")

    cdef bool _is_child_equal(self, other):
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

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if self._is_child_equal(_other):
            res = <ProdExpr>_copy(self, ProdExpr, copy=True)
            res.coef += (<ProdExpr>_other).coef
            return res
        return super().__add__(_other)

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if self._is_child_equal(_other):
            self.coef += (<ProdExpr>_other).coef
            return self
        return super().__iadd__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_const(_other) and _other[CONST] != 0 and _other[CONST] != 1:
            res = <ProdExpr>_copy(self, ProdExpr, copy=True)
            res.coef *= _other[CONST]
            return res
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if Expr._is_const(_other):
            if _other[CONST] == 0:
                self = ConstExpr(0.0)
            else:
                self.coef *= _other[CONST]
            return self
        return super().__imul__(_other)

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    def __repr__(self) -> str:
        return f"ProdExpr({{{tuple(self)}: {self.coef}}})"

    def _normalize(self) -> Union[ConstExpr, ProdExpr]:
        if self.coef == 0:
            self = ConstExpr(0.0)
        return self


cdef class PowExpr(FuncExpr):
    """Expression like `pow(expression, exponent)`."""

    cdef readonly float expo
    __slots__ = ("expo",)

    def __init__(self, base: Union[Term, Expr, _ExprKey], float expo = 1.0):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self) -> int:
        return (frozenset(self), self.expo).__hash__()

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if isinstance(_other, PowExpr) and self._is_child_equal(_other):
            return PowExpr(self._fchild(), self.expo + _other.expo)
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if isinstance(_other, PowExpr) and self._is_child_equal(_other):
            self.expo += _other.expo
            return self
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = Expr._from_other(other)
        if (
            isinstance(_other, PowExpr)
            and not self._is_equal(_other)
            and self._is_child_equal(_other)
        ):
            return PowExpr(self._fchild(), self.expo - _other.expo)
        return super().__truediv__(_other)

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    def __repr__(self) -> str:
        return f"PowExpr({self._fchild()}, {self.expo})"

    def _normalize(self) -> Expr:
        if self.expo == 0:
            self = ConstExpr(1.0)
        elif self.expo == 1:
            self = _unwrap(self._fchild())
            if isinstance(self, Term):
                self = PolynomialExpr({self: 1.0})
        return self


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

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    def __repr__(self) -> str:
        if Expr._is_const(child := _unwrap(self._fchild())):
            return f"{type(self).__name__}({child[CONST]})"
        elif Expr._is_term(child) and child[(term := (<Expr>child)._fchild())] == 1:
            return f"{type(self).__name__}({term})"
        return f"{type(self).__name__}({child})"


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


cdef dict UFUNC_DISPATCH = {
    np.add: operator.add,
    np.subtract: operator.sub,
    np.multiply: operator.mul,
    np.divide: operator.truediv,
    np.power: operator.pow,
    np.negative: operator.neg,
    np.less_equal: operator.le,
    np.greater_equal: operator.ge,
    np.equal: operator.eq,
    np.abs: AbsExpr,
    np.exp: ExpExpr,
    np.log: LogExpr,
    np.sqrt: SqrtExpr,
    np.sin: SinExpr,
    np.cos: CosExpr,
}


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
        self.expr = (self.expr - c)._normalize()
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
    cdef object i
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
    cdef object i
    for i in expressions:
        res *= i
    return res


cdef inline _ensure_unary_compatible(x):
    return ConstExpr(<float>x) if isinstance(x, Number) else x


def exp(
    x: Union[Number, Variable, Expr, np.ndarray, MatrixExpr],
) -> Union[ExpExpr, np.ndarray, MatrixExpr]:
    """
    exp(x)

    Parameters
    ----------
    x : Number, Variable, Expr, np.ndarray, MatrixExpr

    Returns
    -------
    ExpExpr, np.ndarray, MatrixExpr
    """
    return np.exp(_ensure_unary_compatible(x))


def log(
    x: Union[Number, Variable, Expr, np.ndarray, MatrixExpr],
) -> Union[LogExpr, np.ndarray, MatrixExpr]:
    """
    log(x)

    Parameters
    ----------
    x : Number, Variable, Expr, np.ndarray, MatrixExpr

    Returns
    -------
    LogExpr, np.ndarray, MatrixExpr
    """
    return np.log(_ensure_unary_compatible(x))


def sqrt(
    x: Union[Number, Variable, Expr, np.ndarray, MatrixExpr],
) -> Union[SqrtExpr, np.ndarray, MatrixExpr]:
    """
    sqrt(x)

    Parameters
    ----------
    x : Number, Variable, Expr, np.ndarray, MatrixExpr

    Returns
    -------
    SqrtExpr, np.ndarray, MatrixExpr
    """
    return np.sqrt(_ensure_unary_compatible(x))


def sin(
    x: Union[Number, Variable, Expr, np.ndarray, MatrixExpr],
) -> Union[SinExpr, np.ndarray, MatrixExpr]:
    """
    sin(x)

    Parameters
    ----------
    x : Number, Variable, Expr, np.ndarray, MatrixExpr

    Returns
    -------
    SinExpr, np.ndarray, MatrixExpr
    """
    return np.sin(_ensure_unary_compatible(x))


def cos(
    x: Union[Number, Variable, Expr, np.ndarray, MatrixExpr],
) -> Union[CosExpr, np.ndarray, MatrixExpr]:
    """
    cos(x)

    Parameters
    ----------
    x : Number, Variable, Expr, np.ndarray, MatrixExpr

    Returns
    -------
    CosExpr, np.ndarray, MatrixExpr
    """
    return np.cos(_ensure_unary_compatible(x))
