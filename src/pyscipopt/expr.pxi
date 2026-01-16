##@file expr.pxi
from numbers import Number
from typing import TYPE_CHECKING, Iterator, Optional, Type, Union

import numpy as np

from cpython.object cimport Py_LE, Py_EQ, Py_GE
from pyscipopt.scip cimport Variable


if TYPE_CHECKING:
    double = float


cdef class Term:
    """A monomial term consisting of one or more variables."""

    cdef readonly tuple vars
    cdef int _hash

    def __init__(self, *vars: Variable):
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

    def __eq__(self, other: Term) -> bool:
        if self is other:
            return True
        if not isinstance(other, Term):
            return False

        cdef Term _other = <Term>other
        return False if self._hash != _other._hash else self.vars == _other.vars

    def __mul__(self, Term other) -> Term:
        return Term(*(self.vars + other.vars))

    def __repr__(self) -> str:
        return f"Term({self[0]})" if self.degree() == 1 else f"Term{self.vars}"

    def degree(self) -> int:
        return len(self)

    cpdef list _to_node(self, double coef = 1, int start = 0):
        cdef list node = []
        if coef == 0:
            ...
        elif self.degree() == 0:
            node.append((ConstExpr, coef))
        else:
            node.extend([(Variable, i) for i in self])
            if coef != 1:
                node.append((ConstExpr, coef))
            if len(node) > 1:
                node.append((ProdExpr, list(range(start, start + len(node)))))
        return node


cdef class _ExprKey:

    cdef readonly Expr expr

    def __init__(self, Expr expr):
        self.expr = expr

    def __hash__(self) -> int:
        return hash(self.expr)

    def __eq__(self, other) -> bool:
        return isinstance(other, _ExprKey) and _is_expr_equal(self.expr, other.expr)

    def __repr__(self) -> str:
        return repr(self.expr)


cdef class UnaryOperatorMixin:

    def __abs__(self) -> AbsExpr:
        return <AbsExpr>_unary(_ensure_unary(self), AbsExpr)

    def exp(self) -> ExpExpr:
        return <ExpExpr>_unary(_ensure_unary(self), ExpExpr)
    
    def log(self) -> LogExpr:
        return <LogExpr>_unary(_ensure_unary(self), LogExpr)
    
    def sqrt(self) -> SqrtExpr:
        return <SqrtExpr>_unary(_ensure_unary(self), SqrtExpr)

    def sin(self) -> SinExpr:
        return <SinExpr>_unary(_ensure_unary(self), SinExpr)

    def cos(self) -> CosExpr:
        return <CosExpr>_unary(_ensure_unary(self), CosExpr)


cdef class Expr(UnaryOperatorMixin):
    """Base class for mathematical expressions."""

    __array_priority__ = 100

    def __cinit__(self, *args, **kwargs):
        self.coef = 1.0
        self.expo = 1.0
        self._hash = -1

    def __init__(
        self,
        children: Optional[dict[Union[Term, Expr, _ExprKey], double]] = None,
    ):
        for i in (children or {}):
            if not isinstance(i, (Term, Expr, _ExprKey)):
                raise TypeError(
                    f"expected Term, Expr, or _ExprKey, but got {type(i).__name__!s}"
                )

        self._children = {_wrap(k): v for k, v in (children or {}).items()}

    @property
    def children(self):
        return {_unwrap(k): v for k, v in self.items()}

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__":
            return NotImplemented

        for arg in args:
            if not isinstance(arg, (Number, Variable, Expr)):
                return NotImplemented

        if ufunc is np.add:
            return args[0] + args[1]
        elif ufunc is np.subtract:
            return args[0] - args[1]
        elif ufunc is np.multiply:
            return args[0] * args[1]
        elif ufunc is np.true_divide:
            return args[0] / args[1]
        elif ufunc is np.power:
            return args[0] ** args[1]
        elif ufunc is np.negative:
            return -args[0]
        elif ufunc is np.less_equal:
            return args[0] <= args[1]
        elif ufunc is np.greater_equal:
            return args[0] >= args[1]
        elif ufunc is np.equal:
            return args[0] == args[1]
        elif ufunc is np.absolute:
            return <AbsExpr>_unary(_ensure_unary(args[0]), AbsExpr)
        elif ufunc is np.exp:
            return <ExpExpr>_unary(_ensure_unary(args[0]), ExpExpr)
        elif ufunc is np.log:
            return <LogExpr>_unary(_ensure_unary(args[0]), LogExpr)
        elif ufunc is np.sqrt:
            return <SqrtExpr>_unary(_ensure_unary(args[0]), SqrtExpr)
        elif ufunc is np.sin:
            return <SinExpr>_unary(_ensure_unary(args[0]), SinExpr)
        elif ufunc is np.cos:
            return <CosExpr>_unary(_ensure_unary(args[0]), CosExpr)
        return NotImplemented

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash(frozenset(self.items())))
        return self._hash

    def __getitem__(self, key: Union[Variable, Term, Expr, _ExprKey]) -> double:
        if not isinstance(key, (Variable, Term, Expr, _ExprKey)):
            raise TypeError(
                f"excepted Variable, Term, or Expr, but got {type(key).__name__!s}"
            )

        if isinstance(key, Variable):
            key = _term((key,))
        return self._children.get(_wrap(key), 0.0)

    def __iter__(self) -> Iterator[Union[Term, Expr]]:
        for i in self._children:
            yield _unwrap(i)

    def __bool__(self) -> bool:
        return bool(self._children)

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self):
            return _other.copy()
        elif _is_zero(_other):
            return self.copy()
        elif _is_sum(self):
            return _expr(self._to_dict(_other))
        elif _is_sum(_other):
            return _expr(_other._to_dict(self))
        elif _is_expr_equal(self, _other):
            return self * 2.0
        return _expr({_wrap(self): 1.0, _wrap(_other): 1.0})

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self):
            return _other
        elif _is_zero(_other):
            return self
        elif _is_sum(self) and _is_sum(_other):
            self._to_dict(_other, copy=False)
            self._hash = -1
            if isinstance(self, PolynomialExpr) and isinstance(_other, PolynomialExpr):
                return self.copy(False, PolynomialExpr)
            return self.copy(False)
        return self + _other

    def __radd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return self + other

    def __sub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_expr_equal(self, _other):
            return _const(0.0)
        return self + (-_other)

    def __isub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        return _const(0.0) if _is_expr_equal(self, _other) else self + (-_other)

    def __rsub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return (-self) + other

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self) or _is_zero(_other):
            return _const(0.0)
        elif _is_const(self):
            if _c(self) == 1:
                return _other.copy()
            elif _is_sum(_other):
                return _expr({k: v * _c(self) for k, v in _other.items() if v != 0})
            return _expr({_wrap(_other): _c(self)})
        elif _is_const(_other):
            if _c(_other) == 1:
                return self.copy()
            elif _is_sum(self):
                return _expr({k: v * _c(_other) for k, v in self.items() if v != 0})
            return _expr({_wrap(self): _c(_other)})
        elif _is_expr_equal(self, _other):
            return _pow(_wrap(self), 2.0)
        return _prod((_wrap(self), _wrap(_other)))

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_sum(self) and _is_const(_other) and _c(_other) != 0:
            self._children = {k: v * _c(_other) for k, v in self.items() if v != 0}
            self._hash = -1
            return self.copy(False)
        return self * _other

    def __rmul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return self * other

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self):
            return _const(0.0)
        elif _is_zero(_other):
            raise ZeroDivisionError("division by zero")
        elif _is_expr_equal(self, _other):
            return _const(1.0)
        return self * (_other ** _const(-1.0))

    def __rtruediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        return _to_expr(other) / self

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if not _is_const(_other):
            raise TypeError("excepted a constant exponent")
        if _is_zero(self):
            return _const(0.0)
        elif _is_zero(_other):
            return _const(1.0)
        return _pow(_wrap(self), _c(_other))

    def __rpow__(self, other: Union[Number, Expr]) -> Union[ExpExpr, ConstExpr]:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _c(_other) <= 0.0:
            raise ValueError("excepted a positive base")
        return _const(1.0) if _is_zero(self) else ExpExpr(self * LogExpr(_other))

    def __neg__(self) -> Expr:
        cdef Expr res = self.copy(False)
        res._children = {k: -v for k, v in self._children.items()}
        return res

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op):
        return _expr_cmp(self, other, op)

    def __repr__(self) -> str:
        return f"Expr({self._children})"

    def degree(self) -> double:
        return max((i.degree() for i in self)) if self else 0

    def keys(self):
        return self._children.keys()

    def items(self):
        return self._children.items()

    def _normalize(self) -> Expr:
        self._children = {k: v for k, v in self.items() if v != 0}
        self._hash = -1
        return self

    cdef dict _to_dict(self, Expr other, bool copy = True):
        cdef dict children = self._children.copy() if copy else self._children
        cdef object k
        cdef double v
        for k, v in (other if _is_sum(other) else {_wrap(other): 1.0}).items():
            children[k] = children.get(k, 0.0) + v
        return children

    cpdef list _to_node(self, double coef = 1, int start = 0):
        cdef list node = []
        cdef list sub_node
        cdef list[int] index = []
        cdef object k
        cdef double v

        if coef == 0:
            return node

        for k, v in self.items():
            if v != 0 and (sub_node := _unwrap(k)._to_node(v * coef, start + len(node))):
                node.extend(sub_node)
                index.append(start + len(node) - 1)

        if len(node) > 1:
            node.append((Expr, index))
        return node

    cdef Expr copy(self, bool copy = True, cls: Optional[Type[Expr]] = None):
        cls = cls or type(self)
        cdef Expr res = <Expr>cls.__new__(cls)
        res._children = self._children.copy() if copy else self._children
        if cls is ProdExpr:
            res.coef = self.coef
        elif cls is PowExpr:
            res.expo = self.expo
        return res


cdef class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict[Term, double]] = None):
        for i in (children or {}):
            if not isinstance(i, Term):
                raise TypeError(f"expected Term, but got {type(i).__name__!s}")

        super().__init__(children)

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and isinstance(_other, PolynomialExpr) and not _is_zero(_other):
            return _expr(self._to_dict(_other), PolynomialExpr)
        return super().__add__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        cdef PolynomialExpr res
        cdef Term k1, k2, child
        cdef double v1, v2
        if self and isinstance(_other, PolynomialExpr) and other and not (
            _is_const(_other) and (_c(_other) == 0 or _c(_other) == 1)
        ):
            res = <PolynomialExpr>_expr({}, PolynomialExpr)
            for k1, v1 in self.items():
                for k2, v2 in _other.items():
                    child = k1 * k2
                    res._children[child] = res._children.get(child, 0.0) + v1 * v2
            return res
        return super().__mul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_const(_other):
            return self * (1.0 / _c(_other))
        return super().__truediv__(_other)

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_const(_other) and _c(_other).is_integer() and _c(_other) > 0:
            res = _const(1.0)
            for _ in range(int(_c(_other))):
                res *= self
            return res
        return super().__pow__(_other)


cdef class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, double constant = 0.0):
        super().__init__({CONST: constant})

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_const(_other):
            return _const(_c(self) + _c(_other))
        return super().__add__(_other)

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_const(_other):
            self._children[CONST] += _c(_other)
            self._hash = -1
            return self
        return super().__iadd__(_other)

    def __sub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_const(_other):
            return _const(_c(self) - _c(_other))
        return super().__sub__(_other)

    def __isub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_const(_other):
            self._children[CONST] -= _c(_other)
            self._hash = -1
            return self
        return super().__isub__(_other)

    def __pow__(self, other: Union[Number, Expr]) -> ConstExpr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_const(_other):
            return _const(_c(self) ** _c(_other))
        return <ConstExpr>super().__pow__(_other)

    def __neg__(self) -> ConstExpr:
        return _const(-_c(self))

    def __abs__(self) -> ConstExpr:
        return _const(abs(_c(self)))

    cpdef list _to_node(self, double coef = 1, int start = 0):
        cdef double res = _c(self) * coef
        return [(ConstExpr, res)] if res != 0 else []


cdef class FuncExpr(Expr):

    def __neg__(self):
        return self * _const(-1.0)

    def degree(self) -> double:
        return INF


cdef class ProdExpr(FuncExpr):
    """Expression like `coefficient * expression`."""

    def __init__(self, *children: Union[Term, Expr, _ExprKey]):
        if len(children) < 2:
            raise ValueError("ProdExpr must have at least two children")
        if len(set(children)) != len(children):
            raise ValueError("ProdExpr can't have duplicate children")

        super().__init__(dict.fromkeys(children, 1.0))

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash((frozenset(self.keys()), self.coef)))
        return self._hash

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            res = self.copy()
            res.coef += _other.coef
            return res._normalize()
        return super().__add__(_other)

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            self.coef += _other.coef
            self._hash = -1
            return self._normalize()
        return super().__iadd__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_const(_other):
            res = self.copy()
            res.coef *= _c(_other)
            return res._normalize()
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_const(_other):
            self.coef *= _c(_other)
            self._hash = -1
            return self._normalize()
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_const(_other):
            res = self.copy()
            res.coef /= _c(_other)
            return res._normalize()
        return super().__truediv__(_other)

    def __neg__(self) -> ProdExpr:
        cdef ProdExpr res = <ProdExpr>self.copy()
        res.coef = -self.coef
        return res

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op):
        return _expr_cmp(self, other, op)

    def __repr__(self) -> str:
        return f"ProdExpr({{{tuple(self)}: {self.coef}}})"

    def _normalize(self) -> Expr:
        return _const(0.0) if not self or self.coef == 0 else self

    cpdef list _to_node(self, double coef = 1, int start = 0):
        cdef list node = []
        cdef list sub_node
        cdef list[int] index = []
        cdef object i

        if coef == 0:
            return node

        for i in self:
            if (sub_node := i._to_node(1, start + len(node))):
                node.extend(sub_node)
                index.append(start + len(node) - 1)

        if self.coef * coef != 1:
            node.append((ConstExpr, self.coef * coef))
            index.append(start + len(node) - 1)
        if len(node) > 1:
            node.append((ProdExpr, index))
        return node


cdef class PowExpr(FuncExpr):
    """Expression like `pow(expression, exponent)`."""

    def __init__(self, base: Union[Term, Expr, _ExprKey], double expo):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash((frozenset(self.keys()), self.expo)))
        return self._hash

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            res = self.copy()
            res.expo += _other.expo
            return res._normalize()
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            self.expo += _other.expo
            self._hash = -1
            return self._normalize()
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, (Number, Variable, Expr)):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            res = self.copy()
            res.expo -= _other.expo
            return res._normalize()
        return super().__truediv__(_other)

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op):
        return _expr_cmp(self, other, op)

    def __repr__(self) -> str:
        return f"PowExpr({_fchild(self)}, {self.expo})"

    def _normalize(self) -> Expr:
        if not self or self.expo == 0:
            return _const(1.0)
        elif self.expo == 1:
            return (
                <PolynomialExpr>_expr({_fchild(self): 1.0}, PolynomialExpr)
                if isinstance(_fchild(self), Term) else <Expr>_unwrap(_fchild(self))
            )
        return self

    cpdef list _to_node(self, double coef = 1, int start = 0):
        if coef == 0:
            return []

        cdef list node = _unwrap(_fchild(self))._to_node(1, start)
        node.append((ConstExpr, self.expo))
        node.append((PowExpr, [start + len(node) - 2, start + len(node) - 1]))
        if coef != 1:
            node.append((ConstExpr, coef))
            node.append((ProdExpr, [start + len(node) - 2, start + len(node) - 1]))
        return node


cdef class UnaryExpr(FuncExpr):
    """Expression like `f(expression)`."""

    def __init__(self, expr: Union[Number, Variable, Term, Expr, _ExprKey]):
        super().__init__({_ensure_unary(expr): 1.0})

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash(_fchild(self)))
        return self._hash

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op):
        return _expr_cmp(self, other, op)

    def __repr__(self) -> str:
        name = type(self).__name__
        if _is_const(child := _unwrap(_fchild(self))):
            return f"{name}({_c(<Expr>child)})"
        elif _is_term(child) and (<Expr>child)[(term := _fchild(<Expr>child))] == 1:
            return f"{name}({term})"
        return f"{name}({child})"

    cpdef list _to_node(self, double coef = 1, int start = 0):
        if coef == 0:
            return []

        cdef list node = _unwrap(_fchild(self))._to_node(1, start)
        node.append((type(self), start + len(node) - 1))
        if coef != 1:
            node.append((ConstExpr, coef))
            node.append((ProdExpr, [start + len(node) - 2, start + len(node) - 1]))
        return node


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

    def __init__(
        self,
        Expr expr,
        lhs: Optional[double] = None,
        rhs: Optional[double] = None,
    ):
        if lhs is None and rhs is None:
            raise ValueError("ExprCons (with both lhs and rhs) doesn't supported")

        self.expr = expr
        self._lhs = lhs
        self._rhs = rhs
        self._normalize()

    def _normalize(self) -> ExprCons:
        """Move constant children in expression to bounds"""
        c = _c(self.expr)
        self.expr = (self.expr - c)._normalize()
        if self._lhs is not None:
            self._lhs = <double>self._lhs - c
        if self._rhs is not None:
            self._rhs = <double>self._rhs - c
        return self

    def __richcmp__(self, double other, int op) -> ExprCons:
        if op == Py_LE:
            if self._rhs is not None:
                raise TypeError("ExprCons already has upper bound")
            return ExprCons(self.expr, lhs=<double>self._lhs, rhs=other)
        elif op == Py_GE:
            if self._lhs is not None:
                raise TypeError("ExprCons already has lower bound")
            return ExprCons(self.expr, lhs=other, rhs=<double>self._rhs)

        raise NotImplementedError("can only support with '<=' or '>='")

    def __repr__(self) -> str:
        return f"ExprCons({self.expr}, {self._lhs}, {self._rhs})"

    def __bool__(self):
        """Make sure that equality of expressions is not asserted with =="""

        msg = """can't evaluate constraints as booleans.

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
    cdef Expr res = _const(0.0)
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
    cdef Expr res = _const(1.0)
    cdef object i
    for i in expressions:
        res *= i
    return res


cdef inline int _ensure_hash(int h) noexcept:
    return -2 if h == -1 else h


cdef inline Term _term(tuple vars):
    cdef Term res = Term.__new__(Term)
    res.vars = tuple(sorted(vars, key=hash))
    res._hash = hash(res.vars)
    return res


cdef double INF = float("inf")
CONST = Term()


cdef inline double _c(Expr expr):
    return expr._children.get(CONST, 0.0)


cdef inline ConstExpr _const(double c):
    cdef ConstExpr res = ConstExpr.__new__(ConstExpr)
    res._children = {CONST: c}
    return res


_vec_const = np.frompyfunc(_const, 1, 1)


cdef inline Expr _expr(dict children, cls: Type[Expr] = Expr):
    cdef Expr res = <Expr>cls.__new__(cls)
    res._children = children
    return res


cdef inline ProdExpr _prod(tuple children):
    cdef ProdExpr res = ProdExpr.__new__(ProdExpr)
    res._children = dict.fromkeys(children, 1.0)
    return res


cdef inline PowExpr _pow(base: Union[Term, _ExprKey], double expo):
    cdef PowExpr res = PowExpr.__new__(PowExpr)
    res._children = {base: 1.0} 
    res.expo = expo
    return res


cdef inline _wrap(x):
    return _ExprKey(x) if isinstance(x, Expr) else x


cdef inline _unwrap(x):
    return x.expr if isinstance(x, _ExprKey) else x


cdef Expr _to_expr(x: Union[Number, Variable, Expr]):
    if isinstance(x, Number):
        return _const(<double>x)
    elif isinstance(x, Variable):
        return _var_to_expr(x)
    elif isinstance(x, Expr):
        return x
    raise TypeError(f"expected Number, Variable, or Expr, but got {type(x).__name__!s}")


cdef inline PolynomialExpr _var_to_expr(Variable x):
    return <PolynomialExpr>_expr({_term((x,)): 1.0}, PolynomialExpr)


cdef object _expr_cmp(Expr self, other: Union[Number, Variable, Expr], int op):
    if isinstance(other, np.ndarray):
        return NotImplemented
    cdef Expr _other = _to_expr(other)
    if op == Py_LE:
        if _is_const(_other):
            return ExprCons(self, rhs=_c(_other))
        return ExprCons(self - _other, rhs=0.0)
    elif op == Py_GE:
        if _is_const(_other):
            return ExprCons(self, lhs=_c(_other))
        return ExprCons(self - _other, lhs=0.0)
    elif op == Py_EQ:
        if _is_const(_other):
            return ExprCons(self, lhs=_c(_other), rhs=_c(_other))
        return ExprCons(self - _other, lhs=0.0, rhs=0.0)

    raise NotImplementedError("can only support with '<=', '>=', or '=='")


cdef inline bool _is_sum(expr):
    return type(expr) is Expr or isinstance(expr, PolynomialExpr)


cdef inline bool _is_const(expr):
    return _is_sum(expr) and len(expr._children) == 1 and _fchild(<Expr>expr) == CONST


cdef inline bool _is_zero(Expr expr):
    return not expr or (_is_const(expr) and _c(expr) == 0)


cdef bool _is_term(expr):
    return (
        _is_sum(expr)
        and len(expr._children) == 1
        and type(_fchild(<Expr>expr)) is Term
        and expr._children[_fchild(<Expr>expr)] == 1
    )


cdef inline _fchild(Expr expr):
    return next(iter(expr._children))


cdef bool _is_expr_equal(Expr x, object y):
    if x is y:
        return True
    if not isinstance(y, Expr):
        return False

    cdef Expr _y = <Expr>y
    if len(x._children) != len(_y._children) or x._hash != _y._hash:
        return False

    cdef object t_x = type(x)
    if _is_sum(x):
        if not _is_sum(_y):
            return False
    else:
        if t_x is not type(_y):
            return False

        if t_x is ProdExpr:
            if x.coef != _y.coef:
                return False
        elif t_x is PowExpr:
            if x.expo != _y.expo:
                return False
    return x._children == _y._children


cdef bool _is_child_equal(Expr x, object y):
    if x is y:
        return True
    if type(y) is not type(x):
        return False

    cdef Expr _y = <Expr>y
    if len(x._children) != len(_y._children):
        return False
    return x.keys() == _y.keys()


cdef _ensure_unary(x: Union[Number, Variable, Term, Expr, _ExprKey]):
    if isinstance(x, Number):
        return _ExprKey(_const(<double>x))
    elif isinstance(x, Variable):
        return _term((x,))
    elif isinstance(x, Expr):
        return _ExprKey(x)
    elif isinstance(x, (Term, _ExprKey)):
        return x
    raise TypeError(
        f"expected Number, Variable, _ExprKey, or Expr, but got {type(x).__name__!s}"
    )


cdef inline UnaryExpr _unary(x: Union[Term, _ExprKey], cls: Type[UnaryExpr]):
    cdef UnaryExpr res = <UnaryExpr>cls.__new__(cls)
    res._children = {x: 1.0}
    return res


cdef inline _ensure_const(x):
    if isinstance(x, Number):
        _const(<double>x)
    elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
        return _vec_const(x).view(MatrixExpr)
    return  x


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
    return np.exp(_ensure_const(x))


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
    return np.log(_ensure_const(x))


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
    return np.sqrt(_ensure_const(x))


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
    return np.sin(_ensure_const(x))


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
    return np.cos(_ensure_const(x))
