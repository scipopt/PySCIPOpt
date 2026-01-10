##@file expr.pxi
from numbers import Number
from typing import Iterator, Optional, Type, Union

import numpy as np

from cpython.object cimport Py_LE, Py_EQ, Py_GE
from pyscipopt.scip cimport Variable

include "matrix.pxi"


cdef class Term:
    """A monomial term consisting of one or more variables."""

    cdef readonly tuple vars
    cdef int _hash

    def __init__(self, *vars: Variable):
        if not all(isinstance(i, Variable) for i in vars):
            raise TypeError("All arguments must be Variable instances")

        self.vars = tuple(sorted(vars, key=hash))
        self._hash = hash(self.vars)

    @staticmethod
    cdef Term create(tuple[Variable] vars):
        cdef Term res = Term.__new__(Term)
        res.vars = tuple(sorted(vars, key=hash))
        res._hash = hash(res.vars)
        return res

    def __iter__(self) -> Iterator[Variable]:
        return iter(self.vars)

    def __getitem__(self, key: int) -> Variable:
        return self.vars[key]

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, Term) and hash(self) == hash(other)

    def __mul__(self, Term other) -> Term:
        return Term.create((*self.vars, *other.vars))

    def __repr__(self) -> str:
        return f"Term({self[0]})" if self.degree() == 1 else f"Term{self.vars}"

    def degree(self) -> int:
        return len(self.vars)

    cpdef list _to_node(self, float coef = 1, int start = 0):
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
        return isinstance(other, _ExprKey) and self.expr._is_equal(other.expr)

    def __repr__(self) -> str:
        return repr(self.expr)


CONST = Term()


cdef inline _wrap(x):
    return _ExprKey(x) if isinstance(x, Expr) else x


cdef inline _unwrap(x):
    return x.expr if isinstance(x, _ExprKey) else x


cdef class UnaryOperatorMixin:

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


cdef class Expr(UnaryOperatorMixin):
    """Base class for mathematical expressions."""

    cdef readonly dict _children
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
            return AbsExpr(*args, **kwargs)
        elif ufunc is np.exp:
            return ExpExpr(*args, **kwargs)
        elif ufunc is np.log:
            return LogExpr(*args, **kwargs)
        elif ufunc is np.sqrt:
            return SqrtExpr(*args, **kwargs)
        elif ufunc is np.sin:
            return SinExpr(*args, **kwargs)
        elif ufunc is np.cos:
            return CosExpr(*args, **kwargs)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

    def __getitem__(self, key: Union[Variable, Term, Expr, _ExprKey]) -> float:
        if not isinstance(key, (Variable, Term, Expr, _ExprKey)):
            raise TypeError("key must be Variable, Term, or Expr")

        if isinstance(key, Variable):
            key = Term.create((key,))
        return self._children.get(_wrap(key), 0.0)

    def __iter__(self) -> Iterator[Union[Term, Expr]]:
        for i in self._children:
            yield _unwrap(i)

    def __bool__(self) -> bool:
        return bool(self._children)

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if _is_zero(self):
            return _other.copy()
        elif _is_zero(_other):
            return self.copy()
        elif _is_sum(self):
            return Expr(self._to_dict(_other))
        elif _is_sum(_other):
            return Expr(_other._to_dict(self))
        elif self._is_equal(_other):
            return self * 2.0
        return Expr({_wrap(self): 1.0, _wrap(_other): 1.0})

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if _is_zero(_other):
            return self
        elif _is_sum(self) and _is_sum(_other):
            self._to_dict(_other, copy=False)
            if isinstance(self, PolynomialExpr) and isinstance(_other, PolynomialExpr):
                return self.copy(False, PolynomialExpr)
            return self.copy(False)
        return self + _other

    def __radd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return self + other

    def __sub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_equal(_other):
            return _const(0.0)
        return self + (-_other)

    def __isub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_equal(_other):
            return _const(0.0)
        return self + (-_other)

    def __rsub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return (-self) + other

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if _is_zero(self) or _is_zero(_other):
            return _const(0.0)
        elif Expr._is_const(self):
            if _c(self) == 1:
                return _other.copy()
            elif _is_sum(_other):
                return Expr({k: v * _c(self) for k, v in _other.items() if v != 0})
            return Expr({_other: _c(self)})
        elif Expr._is_const(_other):
            if _c(_other) == 1:
                return self.copy()
            elif _is_sum(self):
                return Expr({k: v * _c(_other) for k, v in self.items() if v != 0})
            return Expr({self: _c(_other)})
        elif self._is_equal(_other):
            return PowExpr(self, 2.0)
        return ProdExpr(self, _other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self and _is_sum(self) and Expr._is_const(_other) and _c(_other) != 0:
            self._children = {k: v * _c(_other) for k, v in self.items() if v != 0}
            return self.copy(False)
        return self * _other

    def __rmul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return self * other

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if _is_zero(_other):
            raise ZeroDivisionError("division by zero")
        if self._is_equal(_other):
            return _const(1.0)
        return self * (_other ** _const(-1.0))

    def __rtruediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        return _to_expr(other) / self

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if not Expr._is_const(_other):
            raise TypeError("exponent must be a number")
        return _const(1.0) if _is_zero(_other) else PowExpr(self, _c(_other))

    def __rpow__(self, other: Union[Number, Expr]) -> ExpExpr:
        cdef Expr _other = _to_expr(other)
        if _c(_other) <= 0.0:
            raise ValueError("base must be positive")
        return ExpExpr(self * LogExpr(_other))

    def __neg__(self) -> Expr:
        cdef Expr res = self.copy(False)
        res._children = {k: -v for k, v in self._children.items()}
        return res

    cdef ExprCons _cmp(self, other: Union[Number, Variable, Expr], int op):
        cdef Expr _other = _to_expr(other)
        if op == Py_LE:
            if Expr._is_const(_other):
                return ExprCons(self, rhs=_c(_other))
            return ExprCons(self - _other, rhs=0.0)
        elif op == Py_GE:
            if Expr._is_const(_other):
                return ExprCons(self, lhs=_c(_other))
            return ExprCons(self - _other, lhs=0.0)
        elif op == Py_EQ:
            if Expr._is_const(_other):
                return ExprCons(self, lhs=_c(_other), rhs=_c(_other))
            return ExprCons(self - _other, lhs=0.0, rhs=0.0)

        raise NotImplementedError("Expr can only support with '<=', '>=', or '=='.")

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

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
    cdef PolynomialExpr _from_var(Variable x):
        return PolynomialExpr.create({Term.create((x,)): 1.0})

    cdef dict _to_dict(self, Expr other, bool copy = True):
        cdef dict children = self._children.copy() if copy else self._children
        cdef object child
        cdef float coef
        for child, coef in (other if _is_sum(other) else {other: 1.0}).items():
            key = _wrap(child)
            children[key] = children.get(key, 0.0) + coef
        return children

    cpdef list _to_node(self, float coef = 1, int start = 0):
        cdef list node = []
        cdef list sub_node
        cdef list[int] index = []
        cdef object k
        cdef float v

        if coef == 0:
            return node

        for k, v in self.items():
            if v != 0 and (sub_node := _unwrap(k)._to_node(v * coef, start + len(node))):
                node.extend(sub_node)
                index.append(start + len(node) - 1)

        if len(node) > 1:
            node.append((Expr, index))
        return node

    cdef bool _is_equal(self, object other):
        return (
            isinstance(other, Expr)
            and len(self._children) == len(other._children)
            and (
                (_is_sum(self) and _is_sum(other))
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
    cdef bool _is_const(expr):
        return isinstance(expr, ConstExpr) or (
            _is_sum(expr)
            and len(expr._children) == 1
            and _fchild(<Expr>expr) == CONST
        )

    @staticmethod
    cdef bool _is_term(expr):
        return (
            _is_sum(expr)
            and len(expr._children) == 1
            and isinstance(_fchild(<Expr>expr), Term)
            and (<Expr>expr)[_fchild(<Expr>expr)] == 1
        )

    cdef Expr copy(self, bool copy = True, cls: Optional[Type[Expr]] = None):
        cls = ConstExpr if Expr._is_const(self) else (cls or type(self))
        cdef Expr res = cls.__new__(cls)
        res._children = self._children.copy() if copy else self._children
        if cls is ProdExpr:
            (<ProdExpr>res).coef = (<ProdExpr>self).coef
        elif cls is PowExpr:
            (<PowExpr>res).expo = (<PowExpr>self).expo
        return res


cdef inline float _c(Expr expr):
    return expr._children.get(CONST, 0.0)


cdef inline Expr _to_expr(x: Union[Number, Variable, Expr]):
    if isinstance(x, Number):
        return _const(<float>x)
    elif isinstance(x, Variable):
        return Expr._from_var(x)
    elif isinstance(x, Expr):
        return x
    return NotImplemented


cdef inline bool _is_sum(expr):
    return type(expr) is Expr or isinstance(expr, PolynomialExpr)


cdef inline bool _is_zero(Expr expr):
    return not expr or (Expr._is_const(expr) and _c(expr) == 0)


cdef inline _fchild(Expr expr):
    return next(iter(expr._children))


cdef class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __init__(self, children: Optional[dict[Term, float]] = None):
        if children and not all(isinstance(t, Term) for t in children):
            raise TypeError("All keys must be Term instances")

        super().__init__(<dict>children)

    @staticmethod
    cdef PolynomialExpr create(dict[Term, float] children):
        cdef PolynomialExpr res = PolynomialExpr.__new__(PolynomialExpr)
        res._children = children
        return res

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if isinstance(_other, PolynomialExpr) and not _is_zero(_other):
            return PolynomialExpr.create(self._to_dict(_other)).copy(False)
        return super().__add__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        cdef PolynomialExpr res
        cdef Term k1, k2, child
        cdef float v1, v2
        if self and isinstance(_other, PolynomialExpr) and other and not (
            Expr._is_const(_other) and (_c(_other) == 0 or _c(_other) == 1)
        ):
            res = PolynomialExpr.create({})
            for k1, v1 in self.items():
                for k2, v2 in _other.items():
                    child = k1 * k2
                    res._children[child] = res._children.get(child, 0.0) + v1 * v2
            return res
        return super().__mul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if Expr._is_const(_other):
            return self * (1.0 / _c(_other))
        return super().__truediv__(_other)

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if Expr._is_const(_other) and _c(_other).is_integer() and _c(_other) > 0:
            res = _const(1.0)
            for _ in range(int(_c(_other))):
                res *= self
            return res
        return super().__pow__(_other)


cdef class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __init__(self, float constant = 0.0):
        super().__init__({CONST: constant})

    def __abs__(self) -> ConstExpr:
        return _const(abs(_c(self)))

    def __neg__(self) -> ConstExpr:
        return _const(-_c(self))

    def __pow__(self, other: Union[Number, Expr]) -> ConstExpr:
        cdef Expr _other = _to_expr(other)
        if Expr._is_const(_other):
            return _const(_c(self) ** _c(_other))
        return <ConstExpr>super().__pow__(_other)

    cpdef list _to_node(self, float coef = 1, int start = 0):
        cdef float res = _c(self) * coef
        return [(ConstExpr, res)] if res != 0 else []


cdef inline ConstExpr _const(float c):
    cdef ConstExpr res = ConstExpr.__new__(ConstExpr)
    res._children = {CONST: c}
    return res


cdef class FuncExpr(Expr):

    def __neg__(self):
        return self * _const(-1.0)

    def degree(self) -> float:
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

    def __init__(self, *children: Union[Term, Expr, _ExprKey]):
        if len(children) < 2:
            raise ValueError("ProdExpr must have at least two children")
        if len(set(children)) != len(children):
            raise ValueError("ProdExpr can't have duplicate children")

        super().__init__(dict.fromkeys(children, 1.0))
        self.coef = 1.0

    def __hash__(self) -> int:
        return hash((frozenset(self), self.coef))

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_child_equal(_other):
            res = self.copy()
            (<ProdExpr>res).coef += (<ProdExpr>_other).coef
            return res._normalize()
        return super().__add__(_other)

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_child_equal(_other):
            self.coef += (<ProdExpr>_other).coef
            return self._normalize()
        return super().__iadd__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if Expr._is_const(_other):
            res = self.copy()
            (<ProdExpr>res).coef *= _c(_other)
            return res._normalize()
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if Expr._is_const(_other):
            self.coef *= _c(_other)
            return self._normalize()
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if Expr._is_const(_other):
            res = self.copy()
            (<ProdExpr>res).coef /= _c(_other)
            return res._normalize()
        return super().__truediv__(_other)

    def __neg__(self) -> ProdExpr:
        cdef ProdExpr res = <ProdExpr>self.copy()
        res.coef = -self.coef
        return res

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    def __repr__(self) -> str:
        return f"ProdExpr({{{tuple(self)}: {self.coef}}})"

    def _normalize(self) -> Expr:
        return _const(0.0) if not self or self.coef == 0 else self

    cpdef list _to_node(self, float coef = 1, int start = 0):
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

    cdef readonly float expo

    def __init__(self, base: Union[Term, Expr, _ExprKey], float expo = 1.0):
        super().__init__({base: 1.0})
        self.expo = expo

    def __hash__(self) -> int:
        return hash((frozenset(self), self.expo))

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_child_equal(_other):
            res = self.copy()
            (<PowExpr>res).expo += (<PowExpr>_other).expo
            return res._normalize()
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_child_equal(_other):
            self.expo += (<PowExpr>_other).expo
            return self._normalize()
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if self._is_child_equal(_other):
            res = self.copy()
            (<PowExpr>res).expo -= (<PowExpr>_other).expo
            return res._normalize()
        return super().__truediv__(_other)

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    def __repr__(self) -> str:
        return f"PowExpr({_fchild(self)}, {self.expo})"

    def _normalize(self) -> Expr:
        if not self or self.expo == 0:
            return _const(1.0)
        elif self.expo == 1:
            return (
                PolynomialExpr.create({_fchild(self): 1.0})
                if isinstance(_fchild(self), Term) else _unwrap(_fchild(self))
            )
        return self

    cpdef list _to_node(self, float coef = 1, int start = 0):
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
        if isinstance(expr, Number):
            expr = _const(<float>expr)
        elif isinstance(expr, Variable):
            expr = Term.create((expr,))
        super().__init__({expr: 1.0})

    def __hash__(self) -> int:
        return hash(frozenset(self))

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op) -> ExprCons:
        return self._cmp(other, op)

    def __repr__(self) -> str:
        if Expr._is_const(child := _unwrap(_fchild(self))):
            return f"{type(self).__name__}({_c(child)})"
        elif Expr._is_term(child) and child[(term := _fchild(<Expr>child))] == 1:
            return f"{type(self).__name__}({term})"
        return f"{type(self).__name__}({child})"

    cpdef list _to_node(self, float coef = 1, int start = 0):
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
        c = _c(self.expr)
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


cdef inline _ensure_unary_compatible(x):
    return _const(<float>x) if isinstance(x, Number) else x


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
