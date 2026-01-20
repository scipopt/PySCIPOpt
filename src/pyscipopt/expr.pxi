##@file expr.pxi
import math
from typing import TYPE_CHECKING, Iterator, Optional, Type, Union

import numpy as np

from cpython.dict cimport PyDict_Next, PyDict_GetItem
from cpython.tuple cimport PyTuple_GET_ITEM
from cpython.object cimport Py_LE, Py_EQ, Py_GE, PyObject
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

    def __getitem__(self, key):
        return self.vars[key]

    def __hash__(self) -> int:
        return self._hash

    def __len__(self) -> int:
        return len(self.vars)

    def __eq__(self, other: Term) -> bool:
        if self is other:
            return True
        if type(other) is not Term:
            return False

        cdef Term _other = <Term>other
        if self._hash != _other._hash:
            return False

        cdef int n = len(self)
        if n != len(_other) or self._hash != _other._hash:
            return False

        cdef int i
        cdef Variable var1, var2
        for i in range(n):
            var1 = <Variable>PyTuple_GET_ITEM(self.vars, i)
            var2 = <Variable>PyTuple_GET_ITEM(_other.vars, i)
            if var1 is not var2:
                return False
        return True

    def __mul__(self, Term other) -> Term:
        cdef int n1 = len(self)
        cdef int n2 = len(other)
        if n1 == 0: return other
        if n2 == 0: return self

        cdef list vars = [None] * (n1 + n2)
        cdef int i = 0, j = 0, k = 0
        cdef Variable var1, var2
        while i < n1 and j < n2:
            var1 = <Variable>PyTuple_GET_ITEM(self.vars, i)
            var2 = <Variable>PyTuple_GET_ITEM(other.vars, j)
            if hash(var1) <= hash(var2):
                vars[k] = var1
                i += 1
            else:
                vars[k] = var2
                j += 1
            k += 1
        while i < n1:
            vars[k] = <Variable>PyTuple_GET_ITEM(self.vars, i)
            i += 1
            k += 1
        while j < n2:
            vars[k] = <Variable>PyTuple_GET_ITEM(other.vars, j)
            j += 1
            k += 1

        cdef Term res = Term.__new__(Term)
        res.vars = tuple(vars)
        res._hash = hash(res.vars)
        return res

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


cdef class ExprLike:

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__":
            return NotImplemented

        for arg in args:
            if not isinstance(arg, EXPR_OP_TYPES):
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
            if type(args[0]) is AbsExpr:
                return args[0].copy()
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

    def __getitem__(self, key):
        return self._as_expr()[key]

    def __iter__(self) -> Iterator[Union[Term, Expr]]:
        for i in self._as_expr().children:
            yield _unwrap(i)

    def __bool__(self) -> bool:
        return bool(self._as_expr().children)

    def __add__(self, other):
        return self._as_expr() + other

    def __radd__(self, other):
        return self._as_expr() + other

    def __sub__(self, other):
        return self._as_expr() - other

    def __rsub__(self, other):
        return -self._as_expr() + other

    def __mul__(self, other):
        return self._as_expr() * other

    def __rmul__(self, other):
        return self._as_expr() * other

    def __truediv__(self, other):
        return self._as_expr() / other

    def __rtruediv__(self, other):
        return other / self._as_expr()

    def __pow__(self, other):
        return self._as_expr() ** other

    def __rpow__(self, other):
        return other ** self._as_expr()

    def __neg__(self):
        return -self._as_expr()

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

    def degree(self) -> float:
        return self._as_expr().degree()

    def keys(self):
        return self._as_expr().children.keys()

    def items(self):
        return self._as_expr().children.items()

    cpdef list _to_node(self, double coef = 1, int start = 0):
        return self._as_expr()._to_node(coef, start)

    cdef Expr _as_expr(self):
        raise NotImplementedError(
            f"Class {type(self).__name__!s} must implement '_as_expr' method."
        )


cdef class Expr(ExprLike):
    """Base class for mathematical expressions."""

    def __cinit__(self, *_):
        self.coef = 1.0
        self.expo = 1.0
        self._hash = -1

    def __init__(self, *_):
        raise NotImplementedError(
            "Direct instantiation of 'Expr' is not supported. "
            "Please use Variable objects and arithmetic operators to build expressions."
        )

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
            key = _fchild((<Variable>key)._expr_view)
        return self.children.get(_wrap(key), 0.0)

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        cdef Expr res
        if _is_zero(self):
            return _other.copy()
        elif _is_zero(_other):
            return self.copy()
        elif _is_sum(self):
            if _is_single_poly(res := _expr(_to_dict(self, _other))):
                return _to_poly(res)
            return res
        elif _is_sum(_other):
            if _is_single_poly(res := _expr(_to_dict(_other, self))):
                return _to_poly(res)
            return res
        elif _is_expr_equal(self, _other):
            return self * _const(2.0)
        return _expr({_wrap(self): 1.0, _wrap(_other): 1.0})

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self):
            return _other
        elif _is_zero(_other):
            return self
        elif _is_sum(self) and _is_sum(_other):
            _to_dict(self, _other, copy=False)
            _reset_hash(self)

            if _is_single_poly(self):
                return _to_poly(self)
            if type(self) is type(_other):
                return self
            elif isinstance(self, type(_other)):
                return self.copy(False, type(_other))
            return self
        return self + _other

    def __sub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_expr_equal(self, _other):
            return _const(0.0)
        return self + (-_other)

    def __isub__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        return _const(0.0) if _is_expr_equal(self, _other) else self.__iadd__(-_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self) or _is_zero(_other):
            return _const(0.0)
        elif type(self) is ConstExpr:
            if _is_sum(_other):
                return _expr(_normalize(_other, _c(self)))
            return _expr({_wrap(_other): _c(self)})
        elif type(_other) is ConstExpr:
            if _is_sum(self):
                return _expr(_normalize(self, _c(_other)))
            return _expr({_wrap(self): _c(_other)})
        elif _is_expr_equal(self, _other):
            return _pow(_wrap(self), 2.0)
        return _prod((_wrap(self), _wrap(_other)))

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_sum(self) and type(_other) is ConstExpr and _c(_other) != 0:
            self.children = _normalize(self, _c(_other))
            _reset_hash(self)
            return self
        return self * _other

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if _is_zero(self):
            return _const(0.0)
        elif _is_zero(_other):
            raise ZeroDivisionError("division by zero")
        return self * (_other ** _const(-1.0))

    def __rtruediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        return _to_expr(other) / self

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        if not type(_other) is ConstExpr:
            raise TypeError("excepted a constant exponent")
        if _is_zero(self):
            return _const(0.0)
        elif type(_other) is ConstExpr:
            if _c(_other) == 0:
                return _const(1.0)
            elif _c(_other) == 1:
                return self.copy()
        return _pow(_wrap(self), _c(_other))

    def __rpow__(self, other: Union[Number, Expr]) -> Union[ExpExpr, ConstExpr]:
        cdef Expr _other = _to_expr(other)
        if not (type(_other) is ConstExpr and _c(_other) >= 0):
            raise ValueError("excepted a positive base")
        return _const(1.0) if _is_zero(self) else ExpExpr(self * LogExpr(_other))

    def __neg__(self) -> Expr:
        cdef Expr res = self.copy(False)
        res.children = _normalize(self, -1.0)
        return res

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op):
        return _expr_cmp(self, other, op)

    def __repr__(self) -> str:
        return f"Expr({self.children})"

    def degree(self) -> double:
        return max((i.degree() for i in self)) if self else 0

    def _normalize(self) -> Expr:
        self.children = _normalize(self)
        _reset_hash(self)
        return self

    cdef Expr _as_expr(self):
        return self

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
        cdef Expr res = cls.__new__(cls)
        res.children = self.children.copy() if copy else self.children
        if cls is ProdExpr:
            res.coef = self.coef
        elif cls is PowExpr:
            res.expo = self.expo
        return res


cdef class PolynomialExpr(Expr):
    """Expression like `2*x**3 + 4*x*y + constant`."""

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented

        cdef Expr _other = _to_expr(other)
        if not self or not other or type(_other) is not PolynomialExpr:
            return super().__mul__(_other)

        cdef dict res = {}
        cdef Py_ssize_t pos1 = <Py_ssize_t>0, pos2 = <Py_ssize_t>0
        cdef PyObject *k1_ptr = NULL
        cdef PyObject *v1_ptr = NULL
        cdef PyObject *k2_ptr = NULL
        cdef PyObject *v2_ptr = NULL
        cdef PyObject *old_v_ptr = NULL
        cdef object child
        cdef double v1_val, v2_val, prod_v
        while PyDict_Next(self.children, &pos1, &k1_ptr, &v1_ptr):
            if (v1_val := <double>(<object>v1_ptr)) == 0:
                continue

            pos2 = <Py_ssize_t>0
            while PyDict_Next(_other.children, &pos2, &k2_ptr, &v2_ptr):
                if (v2_val := <double>(<object>v2_ptr)) == 0:
                    continue

                child = (<Term>k1_ptr) * (<Term>k2_ptr)
                prod_v = v1_val * v2_val
                if (old_v_ptr := PyDict_GetItem(res, child)) != NULL:
                    res[child] = <double>(<object>old_v_ptr) + prod_v
                else:
                    res[child] = prod_v
        return <PolynomialExpr>_expr(res, PolynomialExpr)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and type(_other) is ConstExpr:
            return self * _const(1.0 / _c(_other))
        return super().__truediv__(_other)

    def __pow__(self, other: Union[Number, Expr]) -> Expr:
        cdef Expr _other = _to_expr(other)
        cdef PolynomialExpr res
        cdef double f_epxo
        cdef int expo
        if (
            self
            and type(_other) is ConstExpr
            and (f_epxo := _c(_other)) > 0
            and f_epxo == (expo := <int>f_epxo)
            and expo != 1
        ):
            res = _const(1.0)
            for _ in range(expo):
                res *= self
            return res
        return super().__pow__(_other)


cdef class ConstExpr(PolynomialExpr):
    """Expression representing for `constant`."""

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Union[ConstExpr, Expr]:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if type(_other) is ConstExpr:
            return _const(_c(self) * _c(_other))
        return super().__mul__(_other)

    def __pow__(self, other: Union[Number, Expr]) -> ConstExpr:
        cdef Expr _other = _to_expr(other)
        if type(_other) is ConstExpr:
            return _const(_c(self) ** _c(_other))
        return <ConstExpr>super().__pow__(_other)

    def __neg__(self) -> ConstExpr:
        return _const(-_c(self))

    def __abs__(self) -> ConstExpr:
        return _const(abs(_c(self)))

    def exp(self) -> ConstExpr:
        return _const(math.exp(_c(self)))

    def log(self) -> ConstExpr:
        return _const(math.log(_c(self)))

    def sqrt(self) -> ConstExpr:
        return _const(math.sqrt(_c(self)))

    def sin(self) -> ConstExpr:
        return _const(math.sin(_c(self)))

    def cos(self) -> ConstExpr:
        return _const(math.cos(_c(self)))

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

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash((frozenset(self.keys()), self.coef)))
        return self._hash

    def __add__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            res = self.copy()
            res.coef += _other.coef
            return res._normalize()
        return super().__add__(_other)

    def __iadd__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            self.coef += _other.coef
            _reset_hash(self)
            return self._normalize()
        return super().__iadd__(_other)

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and type(_other) is ConstExpr:
            res = self.copy()
            res.coef *= _c(_other)
            return res._normalize()
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and type(_other) is ConstExpr:
            self.coef *= _c(_other)
            _reset_hash(self)
            return self._normalize()
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and type(_other) is ConstExpr:
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
        if coef == 0:
            return []

        cdef list node = []
        cdef list sub_node
        cdef list[int] index = []
        cdef object i
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

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash((frozenset(self.keys()), self.expo)))
        return self._hash

    def __mul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            res = self.copy()
            res.expo += _other.expo
            return res._normalize()
        return super().__mul__(_other)

    def __imul__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
            return NotImplemented
        cdef Expr _other = _to_expr(other)
        if self and _is_child_equal(self, _other):
            self.expo += _other.expo
            _reset_hash(self)
            return self._normalize()
        return super().__imul__(_other)

    def __truediv__(self, other: Union[Number, Variable, Expr]) -> Expr:
        if not isinstance(other, EXPR_OP_TYPES):
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

    def __hash__(self) -> int:
        if self._hash != -1:
            return self._hash
        self._hash = _ensure_hash(hash(_fchild(self)))
        return self._hash

    def __richcmp__(self, other: Union[Number, Variable, Expr], int op):
        return _expr_cmp(self, other, op)

    def __repr__(self) -> str:
        cdef object child = _unwrap(_fchild(self))
        if _is_single_poly(child) and child[_fchild(<Expr>child)] == 1:
            return f"{type(self).__name__}({_fchild(<Expr>child)})"
        return f"{type(self).__name__}({child})"

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

    def __abs__(self) -> AbsExpr:
        return <AbsExpr>self.copy()


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


cpdef Expr quickprod(expressions: Iterator[Union[Variable, Expr]]):
    """
    Use inplace multiplication to multiply a list of expressions quickly, avoiding
    intermediate data structures created by Python's built-in prod function.

    Parameters
    ----------
    expressions : Iterator[Union[Variable, Expr]]
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


Number = Union[int, float, np.number]
cdef double INF = float("inf")
cdef tuple NUMBER_TYPES = (int, float, np.number)
cdef tuple EXPR_OP_TYPES = NUMBER_TYPES + (Variable, Expr)
CONST = Term()
exp = np.exp
log = np.log
sqrt = np.sqrt
sin = np.sin
cos = np.cos


cdef inline int _ensure_hash(int h) noexcept:
    return -2 if h == -1 else h


cdef inline void _reset_hash(Expr expr) noexcept:
    if expr._hash != -1: expr._hash = -1


cdef inline double _c(Expr expr):
    return expr.children.get(CONST, 0.0)


cdef inline ConstExpr _const(double c):
    cdef ConstExpr res = ConstExpr.__new__(ConstExpr)
    res.children = {CONST: c}
    return res


cdef inline Expr _expr(dict children, cls: Type[Expr] = Expr):
    cdef Expr res = cls.__new__(cls)
    res.children = children
    return res


cdef inline ProdExpr _prod(tuple children):
    cdef ProdExpr res = ProdExpr.__new__(ProdExpr)
    res.children = dict.fromkeys(children, 1.0)
    return res


cdef inline PowExpr _pow(base: Union[Term, _ExprKey], double expo):
    cdef PowExpr res = PowExpr.__new__(PowExpr)
    res.children = {base: 1.0} 
    res.expo = expo
    return res


cdef inline _wrap(x):
    return _ExprKey(x) if isinstance(x, Expr) else x


cdef inline _unwrap(x):
    return x.expr if isinstance(x, _ExprKey) else x


cdef Expr _to_expr(x: Union[Number, Variable, Expr]):
    if type(x) is Variable:
        return (<Variable>x)._expr_view
    elif isinstance(x, Expr):
        return x
    elif isinstance(x, NUMBER_TYPES):
        return _const(<double>x)
    raise TypeError(f"expected Number, Variable, or Expr, but got {type(x).__name__!s}")


cdef inline Expr _to_poly(Expr expr):
    if _fchild(expr) is CONST:
        return expr if type(expr) is ConstExpr else expr.copy(False, ConstExpr)
    return expr if type(expr) is PolynomialExpr else expr.copy(False, PolynomialExpr)


cdef dict _to_dict(Expr expr, Expr other, bool copy = True):
    cdef dict children = expr.children.copy() if copy else expr.children
    cdef Py_ssize_t pos = <Py_ssize_t>0
    cdef PyObject* k_ptr = NULL
    cdef PyObject* v_ptr = NULL
    cdef PyObject* old_v_ptr = NULL
    cdef double other_v
    cdef object k_obj

    if _is_sum(other):
        while PyDict_Next(other.children, &pos, &k_ptr, &v_ptr):
            if (other_v := <double>(<object>v_ptr)) == 0: 
                continue

            k_obj = <object>k_ptr
            old_v_ptr = PyDict_GetItem(children, k_obj)
            if old_v_ptr != NULL:
                children[k_obj] = <double>(<object>old_v_ptr) + other_v
            else:
                children[k_obj] = <object>v_ptr
    else:
        k_obj = _wrap(other)
        old_v_ptr = PyDict_GetItem(children, k_obj)
        if old_v_ptr != NULL:
            children[k_obj] = <double>(<object>old_v_ptr) + 1.0
        else:
            children[k_obj] = 1.0
    return children


cdef object _expr_cmp(Expr expr, other: Union[Number, Variable, Expr], int op):
    if isinstance(other, np.ndarray):
        return NotImplemented
    cdef Expr _other = _to_expr(other)
    if op == Py_LE:
        if type(_other) is ConstExpr:
            return ExprCons(expr, rhs=_c(_other))
        return ExprCons(expr - _other, rhs=0.0)
    elif op == Py_GE:
        if type(_other) is ConstExpr:
            return ExprCons(expr, lhs=_c(_other))
        return ExprCons(expr - _other, lhs=0.0)
    elif op == Py_EQ:
        if type(_other) is ConstExpr:
            return ExprCons(expr, lhs=_c(_other), rhs=_c(_other))
        return ExprCons(expr - _other, lhs=0.0, rhs=0.0)

    raise NotImplementedError("can only support with '<=', '>=', or '=='")


cdef inline bool _is_sum(expr):
    return type(expr) is Expr or type(expr) is PolynomialExpr or type(expr) is ConstExpr


cdef inline bool _is_zero(Expr expr):
    return not expr or (type(expr) is ConstExpr and _c(expr) == 0)


cdef inline bool _is_single_poly(expr):
    return (
        _is_sum(expr)
        and len(expr.children) == 1
        and type(_fchild(<Expr>expr)) is Term
    )


cdef inline object _fchild(Expr expr):
    cdef Py_ssize_t pos = <Py_ssize_t>0
    cdef PyObject* k_ptr = NULL
    cdef PyObject* v_ptr = NULL
    if PyDict_Next(expr.children, &pos, &k_ptr, &v_ptr):
        return <object>k_ptr
    raise StopIteration("Expr is empty")


cdef bool _is_expr_equal(Expr x, object y):
    if x is y:
        return True
    if not isinstance(y, Expr):
        return False

    cdef Expr _y = <Expr>y
    if len(x.children) != len(_y.children) or x._hash != _y._hash:
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
    return x.children == _y.children


cdef bool _is_child_equal(Expr x, object y):
    if x is y:
        return True
    if type(y) is not type(x):
        return False

    cdef Expr _y = <Expr>y
    if len(x.children) != len(_y.children):
        return False
    return x.keys() == _y.keys()


cdef dict _normalize(Expr expr, double coef = 1.0):
    if coef == 1:
        return expr.children.copy()

    cdef dict res = {}
    cdef Py_ssize_t pos = <Py_ssize_t>0
    cdef PyObject* k_ptr = NULL
    cdef PyObject* v_ptr = NULL
    cdef double v_val
    while PyDict_Next(expr.children, &pos, &k_ptr, &v_ptr):
        if (v_val := <double>(<object>v_ptr)) == 0:
            continue

        if coef != 1.0:
            res[<object>k_ptr] = v_val * coef
        else:
            res[<object>k_ptr] = v_val
    return res


cdef _ensure_unary(x):
    if isinstance(x, Variable):
        return _fchild((<Variable>x)._expr_view)
    elif isinstance(x, Expr):
        return _ExprKey(x)
    raise TypeError(
        f"expected Variable or Expr, but got {type(x).__name__!s}"
    )


cdef inline UnaryExpr _unary(x: Union[Term, _ExprKey], cls: Type[UnaryExpr]):
    cdef UnaryExpr res = cls.__new__(cls)
    res.children = {x: 1.0}
    return res
