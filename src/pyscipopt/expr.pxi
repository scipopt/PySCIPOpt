##@file expr.pxi
#@brief In this file we implemenet the handling of expressions
#@details @anchor ExprDetails <pre> We have two types of expressions: Expr and GenExpr.
# The Expr can only handle polynomial expressions.
# In addition, one can recover easily information from them.
# A polynomial is a dictionary between `terms` and coefficients.
# A `term` is a tuple of variables
# For examples, 2*x*x*y*z - 1.3 x*y*y + 1 is stored as a
# {Term(x,x,y,z) : 2, Term(x,y,y) : -1.3, Term() : 1}
# Addition of common terms and expansion of exponents occur automatically.
# Given the way `Expr`s are stored, it is easy to access the terms: e.g.
# expr = 2*x*x*y*z - 1.3 x*y*y + 1
# expr[Term(x,x,y,z)] returns 1.3
# expr[Term(x)] returns 0.0
#
# On the other hand, when dealing with expressions more general than polynomials,
# that is, absolute values, exp, log, sqrt or any general exponent, we use GenExpr.
# GenExpr stores expression trees in a rudimentary way.
# Basically, it stores the operator and the list of children.
# We have different types of general expressions that in addition
# to the operation and list of children stores
# SumExpr: coefficients and constant
# ProdExpr: constant
# Constant: constant
# VarExpr: variable
# PowExpr: exponent
# UnaryExpr: nothing
# We do not provide any way of accessing the internal information of the expression tree,
# nor we simplify common terms or do any other type of simplification.
# The `GenExpr` is pass as is to SCIP and SCIP will do what it see fits during presolving.
#
# TODO: All this is very complicated, so we might wanna unify Expr and GenExpr.
# Maybe when consexpr is released it makes sense to revisit this.
# TODO: We have to think about the operations that we define: __isub__, __add__, etc
# and when to copy expressions and when to not copy them.
# For example: when creating a ExprCons from an Expr expr, we store the expression expr
# and then we normalize. When doing the normalization, we do
# ```
# c = self.expr[CONST]
# self.expr -= c
# ```
# which should, in princple, modify the expr. However, since we do not implement __isub__, __sub__
# gets called (I guess) and so a copy is returned.
# Modifying the expression directly would be a bug, given that the expression might be re-used by the user. </pre>
import math
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from cpython.dict cimport PyDict_Next, PyDict_GetItem
from cpython.float cimport PyFloat_Check
from cpython.long cimport PyLong_Check
from cpython.number cimport PyNumber_Check
from cpython.object cimport Py_LE, Py_EQ, Py_GE, Py_TYPE
from cpython.ref cimport PyObject
from cpython.tuple cimport PyTuple_GET_ITEM

from pyscipopt.scip cimport Variable, Solution


if TYPE_CHECKING:
    double = float


cdef class Term:
    '''This is a monomial term'''

    cdef readonly tuple vartuple
    cdef Py_ssize_t hashval

    def __init__(self, *vartuple: Variable):
        self.vartuple = tuple(sorted(vartuple, key=lambda v: v.getIndex()))
        self.hashval = <Py_ssize_t>hash(tuple(v.ptr() for v in self.vartuple))

    def __getitem__(self, idx):
        return self.vartuple[idx]

    def __hash__(self) -> Py_ssize_t:
        return self.hashval

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if <type>Py_TYPE(other) is not Term:
            return False

        cdef int n = len(self)
        cdef Term _other = <Term>other
        if n != len(_other) or self.hashval != _other.hashval:
            return False

        cdef int i
        cdef Variable var1, var2
        for i in range(n):
            var1 = <Variable>PyTuple_GET_ITEM(self.vartuple, i)
            var2 = <Variable>PyTuple_GET_ITEM(_other.vartuple, i)
            if var1.ptr() != var2.ptr():
                return False
        return True

    def __len__(self):
        return len(self.vartuple)

    def __mul__(self, Term other):
        # NOTE: This merge algorithm requires a sorted `Term.vartuple`.
        # This should be ensured in the constructor of Term.
        cdef int n1 = len(self)
        cdef int n2 = len(other)
        if n1 == 0: return other
        if n2 == 0: return self

        cdef list vartuple = [None] * (n1 + n2)
        cdef int i = 0, j = 0, k = 0
        cdef Variable var1, var2
        while i < n1 and j < n2:
            var1 = <Variable>PyTuple_GET_ITEM(self.vartuple, i)
            var2 = <Variable>PyTuple_GET_ITEM(other.vartuple, j)
            if var1.getIndex() <= var2.getIndex():
                vartuple[k] = var1
                i += 1
            else:
                vartuple[k] = var2
                j += 1
            k += 1
        while i < n1:
            vartuple[k] = <Variable>PyTuple_GET_ITEM(self.vartuple, i)
            i += 1
            k += 1
        while j < n2:
            vartuple[k] = <Variable>PyTuple_GET_ITEM(other.vartuple, j)
            j += 1
            k += 1

        cdef Term res = Term.__new__(Term)
        res.vartuple = tuple(vartuple)
        res.hashval = <Py_ssize_t>hash(tuple(v.ptr() for v in res.vartuple))
        return res

    def __repr__(self):
        return 'Term(%s)' % ', '.join([str(v) for v in self.vartuple])

    cpdef double _evaluate(self, Solution sol) except *:
        cdef double res = 1.0
        cdef SCIP* scip_ptr = sol.scip
        cdef SCIP_SOL* sol_ptr = sol.sol
        cdef int i = 0, n = len(self)
        cdef Variable var

        for i in range(n):
            var = <Variable>self.vartuple[i]
            res *= SCIPgetSolVal(scip_ptr, sol_ptr, var.scip_var)
            if res == 0:  # early stop
                return 0.0
        return res


CONST = Term()


# helper function
def buildGenExprObj(expr: Union[int, float, np.number, Expr, GenExpr]) -> GenExpr:
    """helper function to generate an object of type GenExpr"""
    if not _is_genexpr_compatible(expr):
        raise TypeError(f"unsupported type {type(expr).__name__!s}")

    if _is_number(expr):
        return Constant(expr)

    elif isinstance(expr, Expr):
        # loop over terms and create a sumexpr with the sum of each term
        # each term is either a variable (which gets transformed into varexpr)
        # or a product of variables (which gets tranformed into a prod)
        sumexpr = SumExpr()
        for vars, coef in expr.terms.items():
            if len(vars) == 0:
                sumexpr += coef
            elif len(vars) == 1:
                varexpr = VarExpr(vars[0])
                sumexpr += coef * varexpr
            else:
                prodexpr = ProdExpr()
                for v in vars:
                    varexpr = VarExpr(v)
                    prodexpr *= varexpr
                sumexpr += coef * prodexpr
        return sumexpr

    return expr


cdef class ExprLike:

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *args,
        **kwargs,
    ):
        if kwargs.get("out", None) is not None:
            raise TypeError(
                f"{self.__class__.__name__} doesn't support the 'out' parameter in __array_ufunc__"
            )

        if method == "__call__":
            if arrays := [a for a in args if isinstance(a, np.ndarray)]:
                if any(a.dtype.kind not in "fiub" for a in arrays):
                    return NotImplemented
                # If the np.ndarray is of numeric type, all arguments are converted to
                # MatrixExpr or MatrixGenExpr and then the ufunc is applied.
                return ufunc(*[_ensure_matrix(a) for a in args], **kwargs)

            if ufunc is np.add:
                return args[0] + args[1]
            elif ufunc is np.subtract:
                return args[0] - args[1]
            elif ufunc is np.multiply:
                return args[0] * args[1]
            elif ufunc in {np.divide, np.true_divide}:
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
                return args[0].__abs__()
            elif ufunc is np.exp:
                return args[0].exp()
            elif ufunc is np.log:
                return args[0].log()
            elif ufunc is np.sqrt:
                return args[0].sqrt()
            elif ufunc is np.sin:
                return args[0].sin()
            elif ufunc is np.cos:
                return args[0].cos()

        return NotImplemented

    def __abs__(self) -> GenExpr:
        return UnaryExpr(Operator.fabs, buildGenExprObj(self))

    def exp(self) -> GenExpr:
        return UnaryExpr(Operator.exp, buildGenExprObj(self))

    def log(self) -> GenExpr:
        return UnaryExpr(Operator.log, buildGenExprObj(self))

    def sqrt(self) -> GenExpr:
        return UnaryExpr(Operator.sqrt, buildGenExprObj(self))

    def sin(self) -> GenExpr:
        return UnaryExpr(Operator.sin, buildGenExprObj(self))

    def cos(self) -> GenExpr:
        return UnaryExpr(Operator.cos, buildGenExprObj(self))


##@details Polynomial expressions of variables with operator overloading. \n
#See also the @ref ExprDetails "description" in the expr.pxi. 
cdef class Expr(ExprLike):

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

    def __iter__(self):
        return iter(self.terms)

    def __add__(self, other):
        if not _is_expr_compatible(other):
            return NotImplemented

        left = self
        right = other
        terms = left.terms.copy()

        if isinstance(right, Expr):
            # merge the terms by component-wise addition
            for v,c in right.terms.items():
                terms[v] = terms.get(v, 0.0) + c
        elif _is_number(right):
            c = float(right)
            terms[CONST] = terms.get(CONST, 0.0) + c
        return Expr(terms)

    def __iadd__(self, other):
        if not _is_expr_compatible(other):
            return NotImplemented

        if isinstance(other, Expr):
            for v,c in other.terms.items():
                self.terms[v] = self.terms.get(v, 0.0) + c
        elif _is_number(other):
            c = float(other)
            self.terms[CONST] = self.terms.get(CONST, 0.0) + c
        return self

    def __mul__(self, other):
        if not _is_expr_compatible(other):
            return NotImplemented

        cdef dict res = {}
        cdef Py_ssize_t pos1 = <Py_ssize_t>0, pos2 = <Py_ssize_t>0
        cdef PyObject *k1_ptr = NULL
        cdef PyObject *v1_ptr = NULL
        cdef PyObject *k2_ptr = NULL
        cdef PyObject *v2_ptr = NULL
        cdef PyObject *old_v_ptr = NULL
        cdef Term child
        cdef double coef

        if _is_number(other):
            coef = <double>other
            while PyDict_Next(self.terms, &pos1, &k1_ptr, &v1_ptr):
                res[<Term>k1_ptr] = <double>(<object>v1_ptr) * coef

        elif isinstance(other, Expr):
            while PyDict_Next(self.terms, &pos1, &k1_ptr, &v1_ptr):
                pos2 = <Py_ssize_t>0
                while PyDict_Next(other.terms, &pos2, &k2_ptr, &v2_ptr):
                    child = (<Term>k1_ptr) * (<Term>k2_ptr)
                    coef = (<double>(<object>v1_ptr)) * (<double>(<object>v2_ptr))
                    if (old_v_ptr := PyDict_GetItem(res, child)) != NULL:
                        res[child] = <double>(<object>old_v_ptr) + coef
                    else:
                        res[child] = coef
        return Expr(res)

    def __truediv__(self, other):
        if not _is_expr_compatible(other):
            return NotImplemented

        if _is_number(other):
            return 1.0 / other * self
        return buildGenExprObj(self) / other

    def __rtruediv__(self, other):
        ''' other / self '''
        if not _is_expr_compatible(other):
            return NotImplemented
        return buildGenExprObj(other) / self

    def __pow__(self, other, modulo):
        if float(other).is_integer() and other >= 0:
            exp = int(other)
        else: # need to transform to GenExpr
            return buildGenExprObj(self)**other

        res = 1
        for _ in range(exp):
            res *= self
        return res

    def __rpow__(self, other):
        """
        Implements base**x as scip.exp(x * scip.log(base)).
        Note: base must be positive.
        """
        if not _is_number(other):
            raise TypeError(f"Unsupported base type {type(other)} for exponentiation.")

        if (base := <double>other) <= 0.0:
            raise ValueError("Base of a**x must be positive, as expression is reformulated to scip.exp(x * scip.log(a)); got %g" % base)
        return (self * Constant(base).log()).exp()

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

    def __richcmp__(self, other, int op):
        '''turn it into a constraint'''
        return _expr_richcmp(self, other, op)

    def normalize(self):
        '''remove terms with coefficient of 0'''
        self.terms =  {t:c for (t,c) in self.terms.items() if c != 0.0}

    def __repr__(self):
        return 'Expr(%s)' % repr(self.terms)

    def degree(self):
        '''computes highest degree of terms'''
        if len(self.terms) == 0:
            return 0
        else:
            return max(len(v) for v in self.terms)

    cpdef double _evaluate(self, Solution sol) except *:
        cdef double res = 0
        cdef Py_ssize_t pos = <Py_ssize_t>0
        cdef PyObject* key_ptr
        cdef PyObject* val_ptr
        cdef Term term
        cdef double coef

        while PyDict_Next(self.terms, &pos, &key_ptr, &val_ptr):
            term = <Term>key_ptr
            coef = <double>(<object>val_ptr)
            res += coef * term._evaluate(sol)
        return res


cdef class ExprCons:
    '''Constraints with a polynomial expressions and lower/upper bounds.'''
    cdef public expr
    cdef public _lhs
    cdef public _rhs

    def __init__(self, expr, lhs=None, rhs=None):
        self.expr = expr
        self._lhs = lhs
        self._rhs = rhs
        assert not (lhs is None and rhs is None)
        self.normalize()

    def normalize(self):
        '''move constant terms in expression to bounds'''
        if isinstance(self.expr, Expr):
            c = self.expr[CONST]
            self.expr -= c
            assert self.expr[CONST] == 0.0
            self.expr.normalize()
        else:
            assert isinstance(self.expr, GenExpr)
            return

        if not self._lhs is None:
            self._lhs -= c
        if not self._rhs is None:
            self._rhs -= c


    def __richcmp__(self, other, op):
        '''turn it into a constraint'''
        if not _is_number(other):
            raise TypeError('Ranged ExprCons is not well defined!')

        if op == 1: # <=
            if not self._rhs is None:
                raise TypeError('ExprCons already has upper bound')
            assert not self._lhs is None

            return ExprCons(self.expr, lhs=self._lhs, rhs=float(other))
        elif op == 5: # >=
            if not self._lhs is None:
                raise TypeError('ExprCons already has lower bound')
            assert self._lhs is None
            assert not self._rhs is None

            return ExprCons(self.expr, lhs=float(other), rhs=self._rhs)
        else:
            raise NotImplementedError("Ranged ExprCons can only support with '<=' or '>='.")

    def __repr__(self):
        return 'ExprCons(%s, %s, %s)' % (self.expr, self._lhs, self._rhs)

    def __bool__(self):
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

def quickprod(termlist):
    '''multiply linear expressions and constants by avoiding intermediate 
    data structures and multiplying terms inplace
    '''
    result = Expr() + 1
    for term in termlist:
        result *= term
    return result


class Op:
    const = 'const'
    varidx = 'var'
    exp, log, sqrt, sin, cos = 'exp', 'log', 'sqrt', 'sin', 'cos'
    plus, minus, mul, div, power = '+', '-', '*', '/', '**'
    add = 'sum'
    prod = 'prod'
    fabs = 'abs'

Operator = Op()

##@details <pre> General expressions of variables with operator overloading.
#
#@note
#   - these expressions are not smart enough to identify equal terms
#   - in contrast to polynomial expressions, __getitem__ is not implemented
#     so expr[x] will generate an error instead of returning the coefficient of x </pre>
#
#See also the @ref ExprDetails "description" in the expr.pxi. 
cdef class GenExpr(ExprLike):

    cdef public _op
    cdef public children

    def __init__(self): # do we need it
        ''' '''

    def __add__(self, other):
        if not _is_genexpr_compatible(other):
            return NotImplemented

        left = buildGenExprObj(self)
        right = buildGenExprObj(other)
        ans = SumExpr()

        # add left term
        if left.getOp() == Operator.add:
            ans.coefs.extend(left.coefs)
            ans.children.extend(left.children)
            ans.constant += left.constant
        elif left.getOp() == Operator.const:
            ans.constant += left.number
        else:
            ans.coefs.append(1.0)
            ans.children.append(left)

        # add right term
        if right.getOp() == Operator.add:
            ans.coefs.extend(right.coefs)
            ans.children.extend(right.children)
            ans.constant += right.constant
        elif right.getOp() == Operator.const:
            ans.constant += right.number
        else:
            ans.coefs.append(1.0)
            ans.children.append(right)

        return ans

    #def __iadd__(self, other):
    #''' in-place addition, i.e., expr += other '''
    #    assert isinstance(self, Expr)
    #    right = buildGenExprObj(other)
    #
    #    # transform self into sum
    #    if self.getOp() != Operator.add:
    #        newsum = SumExpr()
    #        if self.getOp() == Operator.const:
    #            newsum.constant += self.number
    #        else:
    #            newsum.coefs.append(1.0)
    #            newsum.children.append(self.copy()) # TODO: what is copy?
    #        self = newsum
    #    # add right term
    #    if right.getOp() == Operator.add:
    #        self.coefs.extend(right.coefs)
    #        self.children.extend(right.children)
    #        self.constant += right.constant
    #    elif right.getOp() == Operator.const:
    #        self.constant += right.number
    #    else:
    #        self.coefs.append(1.0)
    #        self.children.append(right)
    #    return self

    def __mul__(self, other):
        if not _is_genexpr_compatible(other):
            return NotImplemented

        left = buildGenExprObj(self)
        right = buildGenExprObj(other)
        ans = ProdExpr()

        # multiply left factor
        if left.getOp() == Operator.prod:
            ans.children.extend(left.children)
            ans.constant *= left.constant
        elif left.getOp() == Operator.const:
            ans.constant *= left.number
        else:
            ans.children.append(left)

        # multiply right factor
        if right.getOp() == Operator.prod:
            ans.children.extend(right.children)
            ans.constant *= right.constant
        elif right.getOp() == Operator.const:
            ans.constant *= right.number
        else:
            ans.children.append(right)

        return ans

    #def __imul__(self, other):
    #''' in-place multiplication, i.e., expr *= other '''
    #    assert isinstance(self, Expr)
    #    right = buildGenExprObj(other)
    #    # transform self into prod
    #    if self.getOp() != Operator.prod:
    #        newprod = ProdExpr()
    #        if self.getOp() == Operator.const:
    #            newprod.constant *= self.number
    #        else:
    #            newprod.children.append(self.copy()) # TODO: what is copy?
    #        self = newprod
    #    # multiply right factor
    #    if right.getOp() == Operator.prod:
    #        self.children.extend(right.children)
    #        self.constant *= right.constant
    #    elif right.getOp() == Operator.const:
    #        self.constant *= right.number
    #    else:
    #        self.children.append(right)
    #    return self

    def __pow__(self, other, modulo):
        expo = buildGenExprObj(other)
        if expo.getOp() != Operator.const:
            raise NotImplementedError("exponents must be numbers")
        if self.getOp() == Operator.const:
            return Constant(self.number**expo.number)
        ans = PowExpr()
        ans.children.append(self)
        ans.expo = expo.number

        return ans

    def __rpow__(self, other):
        """
        Implements base**x as scip.exp(x * scip.log(base)). 
        Note: base must be positive.
        """
        if not _is_number(other):
            raise TypeError(f"Unsupported base type {type(other)} for exponentiation.")

        if (base := <double>other) <= 0.0:
            raise ValueError("Base of a**x must be positive, as expression is reformulated to scip.exp(x * scip.log(a)); got %g" % base)
        return (self * Constant(base).log()).exp()

    #TODO: ipow, idiv, etc
    def __truediv__(self,other):
        if not _is_genexpr_compatible(other):
            return NotImplemented

        divisor = buildGenExprObj(other)
        # we can't divide by 0
        if isinstance(divisor, GenExpr) and divisor.getOp() == Operator.const and divisor.number == 0.0:
            raise ZeroDivisionError("cannot divide by 0")
        return self * divisor**(-1)

    def __rtruediv__(self, other):
        ''' other / self '''
        if not _is_genexpr_compatible(other):
            return NotImplemented
        return buildGenExprObj(other) / self

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return -1.0 * self + other

    def __richcmp__(self, other, int op):
        '''turn it into a constraint'''
        return _expr_richcmp(self, other, op)

    def degree(self):
        '''Note: none of these expressions should be polynomial'''
        return float('inf') 

    def getOp(self):
        '''returns operator of GenExpr'''
        return self._op

    cdef GenExpr copy(self, bool copy = True):
        cdef object cls = <type>Py_TYPE(self)
        cdef GenExpr res = cls.__new__(cls)
        res._op = self._op
        res.children = self.children.copy() if copy else self.children
        if cls is SumExpr:
            (<SumExpr>res).constant = (<SumExpr>self).constant
            (<SumExpr>res).coefs = (<SumExpr>self).coefs.copy() if copy else (<SumExpr>self).coefs
        if cls is ProdExpr:
            (<ProdExpr>res).constant = (<ProdExpr>self).constant
        elif cls is PowExpr:
            (<PowExpr>res).expo = (<PowExpr>self).expo
        return res


# Sum Expressions
cdef class SumExpr(GenExpr):

    cdef public constant
    cdef public coefs

    def __init__(self):
        self.constant = 0.0
        self.coefs = []
        self.children = []
        self._op = Operator.add
    def __repr__(self):
        return self._op + "(" + str(self.constant) + "," + ",".join(map(lambda child : child.__repr__(), self.children)) + ")"

    cpdef double _evaluate(self, Solution sol) except *:
        cdef double res = self.constant
        cdef int i = 0, n = len(self.children)
        cdef list children = self.children
        cdef list coefs = self.coefs
        for i in range(n):
            res += <double>coefs[i] * (<GenExpr>children[i])._evaluate(sol)
        return res


# Prod Expressions
cdef class ProdExpr(GenExpr):

    cdef public constant

    def __init__(self):
        self.constant = 1.0
        self.children = []
        self._op = Operator.prod

    def __repr__(self):
        return self._op + "(" + str(self.constant) + "," + ",".join(map(lambda child : child.__repr__(), self.children)) + ")"

    cpdef double _evaluate(self, Solution sol) except *:
        cdef double res = self.constant
        cdef list children = self.children
        cdef int i = 0, n = len(children)
        for i in range(n):
            res *= (<GenExpr>children[i])._evaluate(sol)
            if res == 0:  # early stop
                return 0.0
        return res


# Var Expressions
cdef class VarExpr(GenExpr):

    cdef public var

    def __init__(self, var):
        self.children = [var]
        self._op = Operator.varidx

    def __repr__(self):
        return self.children[0].__repr__()

    cpdef double _evaluate(self, Solution sol) except *:
        return (<Expr>self.children[0])._evaluate(sol)


# Pow Expressions
cdef class PowExpr(GenExpr):

    cdef public expo

    def __init__(self):
        self.expo = 1.0
        self.children = []
        self._op = Operator.power

    def __repr__(self):
        return self._op + "(" + self.children[0].__repr__() + "," + str(self.expo) + ")"

    cpdef double _evaluate(self, Solution sol) except *:
        return (<GenExpr>self.children[0])._evaluate(sol) ** self.expo


# Exp, Log, Sqrt, Sin, Cos Expressions
cdef class UnaryExpr(GenExpr):
    def __init__(self, op, expr):
        self.children = []
        self.children.append(expr)
        self._op = op

    def __abs__(self) -> UnaryExpr:
        if self._op == "abs":
            return <UnaryExpr>self.copy()
        return UnaryExpr(Operator.fabs, self)

    def __repr__(self):
        return self._op + "(" + self.children[0].__repr__() + ")"

    cpdef double _evaluate(self, Solution sol) except *:
        cdef double res = (<GenExpr>self.children[0])._evaluate(sol)
        return math.fabs(res) if self._op == "abs" else getattr(math, self._op)(res)


# class for constant expressions
cdef class Constant(GenExpr):
    cdef public number
    def __init__(self,number):
        self.number = number
        self._op = Operator.const

    def __repr__(self):
        return str(self.number)

    cpdef double _evaluate(self, Solution sol) except *:
        return self.number


def exp(x):
    """
    returns expression with exp-function

    Parameters
    ----------
    x : Expr, GenExpr, number, np.ndarray, list, or tuple
        - If x is a scalar expression or number, apply the exp function directly to it.
          And if it's a number, convert it to a Constant expression first.
        - If x is a vector (np.ndarray, list, or tuple), apply the exp function
          element-wise using np.frompyfunc to convert each element to a Constant if it's
          a number, and then apply the exp function.

    Returns
    -------
    GenExpr or MatrixGenExpr
        - If x is a scalar expression or number, returns the result of applying the exp
          function to it.
        - If x is a vector, returns an np.ndarray of the same shape with the exp
          function applied element-wise.
    """
    return _wrap_ufunc(x, np.exp)


def log(x):
    """
    returns expression with log-function

    Parameters
    ----------
    x : Expr, GenExpr, number, np.ndarray, list, or tuple
        - If x is a scalar expression or number, apply the log function directly to it.
          And if it's a number, convert it to a Constant expression first.
        - If x is a vector (np.ndarray, list, or tuple), apply the log function
          element-wise using np.frompyfunc to convert each element to a Constant if it's
          a number, and then apply the log function.

    Returns
    -------
    GenExpr or MatrixGenExpr
        - If x is a scalar expression or number, returns the result of applying the log
          function to it.
        - If x is a vector, returns an np.ndarray of the same shape with the log
          function applied element-wise.
    """
    return _wrap_ufunc(x, np.log)


def sqrt(x):
    """
    returns expression with sqrt-function

    Parameters
    ----------
    x : Expr, GenExpr, number, np.ndarray, list, or tuple
        - If x is a scalar expression or number, apply the sqrt function directly to it.
          And if it's a number, convert it to a Constant expression first.
        - If x is a vector (np.ndarray, list, or tuple), apply the sqrt function
          element-wise using np.frompyfunc to convert each element to a Constant if it's
          a number, and then apply the sqrt function.

    Returns
    -------
    GenExpr or MatrixGenExpr
        - If x is a scalar expression or number, returns the result of applying the sqrt
          function to it.
        - If x is a vector, returns an np.ndarray of the same shape with the sqrt
          function applied element-wise.
    """
    return _wrap_ufunc(x, np.sqrt)


def sin(x):
    """
    returns expression with sin-function

    Parameters
    ----------
    x : Expr, GenExpr, number, np.ndarray, list, or tuple
        - If x is a scalar expression or number, apply the sin function directly to it.
          And if it's a number, convert it to a Constant expression first.
        - If x is a vector (np.ndarray, list, or tuple), apply the sin function
          element-wise using np.frompyfunc to convert each element to a Constant if it's
          a number, and then apply the sin function.

    Returns
    -------
    GenExpr or MatrixGenExpr
        - If x is a scalar expression or number, returns the result of applying the sin
          function to it.
        - If x is a vector, returns an np.ndarray of the same shape with the sin
          function applied element-wise.
    """
    return _wrap_ufunc(x, np.sin)


def cos(x):
    """
    returns expression with cos-function

    Parameters
    ----------
    x : Expr, GenExpr, number, np.ndarray, list, or tuple
        - If x is a scalar expression or number, apply the cos function directly to it.
          And if it's a number, convert it to a Constant expression first.
        - If x is a vector (np.ndarray, list, or tuple), apply the cos function
          element-wise using np.frompyfunc to convert each element to a Constant if it's
          a number, and then apply the cos function.

    Returns
    -------
    GenExpr or MatrixGenExpr
        - If x is a scalar expression or number, returns the result of applying the cos
          function to it.
        - If x is a vector, returns an np.ndarray of the same shape with the cos
          function applied element-wise.
    """
    return _wrap_ufunc(x, np.cos)


cdef inline object _to_const(object x):
    return Constant(<double>x) if _is_number(x) else x

cdef object _vec_to_const = np.frompyfunc(_to_const, 1, 1)

cdef inline object _wrap_ufunc(object x, object ufunc):
    """
    Apply a universal function (ufunc) to an expression or a collection of expressions.

    Parameters
    ----------
    x : Expr, GenExpr, number, np.ndarray, list, or tuple
        - If x is a scalar expression or number, apply the ufunc directly to it. And if
          it's a number, convert it to a Constant expression first.
        - If x is a vector (np.ndarray, list, or tuple), apply the ufunc element-wise
          using np.frompyfunc to convert each element to a Constant if it's a number,
          and then apply the ufunc.

    ufunc : np.ufunc
        The universal function to be applied to x.

    Returns
    -------
    GenExpr or MatrixGenExpr
        - If x is a scalar expression or number, returns the result of applying the
          ufunc to it.
        - If x is a vector, returns an np.ndarray of the same shape with the ufunc
          applied element-wise.
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        res = ufunc(_vec_to_const(x))
        return res.view(MatrixGenExpr) if isinstance(res, np.ndarray) else res
    return ufunc(_to_const(x))

cdef inline object _ensure_matrix(object arg):
    if type(arg) is np.ndarray:
        return arg.view(MatrixExpr)
    matrix = MatrixExpr if isinstance(arg, Expr) else MatrixGenExpr
    return np.array(arg, dtype=object).view(matrix)


def expr_to_nodes(expr):
    '''transforms tree to an array of nodes. each node is an operator and the position of the 
    children of that operator (i.e. the other nodes) in the array'''
    assert isinstance(expr, GenExpr)
    nodes = []
    expr_to_array(expr, nodes)
    return nodes

def value_to_array(val, nodes):
    """adds a given value to an array"""
    nodes.append(tuple(['const', [val]]))
    return len(nodes) - 1

# there many hacky things here: value_to_array is trying to mimick
# the multiple dispatch of julia. Also that we have to ask which expression is which
# in order to get the constants correctly
# also, for sums, we are not considering coefficients, because basically all coefficients are 1
# haven't even consider substractions, but I guess we would interpret them as a - b = a + (-1) * b
def expr_to_array(expr, nodes):
    """adds expression to array"""
    op = expr._op
    if op == Operator.const: # FIXME: constant expr should also have children!
        nodes.append(tuple([op, [expr.number]]))
    elif op != Operator.varidx:
        indices = []
        nchildren = len(expr.children)
        for child in expr.children:
            pos = expr_to_array(child, nodes) # position of child in the final array of nodes, 'nodes'
            indices.append(pos)
        if op == Operator.power:
            pos = value_to_array(expr.expo, nodes)
            indices.append(pos)
        elif (op == Operator.add and expr.constant != 0.0) or (op == Operator.prod and expr.constant != 1.0):
            pos = value_to_array(expr.constant, nodes)
            indices.append(pos)
        nodes.append( tuple( [op, indices] ) )
    else: # var
        nodes.append( tuple( [op, expr.children] ) )
    return len(nodes) - 1


cdef inline bint _is_number(object x):
    if PyLong_Check(x) or PyFloat_Check(x):
        return True
    if cnp.PyArray_Check(x) or isinstance(x, (ExprLike, list, tuple)):
        return False
    return PyNumber_Check(x)

cdef inline bint _is_expr_compatible(object x):
    return _is_number(x) or isinstance(x, Expr)

cdef inline bint _is_genexpr_compatible(object x):
    return _is_expr_compatible(x) or isinstance(x, GenExpr)

cdef object _expr_richcmp(
    ExprLike self,
    other: Union[int, float, np.number, Expr, GenExpr],
    int op,
):
    if isinstance(other, np.ndarray):
        return NotImplemented
    if not _is_genexpr_compatible(other):
        raise TypeError(f"unsupported type {type(other).__name__!s}")

    if op == Py_LE:
        if _is_number(other):
            return ExprCons(self, rhs=<double>other)
        return ExprCons(self - other, rhs=0.0)
    elif op == Py_GE:
        if _is_number(other):
            return ExprCons(self, lhs=<double>other)
        return ExprCons(self - other, lhs=0.0)
    elif op == Py_EQ:
        if _is_number(other):
            return ExprCons(self, lhs=<double>other, rhs=<double>other)
        return ExprCons(self - other, lhs=0.0, rhs=0.0)
    raise NotImplementedError("can only support with '<=', '>=', or '=='")
