##@file expr.pxi
#@brief In this file we implemenet the handling of expressions
#@details We have two types of expressions: Expr and GenExpr.
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
# Modifying the expression directly would be a bug, given that the expression might be re-used by the user.


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
        if isinstance(other, Expr) or isinstance(other, GenExpr):
            return (self - other) <= 0.0
        elif _is_number(other):
            return ExprCons(self, rhs=float(other))
        else:
            raise NotImplementedError
    elif op == 5: # >=
        if isinstance(other, Expr) or isinstance(other, GenExpr):
            return (self - other) >= 0.0
        elif _is_number(other):
            return ExprCons(self, lhs=float(other))
        else:
            raise NotImplementedError
    elif op == 2: # ==
        if isinstance(other, Expr) or isinstance(other, GenExpr):
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

# helper function
def buildGenExprObj(expr):
    """helper function to generate an object of type GenExpr"""
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
    else:
        assert isinstance(expr, GenExpr)
        return expr

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

    def __iter__(self):
        return iter(self.terms)

    def __next__(self):
        try: return next(self.terms)
        except: raise StopIteration

    def __abs__(self):
        return abs(buildGenExprObj(self))

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
        elif isinstance(right, GenExpr):
            return buildGenExprObj(left) + right
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
        elif isinstance(other, GenExpr):
            # is no longer in place, might affect performance?
            # can't do `self = buildGenExprObj(self) + other` since I get
            # TypeError: Cannot convert pyscipopt.scip.SumExpr to pyscipopt.scip.Expr
            return buildGenExprObj(self) + other
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
        elif isinstance(other, GenExpr):
            return buildGenExprObj(self) * other
        else:
            raise NotImplementedError

    def __div__(self, other):
        ''' transforms Expr into GenExpr'''
        if _is_number(other):
            f = 1.0/float(other)
            return f * self
        selfexpr = buildGenExprObj(self)
        return selfexpr.__div__(other)

    def __rdiv__(self, other):
        ''' other / self '''
        if _is_number(self):
            f = 1.0/float(self)
            return f * other
        otherexpr = buildGenExprObj(other)
        return otherexpr.__div__(self)

    def __truediv__(self,other):
        if _is_number(other):
            f = 1.0/float(other)
            return f * self
        selfexpr = buildGenExprObj(self)
        return selfexpr.__truediv__(other)

    def __rtruediv__(self, other):
        ''' other / self '''
        if _is_number(self):
            f = 1.0/float(self)
            return f * other
        otherexpr = buildGenExprObj(other)
        return otherexpr.__truediv__(self)

    def __pow__(self, other, modulo):
        if float(other).is_integer() and other >= 0:
            exp = int(other)
        else: # need to transform to GenExpr
            return buildGenExprObj(self)**other

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
        if len(self.terms) == 0:
            return 0
        else:
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
        if isinstance(self.expr, Expr):
            c = self.expr[CONST]
            self.expr -= c
            assert self.expr[CONST] == 0.0
            self.expr.normalize()
        else:
            assert isinstance(self.expr, GenExpr)
            return

        if not self.lhs is None:
            self.lhs -= c
        if not self.rhs is None:
            self.rhs -= c


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
    exp, log, sqrt = 'exp','log', 'sqrt'
    plus, minus, mul, div, power = '+', '-', '*', '/', '**'
    add = 'sum'
    prod = 'prod'
    fabs = 'abs'
    operatorIndexDic={
            varidx:SCIP_EXPR_VARIDX,
            const:SCIP_EXPR_CONST,
            plus:SCIP_EXPR_PLUS,
            minus:SCIP_EXPR_MINUS,
            mul:SCIP_EXPR_MUL,
            div:SCIP_EXPR_DIV,
            sqrt:SCIP_EXPR_SQRT,
            power:SCIP_EXPR_REALPOWER,
            exp:SCIP_EXPR_EXP,
            log:SCIP_EXPR_LOG,
            fabs:SCIP_EXPR_ABS,
            add:SCIP_EXPR_SUM,
            prod:SCIP_EXPR_PRODUCT
            }
    def getOpIndex(self, op):
        '''returns operator index'''
        return Op.operatorIndexDic[op];

Operator = Op()

cdef class GenExpr:
    '''General expressions of variables with operator overloading.

    Notes:
     - this expressions are not smart enough to identify equal terms
     - in constrast to polynomial expressions, __getitem__ is not implemented
     so expr[x] will generate an error instead of returning the coefficient of x
    '''
    cdef public operatorIndex
    cdef public op
    cdef public children


    def __init__(self): # do we need it
        ''' '''

    def __abs__(self):
        return UnaryExpr(Operator.fabs, self)

    def __add__(self, other):
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

    #TODO: ipow, idiv, etc
    def __div__(self, other):
        divisor = buildGenExprObj(other)
        # we can't divide by 0
        if divisor.getOp() == Operator.const and divisor.number == 0.0:
            raise ZeroDivisionError("cannot divide by 0")
        return self * divisor**(-1)

    def __rdiv__(self, other):
        ''' other / self '''
        otherexpr = buildGenExprObj(other)
        return otherexpr.__div__(self)

    def __truediv__(self,other):
        divisor = buildGenExprObj(other)
        # we can't divide by 0
        if divisor.getOp() == Operator.const and divisor.number == 0.0:
            raise ZeroDivisionError("cannot divide by 0")
        return self * divisor**(-1)

    def __rtruediv__(self, other):
        ''' other / self '''
        otherexpr = buildGenExprObj(other)
        return otherexpr.__truediv__(self)

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

    def __richcmp__(self, other, op):
        '''turn it into a constraint'''
        return _expr_richcmp(self, other, op)

    def degree(self):
        '''Note: none of these expressions should be polynomial'''
        return float('inf') 

    def getOp(self):
        '''returns operator of GenExpr'''
        return self.op


# Sum Expressions
cdef class SumExpr(GenExpr):

    cdef public constant
    cdef public coefs

    def __init__(self):
        self.constant = 0.0
        self.coefs = []
        self.children = []
        self.op = Operator.add
        self.operatorIndex = Operator.operatorIndexDic[self.op]
    def __repr__(self):
        return self.op + "(" + str(self.constant) + "," + ",".join(map(lambda child : child.__repr__(), self.children)) + ")"

# Prod Expressions
cdef class ProdExpr(GenExpr):
    cdef public constant
    def __init__(self):
        self.constant = 1.0
        self.children = []
        self.op = Operator.prod
        self.operatorIndex = Operator.operatorIndexDic[self.op]
    def __repr__(self):
        return self.op + "(" + str(self.constant) + "," + ",".join(map(lambda child : child.__repr__(), self.children)) + ")"

# Var Expressions
cdef class VarExpr(GenExpr):
    cdef public var
    def __init__(self, var):
        self.children = [var]
        self.op = Operator.varidx
        self.operatorIndex = Operator.operatorIndexDic[self.op]
    def __repr__(self):
        return self.children[0].__repr__()

# Pow Expressions
cdef class PowExpr(GenExpr):
    cdef public expo
    def __init__(self):
        self.expo = 1.0
        self.children = []
        self.op = Operator.power
        self.operatorIndex = Operator.operatorIndexDic[self.op]
    def __repr__(self):
        return self.op + "(" + self.children[0].__repr__() + "," + str(self.expo) + ")"

# Exp, Log, Sqrt Expressions
cdef class UnaryExpr(GenExpr):
    def __init__(self, op, expr):
        self.children = []
        self.children.append(expr)
        self.op = op
        self.operatorIndex = Operator.operatorIndexDic[op]
    def __repr__(self):
        return self.op + "(" + self.children[0].__repr__() + ")"

# class for constant expressions
cdef class Constant(GenExpr):
    cdef public number
    def __init__(self,number):
        self.number = number
        self.op = Operator.const
        self.operatorIndex = Operator.operatorIndexDic[self.op]

    def __repr__(self):
        return str(self.number)

def exp(expr):
    """returns expression with exp-function"""
    return UnaryExpr(Operator.exp, buildGenExprObj(expr))
def log(expr):
    """returns expression with log-function"""
    return UnaryExpr(Operator.log, buildGenExprObj(expr))
def sqrt(expr):
    """returns expression with sqrt-function"""
    return UnaryExpr(Operator.sqrt, buildGenExprObj(expr))

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
    op = expr.op
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
