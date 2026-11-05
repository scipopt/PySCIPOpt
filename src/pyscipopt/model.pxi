cdef class Column:
    """Base class holding a pointer to corresponding SCIP_COL."""

    @staticmethod
    cdef create(SCIP_COL* scipcol):
        """
        Main method for creating a Column class. Is used instead of __init__.

        Parameters
        ----------
        scipcol : SCIP_COL*
            A pointer to the SCIP_COL

        Returns
        -------
        col : Column
            The Python representative of the SCIP_COL

        """
        if scipcol == NULL:
            raise Warning("cannot create Column with SCIP_COL* == NULL")
        col = Column()
        col.scip_col = scipcol
        return col

    def getLPPos(self):
        """
        Gets position of column in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
        return SCIPcolGetLPPos(self.scip_col)

    def getBasisStatus(self):
        """
        Gets the basis status of a column in the LP solution

        Returns
        -------
        str
            Possible values are "lower", "basic", "upper", and "zero"

        Raises
        ------
        Exception
            If SCIP returns an unknown basis status

        Notes
        -----
        Returns basis status "zero" for columns not in the current SCIP LP.

        """
        cdef SCIP_BASESTAT stat = SCIPcolGetBasisStatus(self.scip_col)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            return "zero"
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        """
        Returns whether the associated variable is of integral type (binary, integer, implicit integer).

        Returns
        -------
        bool

        """
        return SCIPcolIsIntegral(self.scip_col)

    def getVar(self):
        """
        Gets variable this column represents.

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* var = SCIPcolGetVar(self.scip_col)
        return Variable.create(var)

    def getPrimsol(self):
        """
        Gets the primal LP solution of a column.

        Returns
        -------
        float

        """
        return SCIPcolGetPrimsol(self.scip_col)

    def getLb(self):
        """
        Gets lower bound of column.

        Returns
        -------
        float

        """
        return SCIPcolGetLb(self.scip_col)

    def getUb(self):
        """
        Gets upper bound of column.

        Returns
        -------
        float

        """
        return SCIPcolGetUb(self.scip_col)

    def getObjCoeff(self):
        """
        Gets objective value coefficient of a column.

        Returns
        -------
        float

        """
        return SCIPcolGetObj(self.scip_col)

    def getAge(self):
        """
        Gets the age of the column, i.e., the total number of successive times a column was in the LP
        and was 0.0 in the solution.

        Returns
        -------
        int

        """
        return SCIPcolGetAge(self.scip_col)

    def __hash__(self):
        return hash(<size_t>self.scip_col)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_col == (<Column>other).scip_col)

cdef class Row:
    """Base class holding a pointer to corresponding SCIP_ROW."""

    @staticmethod
    cdef create(SCIP_ROW* sciprow):
        """
        Main method for creating a Row class. Is used instead of __init__.

        Parameters
        ----------
        sciprow : SCIP_ROW*
            A pointer to the SCIP_ROW

        Returns
        -------
        row : Row
            The Python representative of the SCIP_ROW

        """
        if sciprow == NULL:
            raise Warning("cannot create Row with SCIP_ROW* == NULL")
        row = Row()
        row.scip_row = sciprow
        return row

    @property
    def name(self):
        return bytes(SCIProwGetName(self.scip_row)).decode("utf-8")

    def getLhs(self):
        """
        Returns the left hand side of row.

        Returns
        -------
        float

        """
        return SCIProwGetLhs(self.scip_row)

    def getRhs(self):
        """
        Returns the right hand side of row.

        Returns
        -------
        float

        """
        return SCIProwGetRhs(self.scip_row)

    def getConstant(self):
        """
        Gets constant shift of row.

        Returns
        -------
        float

        """
        return SCIProwGetConstant(self.scip_row)

    def getDualsol(self):
        """
        Returns the dual solution of row.

        Returns
        -------
        float

        """
        return SCIProwGetDualsol(self.scip_row)

    def getDualfarkas(self):
        """
        Returns the dual Farkas solution of row.

        Returns
        -------
        float

        """
        return SCIProwGetDualfarkas(self.scip_row)

    def getLPPos(self):
        """
        Gets position of row in current LP, or -1 if it is not in LP.

        Returns
        -------
        int

        """
        return SCIProwGetLPPos(self.scip_row)

    def getBasisStatus(self):
        """
        Gets the basis status of a row in the LP solution.

        Returns
        -------
        str
            Possible values are "lower", "basic", and "upper"

        Raises
        ------
        Exception
            If SCIP returns an unknown or "zero" basis status

        Notes
        -----
        Returns basis status "basic" for rows not in the current SCIP LP.

        """
        cdef SCIP_BASESTAT stat = SCIProwGetBasisStatus(self.scip_row)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            # this shouldn't happen!
            raise Exception('SCIP returned base status zero for a row!')
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        """
        Returns TRUE iff the activity of the row (without the row's constant)
        is always integral in a feasible solution.

        Returns
        -------
        bool

        """
        return SCIProwIsIntegral(self.scip_row)

    def isLocal(self):
        """
        Returns TRUE iff the row is only valid locally.

        Returns
        -------
        bool

        """
        return SCIProwIsLocal(self.scip_row)

    def isModifiable(self):
        """
        Returns TRUE iff row is modifiable during node processing (subject to column generation).

        Returns
        -------
        bool

        """
        return SCIProwIsModifiable(self.scip_row)

    def isRemovable(self):
        """
        Returns TRUE iff row is removable from the LP (due to aging or cleanup).

        Returns
        -------
        bool

        """
        return SCIProwIsRemovable(self.scip_row)

    def isInGlobalCutpool(self):
        """
        Return TRUE iff row is a member of the global cut pool.

        Returns
        -------
        bool

        """
        return SCIProwIsInGlobalCutpool(self.scip_row)

    def getOrigintype(self):
        """
        Returns type of origin that created the row.

        Returns
        -------
        PY_SCIP_ROWORIGINTYPE

        """
        return SCIProwGetOrigintype(self.scip_row)

    def getConsOriginConshdlrtype(self):
        """
        Returns type of constraint handler that created the row.

        Returns
        -------
        str

        """
        cdef SCIP_CONS* scip_con = SCIProwGetOriginCons(self.scip_row)
        return bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(scip_con))).decode('UTF-8')

    def getNNonz(self):
        """
        Get number of nonzero entries in row vector.

        Returns
        -------
        int

        """
        return SCIProwGetNNonz(self.scip_row)

    def getNLPNonz(self):
        """
        Get number of nonzero entries in row vector that correspond to columns currently in the SCIP LP.

        Returns
        -------
        int

        """
        return SCIProwGetNLPNonz(self.scip_row)

    def getCols(self):
        """
        Gets list with columns of nonzero entries

        Returns
        -------
        list of Column

        """
        cdef SCIP_COL** cols = SCIProwGetCols(self.scip_row)
        cdef int i
        return [Column.create(cols[i]) for i in range(self.getNNonz())]

    def getVals(self):
        """
        Gets list with coefficients of nonzero entries.

        Returns
        -------
        list of int

        """
        cdef SCIP_Real* vals = SCIProwGetVals(self.scip_row)
        cdef int i
        return [vals[i] for i in range(self.getNNonz())]

    def getAge(self):
        """
        Gets the age of the row. (The consecutive times the row has been non-active in the LP).

        Returns
        -------
        int

        """
        return SCIProwGetAge(self.scip_row)

    def getNorm(self):
        """
        Gets Euclidean norm of row vector.

        Returns
        -------
        float

        """
        return SCIProwGetNorm(self.scip_row)

    def __hash__(self):
        return hash(<size_t>self.scip_row)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_row == (<Row>other).scip_row)

cdef class NLRow:
    """Base class holding a pointer to corresponding SCIP_NLROW."""

    @staticmethod
    cdef create(SCIP_NLROW* scipnlrow):
        """
        Main method for creating a NLRow class. Is used instead of __init__.

        Parameters
        ----------
        scipnlrow : SCIP_NLROW*
            A pointer to the SCIP_NLROW

        Returns
        -------
        nlrow : NLRow
            The Python representative of the SCIP_NLROW

        """
        if scipnlrow == NULL:
            raise Warning("cannot create NLRow with SCIP_NLROW* == NULL")
        nlrow = NLRow()
        nlrow.scip_nlrow = scipnlrow
        return nlrow

    @property
    def name(self):
        return bytes(SCIPnlrowGetName(self.scip_nlrow)).decode("utf-8")

    def getConstant(self):
        """
        Returns the constant of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetConstant(self.scip_nlrow)

    def getLinearTerms(self):
        """
        Returns a list of tuples (var, coef) representing the linear part of a nonlinear row.

        Returns
        -------
        list of tuple

        """
        cdef SCIP_VAR** linvars = SCIPnlrowGetLinearVars(self.scip_nlrow)
        cdef SCIP_Real* lincoefs = SCIPnlrowGetLinearCoefs(self.scip_nlrow)
        cdef int nlinvars = SCIPnlrowGetNLinearVars(self.scip_nlrow)
        cdef int i
        return [(Variable.create(linvars[i]), lincoefs[i]) for i in range(nlinvars)]

    def getLhs(self):
        """
        Returns the left hand side of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetLhs(self.scip_nlrow)

    def getRhs(self):
        """
        Returns the right hand side of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetRhs(self.scip_nlrow)

    def getDualsol(self):
        """
        Gets the dual NLP solution of a nonlinear row.

        Returns
        -------
        float

        """
        return SCIPnlrowGetDualsol(self.scip_nlrow)

    def __hash__(self):
        return hash(<size_t>self.scip_nlrow)

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.scip_nlrow == (<NLRow>other).scip_nlrow)


cdef class Model(BaseModel):
    """Model solving with floating-point arithmetic."""

    def getFirstPrimalBound(self):
        """
        Gets the primal bound of the very first solution in the original space.

        Returns
        -------
        float

        """
        return SCIPgetFirstPrimalBound(self._scip)

    def getAvgLowerbound(self):
        """
        Gets average lower (dual) bound of all unprocessed nodes in transformed problem.

        Returns
        -------
        float

        """
        return SCIPgetAvgLowerbound(self._scip)

    def getAvgDualbound(self):
        """
        Gets average dual bound of all unprocessed nodes for original problem.

        Returns
        -------
        float

        """
        return SCIPgetAvgDualbound(self._scip)

    def getLowerbound(self):
        """
        Gets global lower (dual) bound of the transformed problem.

        Returns
        -------
        float

        """
        return SCIPgetLowerbound(self._scip)

    def getCutoffbound(self):
        """
        Gets the cutoff bound of the transformed problem.

        Returns
        -------
        float

        """
        return SCIPgetCutoffbound(self._scip)

    def getUpperbound(self):
        """
        Gets global upper (primal) bound in transformed problem (objective value of best solution or user objective limit).

        Returns
        -------
        float
        """
        return SCIPgetUpperbound(self._scip)

    def infinity(self):
        """
        Retrieve SCIP's infinity value.

        Returns
        -------
        int

        """
        return SCIPinfinity(self._scip)

    def epsilon(self):
        """
        Retrieve epsilon for e.g. equality checks.

        Returns
        -------
        float

        """
        return SCIPepsilon(self._scip)

    def feastol(self):
        """
        Retrieve feasibility tolerance.

        Returns
        -------
        float

        """
        return SCIPfeastol(self._scip)

    def feasFrac(self, value):
        """
        Returns fractional part of value, i.e. x - floor(x) in feasible tolerance: x - floor(x+feastol).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasFrac(self._scip, value)

    def frac(self, value):
        """
        Returns fractional part of value, i.e. x - floor(x) in epsilon tolerance: x - floor(x+eps).

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfrac(self._scip, value)

    def feasFloor(self, value):
        """
        Rounds value + feasibility tolerance down to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasFloor(self._scip, value)

    def feasCeil(self, value):
        """
        Rounds value - feasibility tolerance up to the next integer.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasCeil(self._scip, value)

    def feasRound(self, value):
        """
        Rounds value to the nearest integer in feasibility tolerance.

        Parameters
        ----------
        value : float

        Returns
        -------
        float

        """
        return SCIPfeasRound(self._scip, value)

    def isZero(self, value):
        """
        Returns whether abs(value) < eps.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisZero(self._scip, value)

    def isFeasZero(self, value):
        """
        Returns whether abs(value) < feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasZero(self._scip, value)

    def isInfinity(self, value):
        """
        Returns whether value is SCIP's infinity.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisInfinity(self._scip, value)

    def isFeasNegative(self, value):
        """
        Returns whether value < -feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasNegative(self._scip, value)

    def isFeasPositive(self, value):
        """
        Returns whether value > feastol.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasPositive(self._scip, value)

    def isFeasIntegral(self, value):
        """
        Returns whether value is integral within the LP feasibility bounds.

        Parameters
        ----------
        value : float

        Returns
        -------
        bool

        """
        return SCIPisFeasIntegral(self._scip, value)

    def isIntegral(self, value):
        """
        Returns whether value is integral within epsilon tolerance.

        Parameters
        ----------
        value : float
            value to check

        Returns
        -------
        bool

        """
        return SCIPisIntegral(self._scip, value)

    def adjustedVarLb(self, Variable var, lb):
        """
        Returns the adjusted (i.e. rounded, if the given variable is of integral type) lower bound value;
        does not change the bounds of the variable.

        Parameters
        ----------
        var : Variable
            variable for which the bound is adjusted
        lb : float
            lower bound value to adjust

        Returns
        -------
        float
            adjusted lower bound

        """
        return SCIPadjustedVarLb(self._scip, var.scip_var, lb)

    def adjustedVarUb(self, Variable var, ub):
        """
        Returns the adjusted (i.e. rounded, if the given variable is of integral type) upper bound value;
        does not change the bounds of the variable.

        Parameters
        ----------
        var : Variable
            variable for which the bound is adjusted
        ub : float
            upper bound value to adjust

        Returns
        -------
        float
            adjusted upper bound

        """
        return SCIPadjustedVarUb(self._scip, var.scip_var, ub)

    def isEQ(self, val1, val2):
        """
        Checks, if values are in range of epsilon.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisEQ(self._scip, val1, val2)

    def isFeasEQ(self, val1, val2):
        """
        Returns if relative difference between val1 and val2 is in range of feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasEQ(self._scip, val1, val2)
    
    def isFeasLT(self, val1, val2):
        """
        Returns whether relative difference between val1 and val2 is lower than minus feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasLT(self._scip, val1, val2)
    
    def isFeasLE(self, val1, val2):
        """
        Returns whether relative difference between val1 and val2 is not greater than feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasLE(self._scip, val1, val2)
    
    def isFeasGT(self, val1, val2):
        """
        Returns whether relative difference between val1 and val2 is greater than feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasGT(self._scip, val1, val2)
    
    def isFeasGE(self, val1, val2):
        """
        Returns whether relative difference of val1 and val2 is not lower than minus feasibility tolerance.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisFeasGE(self._scip, val1, val2)

    def isLE(self, val1, val2):
        """
        Returns whether val1 <= val2 + eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisLE(self._scip, val1, val2)

    def isLT(self, val1, val2):
        """
        Returns whether val1 < val2 - eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisLT(self._scip, val1, val2)

    def isGE(self, val1, val2):
        """
        Returns whether val1 >= val2 - eps.

        Parameters
        ----------
        val1 : float
        val2 : float

        Returns
        -------
        bool

        """
        return SCIPisGE(self._scip, val1, val2)

    def isGT(self, val1, val2):
        """
        Returns whether val1 > val2 + eps.

        Parameters
        ----------
        val1 : float
        val2 : foat

        Returns
        -------
        bool

        """
        return SCIPisGT(self._scip, val1, val2)

    def isHugeValue(self, val):
        """
        Checks if value is huge and should be
        handled separately (e.g., in activity computation).

        Parameters
        ----------
        val : float

        Returns
        -------
        bool

        """
        return SCIPisHugeValue(self._scip, val)

    def isPositive(self, val):
        """
        Returns whether val > eps.

        Parameters
        ----------
        val : float

        Returns
        -------
        bool

        """
        return SCIPisPositive(self._scip, val)

    def isNegative(self, val):
        """
        Returns whether val < -eps.

        Parameters
        ----------
        val : float

        Returns
        -------
        bool

        """
        return SCIPisNegative(self._scip, val)

    def setObjlimit(self, objlimit):
        """
        Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        Parameters
        ----------
        objlimit : float
            limit on the objective function

        """
        PY_SCIP_CALL(SCIPsetObjlimit(self._scip, objlimit))

    def getObjlimit(self):
        """
        Returns current limit on objective function.

        Returns
        -------
        float

        """
        return SCIPgetObjlimit(self._scip)

    def setObjective(self, expr, sense = 'minimize', clear = 'true'):
        """
        Establish the objective function as a linear expression.

        Parameters
        ----------
        expr : Expr or float
            the objective function SCIP Expr, or constant value
        sense : str, optional
            the objective sense ("minimize" or "maximize") (Default value = 'minimize')
        clear : bool, optional
            set all other variables objective coefficient to zero (Default value = 'true')

        """
        cdef SCIP_VAR** vars
        cdef int nvars
        cdef SCIP_Real coef
        cdef int i
        cdef _VarArray wrapper

        if not isinstance(expr, Expr) and not _is_number(expr):
            raise TypeError(
                f"requires Expr or number but got type {type(expr).__name__!s}"
            )

        # turn the constant value into an Expr instance for further processing
        expr = Expr() + expr
        if expr.degree() > 1:
            raise ValueError("SCIP does not support nonlinear objective functions. Consider using set_nonlinear_objective in the pyscipopt.recipe.nonlinear")

        if clear:
            # clear existing objective function
            self.addObjoffset(-self.getObjoffset())
            vars = SCIPgetOrigVars(self._scip)
            nvars = SCIPgetNOrigVars(self._scip)
            for i in range(nvars):
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, vars[i], 0.0))

        if expr[CONST] != 0.0:
            self.addObjoffset(expr[CONST])

        for term, coef in expr.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                wrapper = _VarArray(term[0])
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, wrapper.ptr[0], coef))

        if sense == "minimize":
            self.setMinimize()
        elif sense == "maximize":
            self.setMaximize()
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

    def getObjective(self):
        """
        Retrieve objective function as Expr.

        Returns
        -------
        Expr

        """
        variables = self.getVars()
        objective = Expr()
        for var in variables:
            coeff = var.getObj()
            if coeff != 0:
                objective += coeff * var
        objective.normalize()
        return objective

    def addObjoffset(self, offset, solutions = False):
        """
        Add constant offset to objective.

        Parameters
        ----------
        offset : float
            offset to add
        solutions : bool, optional
            add offset also to existing solutions (Default value = False)

        """
        if solutions:
            PY_SCIP_CALL(SCIPaddObjoffset(self._scip, offset))
        else:
            PY_SCIP_CALL(SCIPaddOrigObjoffset(self._scip, offset))

    def getObjoffset(self, original = True):
        """
        Retrieve constant objective offset

        Parameters
        ----------
        original : bool, optional
            offset of original or transformed problem (Default value = True)

        Returns
        -------
        float

        """
        if original:
            return SCIPgetOrigObjoffset(self._scip)
        else:
            return SCIPgetTransObjoffset(self._scip)

    def getLocalEstimate(self, original = False):
        """
        Gets estimate of best primal solution w.r.t. original or transformed problem contained in current subtree.

        Parameters
        ----------
        original : bool, optional
            get estimate of original or transformed problem (Default value = False)

        Returns
        -------
        float

        """
        if original:
            return SCIPgetLocalOrigEstimate(self._scip)
        else:
            return SCIPgetLocalTransEstimate(self._scip)

    # Variable Functions
    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, pricedVar=False, pricedVarScore=1.0, deletable=False):
        """
        Create a new variable. Default variable is non-negative and continuous.

        Parameters
        ----------
        name : str, optional
            name of the variable, generic if empty (Default value = '')
        vtype : str, optional
            type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
            (Default value = 'C')
        lb : float or None, optional
            lower bound of the variable, use None for -infinity (Default value = 0.0)
        ub : float or None, optional
            upper bound of the variable, use None for +infinity (Default value = None)
        obj : float, optional
            objective value of variable (Default value = 0.0)
        pricedVar : bool, optional
            is the variable a pricing candidate? (Default value = False)
        pricedVarScore : float, optional
            score of variable in case it is priced, the higher the better (Default value = 1.0)

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* scip_var

        # replace empty name with generic one
        if name == '':
            name = 'x'+str(SCIPgetNVars(self._scip)+1)
        cname = str_conversion(name)

        # replace None with corresponding infinity
        if lb is None:
            lb = -SCIPinfinity(self._scip)
        if ub is None:
            ub = SCIPinfinity(self._scip)

        vtype = vtype.upper()
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS))
        elif vtype in ['B', 'BINARY']:
            if ub > 1.0:
                ub = 1.0
            if lb < 0.0:
                lb = 0.0
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_BINARY))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_INTEGER))
        elif vtype in ['M', 'IMPLINT']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_DEPRECATED_VARTYPE_IMPLINT))
        else:
            raise Warning("unrecognized variable type")

        if deletable:
            SCIPvarMarkDeletable(scip_var)

        if pricedVar:
            PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, pricedVarScore))
        else:
            PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

        pyVar = self._getOrCreateVar(scip_var)

        #setting the variable data
        SCIPvarSetData(scip_var, <SCIP_VARDATA*>pyVar)
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &scip_var))
        return pyVar

    def fixVar(self, Variable var, val):
        """
        Fixes the variable var to the value val if possible.

        Parameters
        ----------
        var : Variable
            variable to fix
        val : float
            the fix value

        Returns
        -------
        infeasible : bool
            Is the fixing infeasible?
        fixed : bool
            Was the fixing performed?

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool fixed
        PY_SCIP_CALL(SCIPfixVar(self._scip, var.scip_var, val, &infeasible, &fixed))
        return infeasible, fixed

    def aggregateVars(self, Variable varx, Variable vary, coefx=1.0, coefy=-1.0, rhs=0.0):
        """
        Aggregate two variables by adding an aggregation constraint.

        The aggregation is defined by the linear equation:

            coefx * varx + coefy * vary = rhs

        After aggregation, varx becomes a redundant variable and vary remains active.
        The aggregation effectively substitutes varx with: (rhs - coefy * vary) / coefx

        This method can only be called during presolving.

        Parameters
        ----------
        varx : Variable
            variable to be aggregated (will become redundant)
        vary : Variable
            variable to aggregate with (will remain active)
        coefx : float, optional
            coefficient for varx in the aggregation equation (default: 1.0)
        coefy : float, optional
            coefficient for vary in the aggregation equation (default: -1.0)
        rhs : float, optional
            right-hand side of the aggregation equation (default: 0.0)

        Returns
        -------
        infeasible : bool
            whether the aggregation is infeasible (e.g., bounds are incompatible)
        redundant : bool
            whether the aggregation makes varx redundant
        aggregated : bool
            whether the aggregation was actually performed

        Examples
        --------
        To express x = y (i.e., 1*x + (-1)*y = 0):

            infeas, redun, aggr = model.aggregateVars(x, y, 1.0, -1.0, 0.0)

        To express x = 5 - y (i.e., 1*x + 1*y = 5):

            infeas, redun, aggr = model.aggregateVars(x, y, 1.0, 1.0, 5.0)

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool redundant
        cdef SCIP_Bool aggregated
        PY_SCIP_CALL(SCIPaggregateVars(self._scip,
                                    varx.scip_var,
                                    vary.scip_var,
                                    coefx,
                                    coefy,
                                    rhs,
                                    &infeasible,
                                    &redundant,
                                    &aggregated))
        return infeasible, redundant, aggregated

    def tightenVarLb(self, Variable var, lb, force=False):
        """
        Tighten the lower bound in preprocessing or current node, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        lb : float
            possible new lower bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance (default = False)

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarLb(self._scip, var.scip_var, lb, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarUb(self, Variable var, ub, force=False):
        """
        Tighten the upper bound in preprocessing or current node, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        ub : float
            possible new upper bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarUb(self._scip, var.scip_var, ub, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarUbGlobal(self, Variable var, ub, force=False):
        """
        Tighten the global upper bound, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        ub : float
            possible new upper bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarUbGlobal(self._scip, var.scip_var, ub, force, &infeasible, &tightened))
        return infeasible, tightened

    def tightenVarLbGlobal(self, Variable var, lb, force=False):
        """Tighten the global lower bound, if the bound is tighter.

        Parameters
        ----------
        var : Variable
            SCIP variable
        lb : float
            possible new lower bound
        force : bool, optional
            force tightening even if below bound strengthening tolerance

        Returns
        -------
        infeasible : bool
            Whether new domain is empty
        tightened : bool
            Whether the bound was tightened

        """
        cdef SCIP_Bool infeasible
        cdef SCIP_Bool tightened
        PY_SCIP_CALL(SCIPtightenVarLbGlobal(self._scip, var.scip_var, lb, force, &infeasible, &tightened))
        return infeasible, tightened

    def chgVarLb(self, Variable var, lb):
        """
        Changes the lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLb(self._scip, var.scip_var, lb))

    def chgVarUb(self, Variable var, ub):
        """Changes the upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUb(self._scip, var.scip_var, ub))

    def chgVarLbGlobal(self, Variable var, lb):
        """Changes the global lower bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbGlobal(self._scip, var.scip_var, lb))

    def chgVarUbGlobal(self, Variable var, ub):
        """Changes the global upper bound of the specified variable.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbGlobal(self._scip, var.scip_var, ub))

    def chgVarLbNode(self, Node node, Variable var, lb):
        """Changes the lower bound of the specified variable at the given node.

        Parameters
        ----------
        node : Node
            Node at which the variable bound will be changed
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """

        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbNode(self._scip, node.scip_node, var.scip_var, lb))

    def chgVarUbNode(self, Node node, Variable var, ub):
        """Changes the upper bound of the specified variable at the given node.

        Parameters
        ----------
        node : Node
            Node at which the variable bound will be changed
        var : Variable
            variable to change bound of
        lb : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbNode(self._scip, node.scip_node, var.scip_var, ub))

    def getVarDict(self, transformed=False):
        """
        Gets dictionary with variables names as keys and current variable values as items.

        Parameters
        ----------
        transformed : bool, optional
            Get transformed variables instead of original (Default value = False)

        Returns
        -------
        dict of str to float

        """
        var_dict = {}
        for var in self.getVars(transformed=transformed):
            var_dict[var.name] = self.getVal(var)
        return var_dict

    def updateNodeLowerbound(self, Node node, lb):
        """
        If given value is larger than the node's lower bound (in transformed problem),
        sets the node's lower bound to the new value.

        Parameters
        ----------
        node : Node
            the node to update
        lb : float
            new bound (if greater) for the node

        """
        PY_SCIP_CALL(SCIPupdateNodeLowerbound(self._scip, node.scip_node, lb))

    # LP Methods
    def getLPSolstat(self):
        """
        Gets solution status of current LP.

        Returns
        -------
        SCIP_LPSOLSTAT

        """
        return SCIPgetLPSolstat(self._scip)

    def getLPObjVal(self):
        """
        Gets objective value of current LP (which is the sum of column and loose objective value).

        Returns
        -------
        float

        """

        return SCIPgetLPObjval(self._scip)

    def getLPColsData(self):
        """
        Retrieve current LP columns.

        Returns
        -------
        list of Column

        """
        cdef SCIP_COL** cols
        cdef int ncols
        cdef int i

        PY_SCIP_CALL(SCIPgetLPColsData(self._scip, &cols, &ncols))

        return [Column.create(cols[i]) for i in range(ncols)]

    def getLPRowsData(self):
        """
        Retrieve current LP rows.

        Returns
        -------
        list of Row

        """
        cdef SCIP_ROW** rows
        cdef int nrows
        cdef int i

        PY_SCIP_CALL(SCIPgetLPRowsData(self._scip, &rows, &nrows))

        return [Row.create(rows[i]) for i in range(nrows)]

    def getLPBInvRow(self, row):
        """
        Gets a row from the inverse basis matrix B^-1

        Parameters
        ----------
        row : int
            The row index of the inverse basis matrix

        Returns
        -------
        list of float

        """
        # TODO: sparsity information
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBInvRow(self._scip, row, coefs, NULL, NULL))
        result = [coefs[i] for i in range(nrows)]
        free(coefs)

        return result

    def getLPBInvARow(self, row):
        """
        Gets a row from B^-1 * A.

        Parameters
        ----------
        row : int
            The row index of the inverse basis matrix multiplied by the coefficient matrix

        Returns
        -------
        list of float

        """
        # TODO: sparsity information
        cdef int ncols = SCIPgetNLPCols(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBInvARow(self._scip, row, NULL, coefs, NULL, NULL))
        result = [coefs[i] for i in range(ncols)]
        free(coefs)

        return result

    # LP Col Methods
    def getColRedCost(self, Column col):
        """
        Gets the reduced cost of the column in the current LP.

        Parameters
        ----------
        col : Column

        Returns
        -------
        float

        """
        return SCIPgetColRedcost(self._scip, col.scip_col)

    #TODO: documentation!!
    # LP Row Methods
    def createEmptyRowSepa(self, Sepa sepa, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        """
        Creates and captures an LP row without any coefficients from a separator.

        Parameters
        ----------
        sepa : Sepa
            separator that creates the row
        name : str, optional
            name of row (Default value = "row")
        lhs : float or None, optional
            left hand side of row (Default value = 0)
        rhs : float or None, optional
            right hand side of row (Default value = None)
        local : bool, optional
            is row only valid locally? (Default value = True)
        modifiable : bool, optional
            is row modifiable during node processing (subject to column generation)? (Default value = False)
        removable : bool, optional
            should the row be removed from the LP due to aging or cleanup? (Default value = True)

        Returns
        -------
        Row

        """
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        scip_sepa = SCIPfindSepa(self._scip, str_conversion(sepa.name))
        PY_SCIP_CALL(SCIPcreateEmptyRowSepa(self._scip, &row, scip_sepa, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def createEmptyRowUnspec(self, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        """
        Creates and captures an LP row without any coefficients from an unspecified source.

        Parameters
        ----------
        name : str, optional
            name of row (Default value = "row")
        lhs : float or None, optional
            left hand side of row (Default value = 0)
        rhs : float or None, optional
            right hand side of row (Default value = None)
        local : bool, optional
            is row only valid locally? (Default value = True)
        modifiable : bool, optional
            is row modifiable during node processing (subject to column generation)? (Default value = False)
        removable : bool, optional
            should the row be removed from the LP due to aging or cleanup? (Default value = True)

        Returns
        -------
        Row

        """
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        PY_SCIP_CALL(SCIPcreateEmptyRowUnspec(self._scip, &row, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def getRowActivity(self, Row row):
        """
        Returns the activity of a row in the last LP or pseudo solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIPgetRowActivity(self._scip, row.scip_row)

    def getRowLPActivity(self, Row row):
        """
        Returns the activity of a row in the last LP solution.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIPgetRowLPActivity(self._scip, row.scip_row)

    # TODO: do we need this? (also do we need release var??)
    def releaseRow(self, Row row not None):
        """
        Decreases usage counter of LP row, and frees memory if necessary.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPreleaseRow(self._scip, &row.scip_row))

    def cacheRowExtensions(self, Row row not None):
        """
        Informs row that all subsequent additions of variables to the row
        should be cached and not directly applied;
        after all additions were applied, flushRowExtensions() must be called;
        while the caching of row extensions is activated, information methods of the
        row give invalid results; caching should be used, if a row is build with addVarToRow()
        calls variable by variable to increase the performance.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPcacheRowExtensions(self._scip, row.scip_row))

    def flushRowExtensions(self, Row row not None):
        """
        Flushes all cached row extensions after a call of cacheRowExtensions()
        and merges coefficients with equal columns into a single coefficient

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPflushRowExtensions(self._scip, row.scip_row))

    def addVarToRow(self, Row row not None, Variable var not None, value):
        """
        Resolves variable to columns and adds them with the coefficient to the row.

        Parameters
        ----------
        row : Row
            Row in which the variable will be added
        var : Variable
            Variable which will be added to the row
        value : float
            Coefficient on the variable when placed in the row

        """
        PY_SCIP_CALL(SCIPaddVarToRow(self._scip, row.scip_row, var.scip_var, value))

    def printRow(self, Row row not None):
        """
        Prints row.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPprintRow(self._scip, row.scip_row, NULL))

    def getRowNumIntCols(self, Row row):
        """
        Returns number of intergal columns in the row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        int

        """
        return SCIPgetRowNumIntCols(self._scip, row.scip_row)

    def getRowObjParallelism(self, Row row):
        """
        Returns 1 if the row is parallel, and 0 if orthogonal.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIPgetRowObjParallelism(self._scip, row.scip_row)

    def getRowParallelism(self, Row row1, Row row2, orthofunc=101):
        """
        Returns the degree of parallelism between hyplerplanes. 1 if perfectly parallel, 0 if orthogonal.
        For two row vectors v, w the parallelism is calculated as: abs(v*w)/(abs(v)*abs(w)).
        101 in this case is an 'e' (euclidean) in ASCII. The other acceptable input is 100 (d for discrete).

        Parameters
        ----------
        row1 : Row
        row2 : Row
        orthofunc : int, optional
            101 (default) is an 'e' (euclidean) in ASCII. Alternate value is 100 (d for discrete)

        Returns
        -------
        float

        """
        return SCIProwGetParallelism(row1.scip_row, row2.scip_row, orthofunc)

    def getRowDualSol(self, Row row):
        """
        Gets the dual LP solution of a row.

        Parameters
        ----------
        row : Row

        Returns
        -------
        float

        """
        return SCIProwGetDualsol(row.scip_row)

    # Cutting Plane Methods
    def addPoolCut(self, Row row not None):
        """
        If not already existing, adds row to global cut pool.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPaddPoolCut(self._scip, row.scip_row))

    def getCutEfficacy(self, Row cut not None, Solution sol = None):
        """
        Returns efficacy of the cut with respect to the given primal solution or the
        current LP solution: e = -feasibility/norm

        Parameters
        ----------
        cut : Row
        sol : Solution or None, optional

        Returns
        -------
        float

        """
        return SCIPgetCutEfficacy(self._scip, NULL if sol is None else sol.sol, cut.scip_row)

    def isCutEfficacious(self, Row cut not None, Solution sol = None):
        """
        Returns whether the cut's efficacy with respect to the given primal solution or the
        current LP solution is greater than the minimal cut efficacy.

        Parameters
        ----------
        cut : Row
        sol : Solution or None, optional

        Returns
        -------
        float

        """
        return SCIPisCutEfficacious(self._scip, NULL if sol is None else sol.sol, cut.scip_row)

    def getCutLPSolCutoffDistance(self, Row cut not None, Solution sol not None):
        """
        Returns row's cutoff distance in the direction of the given primal solution.

        Parameters
        ----------
        cut : Row
        sol : Solution

        Returns
        -------
        float

        """
        return SCIPgetCutLPSolCutoffDistance(self._scip, sol.sol, cut.scip_row)

    def addCut(self, Row cut not None, forcecut = False):
        """
        Adds cut to separation storage and returns whether cut has been detected to be infeasible for local bounds.

        Parameters
        ----------
        cut : Row
            The cut that will be added
        forcecut : bool, optional
            Whether the cut should be forced or not, i.e., selected no matter what

        Returns
        -------
        infeasible : bool
            Whether the cut has been detected to be infeasible from local bounds

        """
        cdef SCIP_Bool infeasible
        PY_SCIP_CALL(SCIPaddRow(self._scip, cut.scip_row, forcecut, &infeasible))
        return infeasible

    def _createConsLinear(self, ExprCons lincons, **kwargs):
        """
        The function for creating a linear constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        lincons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        assert isinstance(lincons, ExprCons), "given constraint is not ExprCons but %s" % lincons.__class__.__name__

        assert lincons.expr.degree() <= 1, "given constraint is not linear, degree == %d" % lincons.expr.degree()
        terms = lincons.expr.terms

        cdef int nvars = len(terms.items())
        cdef SCIP_VAR** vars_array = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        cdef SCIP_Real* coeffs_array = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))
        cdef SCIP_CONS* scip_cons
        cdef SCIP_Real coeff
        cdef int i
        cdef _VarArray wrapper

        for i, (key, coeff) in enumerate(terms.items()):
            wrapper = _VarArray(key[0])
            vars_array[i] = wrapper.ptr[0]
            coeffs_array[i] = <SCIP_Real>coeff

        PY_SCIP_CALL(SCIPcreateConsLinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']), nvars, vars_array, coeffs_array,
            kwargs['lhs'], kwargs['rhs'], kwargs['initial'],
            kwargs['separate'], kwargs['enforce'], kwargs['check'],
            kwargs['propagate'], kwargs['local'], kwargs['modifiable'],
            kwargs['dynamic'], kwargs['removable'], kwargs['stickingatnode']))

        PyCons = Constraint.create(scip_cons)

        free(vars_array)
        free(coeffs_array)

        return PyCons

    def _createConsQuadratic(self, ExprCons quadcons, **kwargs):
        """
        The function for creating a quadratic constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        quadcons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2, "given constraint is not quadratic, degree == %d" % quadcons.expr.degree()

        cdef SCIP_CONS* scip_cons
        cdef SCIP_EXPR* prodexpr
        cdef _VarArray wrapper
        PY_SCIP_CALL(SCIPcreateConsQuadraticNonlinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL,        # linear
            0, NULL, NULL, NULL,  # quadratc
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable']))

        for v, c in terms.items():
            if len(v) == 1: # linear
                wrapper = _VarArray(v[0])
                PY_SCIP_CALL(SCIPaddLinearVarNonlinear(self._scip, scip_cons, wrapper.ptr[0], c))
            else: # nonlinear
                assert len(v) == 2, 'term length must be 1 or 2 but it is %s' % len(v)

                varexprs = <SCIP_EXPR**> malloc(2 * sizeof(SCIP_EXPR*))
                wrapper = _VarArray(v[0])
                PY_SCIP_CALL( SCIPcreateExprVar(self._scip, &varexprs[0], wrapper.ptr[0], NULL, NULL) )
                wrapper = _VarArray(v[1])
                PY_SCIP_CALL( SCIPcreateExprVar(self._scip, &varexprs[1], wrapper.ptr[0], NULL, NULL) )
                PY_SCIP_CALL( SCIPcreateExprProduct(self._scip, &prodexpr, 2, varexprs, 1.0, NULL, NULL) )

                PY_SCIP_CALL( SCIPaddExprNonlinear(self._scip, scip_cons, prodexpr, c) )

                PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &prodexpr) )
                PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &varexprs[1]) )
                PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &varexprs[0]) )
                free(varexprs)

        PyCons = Constraint.create(scip_cons)

        return PyCons

    def _createConsNonlinear(self, cons, **kwargs):
        """
        The function for creating a non-linear constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        cons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        cdef SCIP_EXPR* expr
        cdef SCIP_EXPR** varexprs
        cdef SCIP_EXPR** monomials
        cdef SCIP_CONS* scip_cons
        cdef _VarArray wrapper
        cdef int* idxs
        cdef int i
        cdef int j

        terms = cons.expr.terms

        # collect variables
        variables = {i: [var for var in term] for i, term in enumerate(terms)}

        # create monomials for terms
        monomials = <SCIP_EXPR**> malloc(len(terms) * sizeof(SCIP_EXPR*))
        termcoefs = <SCIP_Real*> malloc(len(terms) * sizeof(SCIP_Real))
        for i, (term, coef) in enumerate(terms.items()):
            wrapper = _VarArray(variables[i])

            PY_SCIP_CALL( SCIPcreateExprMonomial(self._scip, &monomials[i], wrapper.size, wrapper.ptr, NULL, NULL, NULL) )
            termcoefs[i] = <SCIP_Real>coef

        # create polynomial from monomials
        PY_SCIP_CALL( SCIPcreateExprSum(self._scip, &expr, <int>len(terms), monomials, termcoefs, 0.0, NULL, NULL))

        # create nonlinear constraint for expr
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip,
            &scip_cons,
            str_conversion(kwargs['name']),
            expr,
            kwargs['lhs'],
            kwargs['rhs'],
            kwargs['initial'],
            kwargs['separate'],
            kwargs['enforce'],
            kwargs['check'],
            kwargs['propagate'],
            kwargs['local'],
            kwargs['modifiable'],
            kwargs['dynamic'],
            kwargs['removable']) )

        PyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL( SCIPreleaseExpr(self._scip, &expr) )
        for i in range(<int>len(terms)):
            PY_SCIP_CALL(SCIPreleaseExpr(self._scip, &monomials[i]))
        free(monomials)
        free(termcoefs)

        return PyCons

    def _createConsGenNonlinear(self, cons, **kwargs):
        """
        The function for creating a general non-linear constraint, but not adding it to the Model.
        Please do not use this function directly, but rather use createConsFromExpr

        Parameters
        ----------
        cons : ExprCons
        kwargs : dict, optional

        Returns
        -------
        Constraint

        """
        cdef SCIP_EXPR* scip_expr
        cdef SCIP_CONS* scip_cons
        cdef GenExpr expr = cons.expr

        # recursively create the SCIP expression tree from python's expression tree
        scip_expr = expr.to_scip_expr(self._scip)

        # create nonlinear constraint for the expression root
        PY_SCIP_CALL(SCIPcreateConsNonlinear(
            self._scip,
            &scip_cons,
            str_conversion(kwargs["name"]),
            scip_expr,
            kwargs["lhs"],
            kwargs["rhs"],
            kwargs["initial"],
            kwargs["separate"],
            kwargs["enforce"],
            kwargs["check"],
            kwargs["propagate"],
            kwargs["local"],
            kwargs["modifiable"],
            kwargs["dynamic"],
            kwargs["removable"]))

        PY_SCIP_CALL(SCIPreleaseExpr(self._scip, &scip_expr))

        return Constraint.create(scip_cons)

    def createConsFromExpr(self, cons, name='', initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """
        Create a linear or nonlinear constraint without adding it to the SCIP problem.
        This is useful for creating disjunction constraints without also enforcing the individual constituents.
        Currently, this can only be used as an argument to `.addConsElemDisjunction`. To add
        an individual linear/nonlinear constraint, prefer `.addCons()`.

        Parameters
        ----------
        cons : ExprCons
            The expression constraint that is not yet an actual constraint
        name : str, optional
            the name of the constraint, generic name if empty (Default value = '')
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be  moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The created Constraint object.

        """
        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode
                      )

        kwargs['lhs'] = -SCIPinfinity(self._scip) if cons._lhs is None else cons._lhs
        kwargs['rhs'] =  SCIPinfinity(self._scip) if cons._rhs is None else cons._rhs

        deg = cons.expr.degree()
        if deg <= 1:
            return self._createConsLinear(cons, **kwargs)
        elif deg <= 2:
            return self._createConsQuadratic(cons, **kwargs)
        elif deg == INFINITY: # general nonlinear
            return self._createConsGenNonlinear(cons, **kwargs)
        else:
            return self._createConsNonlinear(cons, **kwargs)

    # Constraint functions
    def addCons(self, cons, name='', initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """
        Add a linear or nonlinear constraint.

        Parameters
        ----------
        cons : ExprCons
            the constraint expression to add to the model (e.g., x + y <= 5) 
        name : str, optional
            the name of the constraint, generic name if empty (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraints always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The created and added Constraint object.

        """
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        cdef SCIP_CONS* scip_cons

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode
                      )
        #  we have to pass this back to a SCIP_CONS*
        # object to create a new python constraint & handle constraint release
        # correctly. Otherwise, segfaults when trying to query information
        # about the created constraint later.
        pycons_initial = self.createConsFromExpr(cons, **kwargs)
        scip_cons = (<Constraint>pycons_initial).scip_cons

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pycons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pycons

    def addConsDisjunction(self, conss, name = '', initial = True,
        relaxcons = None, enforce=True, check =True,
        local=False, modifiable = False, dynamic = False):
        """
        Add a disjunction constraint.

        Parameters
        ----------
        conss : iterable of ExprCons
            An iterable of constraint objects to be included initially in the disjunction.
            Currently, these must be expressions.
        name : str, optional
            the name of the disjunction constraint.
        initial : bool, optional
            should the LP relaxation of disjunction constraint be in the initial LP? (Default value = True)
        relaxcons : None, optional
            a conjunction constraint containing the linear relaxation of the disjunction constraint, or None.
            NOT YET SUPPORTED. (Default value = None)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)

        Returns
        -------
        Constraint
            The created disjunction constraint

        """
        cdef SCIP_EXPR* scip_expr
        cdef SCIP_CONS* scip_cons
        cdef SCIP_CONS* disj_cons
        cdef int n_conss
        cdef int i

        def ensure_iterable(elem, length):
            if isinstance(elem, Iterable):
                return elem
            else:
                return list(repeat(elem, length))
        assert isinstance(conss, Iterable), "Given constraint list is not iterable"

        conss = list(conss)
        n_conss = len(conss)

        PY_SCIP_CALL(SCIPcreateConsDisjunction(
            self._scip, &disj_cons, str_conversion(name), 0, &scip_cons, NULL,
            initial, enforce, check, local, modifiable, dynamic
        ))


        # TODO add constraints to disjunction
        for i, cons in enumerate(conss):
            pycons = self.createConsFromExpr(cons, name=name, initial = initial,
                                            enforce=enforce, check=check,
                                            local=local, modifiable=modifiable, dynamic=dynamic
                                            )
            PY_SCIP_CALL(SCIPaddConsElemDisjunction(self._scip,disj_cons, (<Constraint>pycons).scip_cons))
            PY_SCIP_CALL(SCIPreleaseCons(self._scip, &(<Constraint>pycons).scip_cons))
        PY_SCIP_CALL(SCIPaddCons(self._scip, disj_cons))
        PyCons = self._getOrCreateCons(disj_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &disj_cons))

        return PyCons

    def addConsElemDisjunction(self, Constraint disj_cons, Constraint cons):
        """
        Appends a constraint to a disjunction.

        Parameters
        ----------
        disj_cons : Constraint
             the disjunction constraint to append to.
        cons : Constraint
            the constraint to append

        Returns
        -------
        disj_cons : Constraint
            The disjunction constraint with `cons` appended.

        """
        PY_SCIP_CALL(SCIPaddConsElemDisjunction(self._scip, (<Constraint>disj_cons).scip_cons, (<Constraint>cons).scip_cons))
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &(<Constraint>cons).scip_cons))
        return disj_cons

    def addMatrixConsDisjunction(self, conss: Iterable,
        name: Union[str, np.ndarray] = '', initial: Union[bool, np.ndarray] = True,
        relaxcons=None, enforce: Union[bool, np.ndarray] = True,
        check: Union[bool, np.ndarray] = True, local: Union[bool, np.ndarray] = False,
        modifiable: Union[bool, np.ndarray] = False,
        dynamic: Union[bool, np.ndarray] = False):
        """
        Add an elementwise disjunction of matrix constraint expressions.

        Given an iterable of ``MatrixExprCons`` with a common shape, creates one
        disjunction per index: for each position ``idx``, the resulting
        disjunction enforces that at least one of ``conss[k][idx]`` holds.
        ``ExprCons`` entries are broadcast to every position. If every entry in
        ``conss`` is a plain ``ExprCons``, the call is forwarded to
        :meth:`addConsDisjunction` and returns a single ``Constraint``.

        Note: this method accepts constraint *expressions* (e.g., ``A @ x <= b``),
        not ``MatrixConstraint``/``Constraint`` objects returned by
        :meth:`addMatrixCons`/:meth:`addCons`.

        Parameters
        ----------
        conss : iterable of MatrixExprCons or ExprCons
            Constraint expressions to combine elementwise into disjunctions. All
            ``MatrixExprCons`` entries must share the same shape.
        name : str or np.ndarray, optional
            Name of the disjunction constraints. (Default value = "")
        initial : bool or np.ndarray, optional
            Should the LP relaxation be in the initial LP? (Default value = True)
        relaxcons : None, optional
            NOT YET SUPPORTED. (Default value = None)
        enforce : bool or np.ndarray, optional
            Should the constraint be enforced during node processing? (Default value = True)
        check : bool or np.ndarray, optional
            Should the constraint be checked for feasibility? (Default value = True)
        local : bool or np.ndarray, optional
            Is the constraint only valid locally? (Default value = False)
        modifiable : bool or np.ndarray, optional
            Is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool or np.ndarray, optional
            Is the constraint subject to aging? (Default value = False)

        Returns
        -------
        MatrixConstraint or Constraint
            A ``MatrixConstraint`` of the common shape when any entry of
            ``conss`` is a ``MatrixExprCons``; otherwise a single
            ``Constraint`` from the scalar fallback.
        """
        if not isinstance(conss, Iterable):
            raise TypeError(f"given constraint list is not iterable, but got {type(conss).__name__}")
        conss = list(conss)

        for cons in conss:
            if not isinstance(cons, (ExprCons, MatrixExprCons)):
                raise TypeError(
                    "element of conss is not MatrixExprCons nor ExprCons but %s"
                    % cons.__class__.__name__
                )

        if all(isinstance(cons, ExprCons) for cons in conss):
            return self.addConsDisjunction(
                conss, name=name, initial=initial, relaxcons=relaxcons,
                enforce=enforce, check=check, local=local,
                modifiable=modifiable, dynamic=dynamic,
            )

        shape = None
        for cons in conss:
            if isinstance(cons, MatrixExprCons):
                if shape is None:
                    shape = cons.shape
                elif cons.shape != shape:
                    raise ValueError(
                        "All MatrixExprCons in conss must have the same shape, "
                        "got %s and %s" % (shape, cons.shape)
                    )

        for arr in (name, initial, enforce, check, local, modifiable, dynamic):
            if isinstance(arr, np.ndarray):
                assert arr.shape == shape

        if isinstance(name, str):
            matrix_names = np.full(shape, name, dtype=object)
            if name != "":
                for idx in np.ndindex(shape):
                    matrix_names[idx] = f"{name}_{'_'.join(map(str, idx))}"
        else:
            matrix_names = name

        matrix_initial = initial if isinstance(initial, np.ndarray) else np.full(shape, initial, dtype=bool)
        matrix_enforce = enforce if isinstance(enforce, np.ndarray) else np.full(shape, enforce, dtype=bool)
        matrix_check = check if isinstance(check, np.ndarray) else np.full(shape, check, dtype=bool)
        matrix_local = local if isinstance(local, np.ndarray) else np.full(shape, local, dtype=bool)
        matrix_modifiable = modifiable if isinstance(modifiable, np.ndarray) else np.full(shape, modifiable, dtype=bool)
        matrix_dynamic = dynamic if isinstance(dynamic, np.ndarray) else np.full(shape, dynamic, dtype=bool)

        matrix_cons = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            elem_conss = [
                cons[idx] if isinstance(cons, MatrixExprCons) else cons
                for cons in conss
            ]
            matrix_cons[idx] = self.addConsDisjunction(
                elem_conss,
                name=matrix_names[idx],
                initial=matrix_initial[idx],
                relaxcons=relaxcons,
                enforce=matrix_enforce[idx],
                check=matrix_check[idx],
                local=matrix_local[idx],
                modifiable=matrix_modifiable[idx],
                dynamic=matrix_dynamic[idx],
            )

        return matrix_cons.view(MatrixConstraint)

    def getConsVars(self, Constraint constraint):
        """
        Gets variables in a constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to get the variables from.

        Returns
        -------
        list of Variable

        """
        cdef SCIP_VAR** _vars
        cdef int nvars
        cdef SCIP_Bool success
        cdef int i

        SCIPgetConsNVars(self._scip, constraint.scip_cons, &nvars, &success)
        _vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))
        SCIPgetConsVars(self._scip, constraint.scip_cons, _vars, nvars, &success)

        if not success:
            free(_vars)
            return None

        vars = []
        for i in range(nvars):
            vars.append(self._getOrCreateVar(_vars[i]))

        free(_vars)
        return vars

    def getConsVals(self, Constraint constraint):
        """
        Returns the value array of an arbitrary SCIP constraint that can be represented as a single linear constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to get the values from.

        Returns
        -------
        list of float

        """
        cdef SCIP_Real* _vals
        cdef int nvars
        cdef SCIP_Bool success
        cdef int i

        nvars = self.getConsNVars(constraint)
        _vals = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))
        PY_SCIP_CALL(SCIPgetConsVals(self._scip, constraint.scip_cons, _vals, nvars, &success))

        if not success:
            free(_vals)
            return None

        vals = [_vals[i] for i in range(nvars)]
        free(_vals)
        return vals

    def getNVarsAnd(self, Constraint and_cons):
        """
        Gets number of variables in and constraint.

        Parameters
        ----------
        and_cons : Constraint
            AND constraint to get the number of variables from.

        Returns
        -------
        int

        """

        return SCIPgetNVarsAnd(self._scip, and_cons.scip_cons)

    def getVarsAnd(self, Constraint and_cons):
        """
        Gets variables in AND constraint.

        Parameters
        ----------
        and_cons : Constraint
            AND Constraint to get the variables from.

        Returns
        -------
        list of Variable

        """
        
        cdef SCIP_VAR** _vars
        cdef int nvars
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(and_cons.scip_cons))).decode('UTF-8')
        assert constype == 'and', "The constraint handler %s does not have this functionality." % constype

        nvars = SCIPgetNVarsAnd(self._scip, and_cons.scip_cons)
        _vars = SCIPgetVarsAnd(self._scip, and_cons.scip_cons)

        vars = []
        for i in range(nvars):
            vars.append(self._getOrCreateVar(_vars[i]))

        return vars

    def getResultantAnd(self, Constraint and_cons):
        """
        Gets the resultant variable in And constraint.

        Parameters
        ----------
        and_cons : Constraint
            Constraint to get the resultant variable from.

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* _resultant

        _resultant = SCIPgetResultantAnd(self._scip, and_cons.scip_cons)
        return self._getOrCreateVar(_resultant)

    def isAndConsSorted(self, Constraint and_cons):
        """
        Returns if the variables of the AND-constraint are sorted with respect to their indices.

        Parameters
        ----------
        and_cons : Constraint
            Constraint to check.

        Returns
        -------
        bool

        """
        
        return SCIPisAndConsSorted(self._scip, and_cons.scip_cons)

    def sortAndCons(self, Constraint and_cons):
        """
        Sorts the variables of the AND-constraint with respect to their indices.

        Parameters
        ----------
        and_cons : Constraint
            Constraint to sort.

        """

        PY_SCIP_CALL(SCIPsortAndCons(self._scip, and_cons.scip_cons))

    def getNVarsLogicor(self, Constraint logicor_cons):
        """
        Gets number of variables in logicor constraint.

        Parameters
        ----------
        logicor_cons : Constraint
            logicor constraint to get the number of variables from.

        Returns
        -------
        int

        """

        return SCIPgetNVarsLogicor(self._scip, logicor_cons.scip_cons)

    def getVarsLogicor(self, Constraint logicor_cons):
        """
        Gets variables in logicor constraint.

        Parameters
        ----------
        logicor_cons : Constraint
            logicor constraint to get the variables from.

        Returns
        -------
        list of Variable

        """

        cdef SCIP_VAR** _vars
        cdef int nvars
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(logicor_cons.scip_cons))).decode('UTF-8')
        assert constype == 'logicor', "The constraint handler %s does not have this functionality." % constype

        nvars = SCIPgetNVarsLogicor(self._scip, logicor_cons.scip_cons)
        _vars = SCIPgetVarsLogicor(self._scip, logicor_cons.scip_cons)

        vars = []
        for i in range(nvars):
            vars.append(self._getOrCreateVar(_vars[i]))

        return vars

    def addCoefLogicor(self, Constraint logicor_cons, Variable var):
        """
        Adds a variable to a logicor constraint.

        Parameters
        ----------
        logicor_cons : Constraint
            logicor constraint to add the variable to.
        var : Variable
            BINARY variable to add.

        """

        PY_SCIP_CALL(SCIPaddCoefLogicor(self._scip, logicor_cons.scip_cons, var.scip_var))

    def getDualsolLogicor(self, Constraint logicor_cons):
        """
        Gets the dual solution of a logicor constraint in the current LP.

        Parameters
        ----------
        logicor_cons : Constraint
            logicor constraint to get the dual solution from.

        Returns
        -------
        float
            Zero if presolve removed the transformed constraint.

        """
        cdef SCIP_CONS* transcons = self._getTransformedCons(logicor_cons, "logicor")
        if transcons == NULL:
            return 0.0

        return SCIPgetDualsolLogicor(self._scip, transcons)

    def getDualfarkasLogicor(self, Constraint logicor_cons):
        """
        Gets the dual Farkas value of a logicor constraint in the current infeasible LP.

        Parameters
        ----------
        logicor_cons : Constraint
            logicor constraint to get the dual Farkas value from.

        Returns
        -------
        float
            Zero if presolve removed the transformed constraint.

        """
        cdef SCIP_CONS* transcons = self._getTransformedCons(logicor_cons, "logicor")
        if transcons == NULL:
            return 0.0

        return SCIPgetDualfarkasLogicor(self._scip, transcons)

    def addExprNonlinear(self, Constraint cons, expr, coef):
        """
        Add coef*expr to nonlinear constraint.

        Parameters
        ----------
        cons : Constraint
        expr : Expr or GenExpr
        coef : float

        """
        assert self.getStage() == 1, "addExprNonlinear cannot be called in stage %i." % self.getStage()
        assert cons.isNonlinear(), "addExprNonlinear can only be called with nonlinear constraints."

        cdef Constraint temp_cons
        cdef SCIP_EXPR* scip_expr

        temp_cons = self.addCons(expr <= 0)
        scip_expr = SCIPgetExprNonlinear(temp_cons.scip_cons)

        PY_SCIP_CALL(SCIPaddExprNonlinear(self._scip, cons.scip_cons, scip_expr, coef))
        self.delCons(temp_cons)

    def addConsCoeff(self, Constraint cons, Variable var, coeff):
        """
        Add coefficient to the constraint (if non-zero).

        Parameters
        ----------
        cons : Constraint
            Constraint to be changed
        var : Variable
            variable to be added
        coeff : float
            coefficient of new variable

        """

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, cons.scip_cons, var.scip_var, coeff))
        elif constype == 'knapsack':
            PY_SCIP_CALL(SCIPaddCoefKnapsack(self._scip, cons.scip_cons, var.scip_var, coeff))
        else:
            raise NotImplementedError("Adding coefficients to %s constraints is not implemented." % constype)

    def addConsNode(self, Node node, ExprCons cons, Node validnode=None, name='',
                    initial=True, separate=True, enforce=True, check=True,
                    propagate=True, local=True, dynamic=False, removable=True,
                    stickingatnode=True):
        """
        Add a constraint to the given node.

        Parameters
        ----------
        node : Node
            node at which the constraint will be added
        cons : ExprCons
            the constraint expression to add to the node (e.g., x + y <= 5)
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)
        name : str, optional
            name of the constraint (Default value = '')
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = True)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = True)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added? (Default value = True)

        Returns
        -------
        Constraint
            The added Constraint object.

        """
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        cdef SCIP_CONS* scip_cons

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check, propagate=propagate,
                      local=local, modifiable=False, dynamic=dynamic,
                      removable=removable, stickingatnode=stickingatnode)
        pycons_initial = self.createConsFromExpr(cons, **kwargs)
        scip_cons = (<Constraint>pycons_initial).scip_cons

        if isinstance(validnode, Node):
            PY_SCIP_CALL(SCIPaddConsNode(self._scip, node.scip_node, scip_cons, validnode.scip_node))
        else:
            PY_SCIP_CALL(SCIPaddConsNode(self._scip, node.scip_node, scip_cons, NULL))

        pycons = Constraint.create(scip_cons)
        pycons.data = (<Constraint>pycons_initial).data
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pycons

    def addConsLocal(self, ExprCons cons, Node validnode=None, name='',
                     initial=True, separate=True, enforce=True, check=True,
                     propagate=True, local=True, dynamic=False, removable=True,
                     stickingatnode=True):
        """
        Add a constraint to the current node.

        Parameters
        ----------
        cons : ExprCons
            the constraint expression to add to the current node (e.g., x + y <= 5)
        validnode : Node or None, optional
            more global node where cons is also valid. (Default=None)
        name : str, optional
            name of the constraint (Default value = '')
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = True)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = True)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added? (Default value = True)

        Returns
        -------
        Constraint
            The added Constraint object.

        """
        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        cdef SCIP_CONS* scip_cons

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check, propagate=propagate,
                      local=local, modifiable=False, dynamic=dynamic,
                      removable=removable, stickingatnode=stickingatnode)
        pycons_initial = self.createConsFromExpr(cons, **kwargs)
        scip_cons = (<Constraint>pycons_initial).scip_cons

        if isinstance(validnode, Node):
            PY_SCIP_CALL(SCIPaddConsLocal(self._scip, scip_cons, validnode.scip_node))
        else:
            PY_SCIP_CALL(SCIPaddConsLocal(self._scip, scip_cons, NULL))

        pycons = Constraint.create(scip_cons)
        pycons.data = (<Constraint>pycons_initial).data
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pycons

    def addConsLinear(self, vars, vals, lhs=None, rhs=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, modifiable=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add a linear constraint lhs <= sum(vals[i] * vars[i]) <= rhs directly from
        variables and coefficients, without building an expression. Faster than addCons
        for building large models.

        Parameters
        ----------
        vars : list of Variable
            variables in the constraint
        vals : list of float
            coefficients of the variables
        lhs : float or None, optional
            left-hand side, -infinity if None (Default value = None)
        rhs : float or None, optional
            right-hand side, infinity if None (Default value = None)
        name : str, optional
            name of the constraint, generic name if empty (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created linear constraint

        """
        cdef int nvars = len(vars)
        cdef int i
        cdef SCIP_Real* vals_array
        cdef SCIP_CONS* scip_cons
        cdef _VarArray wrapper

        if nvars != len(vals):
            raise ValueError("Number of variables and values must be the same.")

        if lhs is None:
            lhs = -SCIPinfinity(self._scip)
        if rhs is None:
            rhs = SCIPinfinity(self._scip)
        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        wrapper = _VarArray(vars)
        vals_array = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))
        for i in range(nvars):
            vals_array[i] = <SCIP_Real>vals[i]

        PY_SCIP_CALL(SCIPcreateConsLinear(
            self._scip, &scip_cons, str_conversion(name), nvars, wrapper.ptr, vals_array,
            lhs, rhs, initial, separate, enforce, check, propagate, local, modifiable,
            dynamic, removable, stickingatnode))

        free(vals_array)

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsKnapsack(self, vars, weights, capacity, name="",
                initial=True, separate=True, enforce=True, check=True,
                modifiable=False, propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of int
            list of weights
        capacity: int
            capacity of the knapsack
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created knapsack constraint        
        """

        cdef int nvars = len(vars)
        cdef int i
        cdef SCIP_Longint* weights_array = <SCIP_Longint*> malloc(nvars * sizeof(SCIP_Real))
        cdef SCIP_CONS* scip_cons
        cdef _VarArray wrapper

        assert nvars == len(weights), "Number of variables and weights must be the same."

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        wrapper = _VarArray(vars)
        for i in range(nvars):
            weights_array[i] = <SCIP_Longint>weights[i]

        PY_SCIP_CALL(SCIPcreateConsKnapsack(
            self._scip, &scip_cons, str_conversion(name), nvars, wrapper.ptr, weights_array,
            capacity, initial, separate, enforce, check, propagate, local, modifiable,
            dynamic, removable, stickingatnode))

        free(weights_array)

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))

        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsCumulative(self, vars, durations, demands, capacity, name="",
                initial=True, separate=True, enforce=True, check=True,
                modifiable=False, propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add a cumulative constraint.

        A cumulative constraint models resource-constrained scheduling: given jobs
        with start times, durations, and demands on a shared resource of fixed
        capacity, it ensures that at every time point t the total demand of all
        jobs active at t does not exceed the capacity. Job j is active at t if
        start[j] <= t < start[j] + duration[j].

        The start times are given as integer variables in `vars`. The `durations`,
        `demands`, and `capacity` arguments must be fixed integers. End times are
        implicit (start + duration); post separate constraints if explicit end
        variables are needed.

        If you simply want the jobs to not overlap, set all durations to '1'.

        Parameters
        ----------
        vars : list of Variable
            list of integer variables corresponding to job start times
        durations : list of int
            list of durations, one for each job
        demands : list of int
            list of demands, one for each job
        capacity : int
            available cumulative capacity at any time point
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created cumulative constraint
        """

        cdef int nvars = len(vars)
        cdef int i
        cdef int* durations_array = <int*> malloc(nvars * sizeof(int))
        cdef int* demands_array = <int*> malloc(nvars * sizeof(int))
        cdef SCIP_CONS* scip_cons
        cdef _VarArray wrapper

        assert nvars == len(durations) == len(demands), "Number of variables, durations, and demands must be the same."

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        wrapper = _VarArray(vars)
        for i in range(nvars):
            durations_array[i] = <int>durations[i]
            demands_array[i] = <int>demands[i]

        PY_SCIP_CALL(SCIPcreateConsCumulative(
            self._scip, &scip_cons, str_conversion(name), nvars, wrapper.ptr, durations_array,
            demands_array, capacity, initial, separate, enforce, check, propagate, local, modifiable,
            dynamic, removable, stickingatnode))

        free(durations_array)
        free(demands_array)

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))

        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsSOS1(self, vars, weights=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add an SOS1 constraint.

        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of float or None, optional
            list of weights (Default value = None)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created SOS1 constraint

        """
        cdef SCIP_CONS* scip_cons
        cdef int nvars
        cdef int i
        cdef _VarArray wrapper 

        PY_SCIP_CALL(SCIPcreateConsSOS1(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        nvars = len(vars)
        wrapper = _VarArray(vars)
        if weights is None:
            for i in range(nvars):
                PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, scip_cons, wrapper.ptr[i]))
        else:
            for i in range(nvars):
                PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, scip_cons, wrapper.ptr[i], weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsSOS2(self, vars, weights=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add an SOS2 constraint.

        Parameters
        ----------
        vars : list of Variable
            list of variables to be included
        weights : list of float or None, optional
            list of weights (Default value = None)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created SOS2 constraint

        """
        cdef SCIP_CONS* scip_cons
        cdef int nvars
        cdef int i
        cdef _VarArray wrapper

        PY_SCIP_CALL(SCIPcreateConsSOS2(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        nvars = len(vars)
        wrapper = _VarArray(vars)
        if weights is None:
            for i in range(nvars):
                PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, scip_cons, wrapper.ptr[i]))
        else:
            for i in range(nvars):
                PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, scip_cons, wrapper.ptr[i], weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsAnd(self, vars, resvar, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add an AND-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables to be included (operators)
        resvar : Variable
            BINARY variable (resultant)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created AND constraint

        """
        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars
        cdef SCIP_VAR* _resvar
        cdef _VarArray resvar_wrapper = _VarArray(resvar)
        cdef _VarArray vars_wrapper = _VarArray(vars)
        cdef SCIP_CONS* scip_cons
        cdef int i

        _resvar = resvar_wrapper.ptr[0]
        _vars = vars_wrapper.ptr

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsAnd(self._scip, &scip_cons, str_conversion(name), _resvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsOr(self, vars, resvar, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add an OR-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables to be included (operators)
        resvar : Variable
            BINARY variable (resultant)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created OR constraint

        """
        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars
        cdef SCIP_VAR* _resvar
        cdef _VarArray resvar_wrapper = _VarArray(resvar)
        cdef _VarArray vars_wrapper = _VarArray(vars)
        cdef SCIP_CONS* scip_cons
        cdef int i

        _resvar = resvar_wrapper.ptr[0]
        _vars = vars_wrapper.ptr

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsOr(self._scip, &scip_cons, str_conversion(name), _resvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsXor(self, vars, rhsvar, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add a XOR-constraint.

        Parameters
        ----------
        vars : list of Variable
            list of binary variables to be included (operators)
        rhsvar : bool
            BOOLEAN value, explicit True, False or bool(obj) is needed (right-hand side)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created XOR constraint

        """
        assert type(rhsvar) is type(bool()), "Provide BOOLEAN value as rhsvar, you gave %s." % type(rhsvar)

        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars
        cdef SCIP_CONS* scip_cons
        cdef int i
        cdef _VarArray wrapper = _VarArray(vars)

        _vars = wrapper.ptr

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsXor(self._scip, &scip_cons, str_conversion(name), rhsvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsLogicor(self, vars, name="",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """
        Add a logicor constraint: at least one of the given binary variables must be one.

        Parameters
        ----------
        vars : list of Variable
            list of BINARY variables, at least one of which must take value one
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        modifiable : bool, optional
            is the constraint modifiable (subject to column generation)? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created logicor constraint

        """
        cdef int nvars = len(vars)
        cdef SCIP_VAR** _vars
        cdef _VarArray vars_wrapper = _VarArray(vars)
        cdef SCIP_CONS* scip_cons

        _vars = vars_wrapper.ptr

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        PY_SCIP_CALL(SCIPcreateConsLogicor(self._scip, &scip_cons, str_conversion(name), nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsCardinality(self, consvars, cardval, indvars=None, weights=None, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """
        Add a cardinality constraint that allows at most 'cardval' many nonzero variables.

        Parameters
        ----------
        consvars : list of Variable
            list of variables to be included
        cardval : int
            nonnegative integer
        indvars : list of Variable or None, optional
            indicator variables indicating which variables may be treated as nonzero in
            cardinality constraint, or None if new indicator variables should be
            introduced automatically (Default value = None)
        weights : list of float or None, optional
            weights determining the variable order, or None if variables should be ordered
            in the same way they were added to the constraint (Default value = None)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created Cardinality constraint

        """
        cdef SCIP_VAR* indvar
        cdef SCIP_CONS* scip_cons
        cdef int i
        cdef _VarArray v_wrapper
        cdef _VarArray indvar_wrapper
        cdef SCIP_VAR* scip_var

        PY_SCIP_CALL(SCIPcreateConsCardinality(self._scip, &scip_cons, str_conversion(name), 0, NULL, cardval, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        # circumvent an annoying bug in SCIP 4.0.0 that does not allow uninitialized weights
        if weights is None:
            weights = list(range(1, len(consvars) + 1))

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        v_wrapper = _VarArray(consvars)
        for i, v in enumerate(consvars):
            scip_var = v_wrapper.ptr[i]
            if indvars:
                indvar_wrapper = _VarArray(indvars[i])
                indvar = indvar_wrapper.ptr[0]
            else:
                indvar = NULL

            if weights is None:
                PY_SCIP_CALL(SCIPappendVarCardinality(self._scip, scip_cons, scip_var, indvar))
            else:
                PY_SCIP_CALL(SCIPaddVarCardinality(self._scip, scip_cons, scip_var, indvar, <SCIP_Real>weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addConsIndicator(self, cons, binvar=None, activeone=True, name="",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an indicator constraint for the linear inequality `cons`.

        The `binvar` argument models the redundancy of the linear constraint. A solution for which
        `binvar` is 1 must satisfy the constraint.

        Parameters
        ----------
        cons : ExprCons
            a linear inequality
        binvar : Variable, optional
            binary indicator variable, or None if it should be created (Default value = None)
        activeone : bool, optional
            constraint should active if binvar is 1 (0 if activeone = False)
        name : str, optional
            name of the constraint (Default value = "")
        initial : bool, optional
            should the LP relaxation of constraint be in the initial LP? (Default value = True)
        separate : bool, optional
            should the constraint be separated during LP processing? (Default value = True)
        enforce : bool, optional
            should the constraint be enforced during node processing? (Default value = True)
        check : bool, optional
            should the constraint be checked for feasibility? (Default value = True)
        propagate : bool, optional
            should the constraint be propagated during node processing? (Default value = True)
        local : bool, optional
            is the constraint only valid locally? (Default value = False)
        dynamic : bool, optional
            is the constraint subject to aging? (Default value = False)
        removable : bool, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool, optional
            should the constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        Constraint
            The newly created Indicator constraint

        """
        cdef SCIP_VAR* _binVar
        cdef SCIP_CONS* scip_cons
        cdef SCIP_Real coeff
        cdef _VarArray wrapper

        assert isinstance(cons, ExprCons), "given constraint is not ExprCons but %s" % cons.__class__.__name__

        if cons._lhs is not None and cons._rhs is not None:
            raise ValueError("expected inequality that has either only a left or right hand side")

        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        if cons.expr.degree() > 1:
            raise ValueError("expected linear inequality, expression has degree %d" % cons.expr.degree())

        if cons._rhs is not None:
            rhs =  cons._rhs
            negate = False
        else:
            rhs = -cons._lhs
            negate = True

        if binvar is not None:
            wrapper = _VarArray(binvar)
            _binVar = wrapper.ptr[0]
            if not activeone:
                PY_SCIP_CALL(SCIPgetNegatedVar(self._scip, _binVar, &_binVar))
        else:
            _binVar = NULL

        PY_SCIP_CALL(SCIPcreateConsIndicator(self._scip, &scip_cons, str_conversion(name), _binVar, 0, NULL, NULL, rhs,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))
        terms = cons.expr.terms

        for key, coeff in terms.items():
            if negate:
                coeff = -coeff
            wrapper = _VarArray(key[0])
            PY_SCIP_CALL(SCIPaddVarIndicator(self._scip, scip_cons, wrapper.ptr[0], <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = self._getOrCreateCons(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addMatrixConsIndicator(self, cons: Union[ExprCons, MatrixExprCons], binvar: Union[Variable, MatrixVariable] = None,
                               activeone: Union[bool, np.ndarray] = True, name: Union[str, np.ndarray] = "",
                               initial: Union[bool, np.ndarray] = True, separate: Union[bool, np.ndarray] = True,
                               enforce: Union[bool, np.ndarray] = True, check: Union[bool, np.ndarray] = True,
                               propagate: Union[bool, np.ndarray] = True, local: Union[bool, np.ndarray] = False,
                               dynamic: Union[bool, np.ndarray] = False, removable: Union[bool, np.ndarray] = False,
                               stickingatnode: Union[bool, np.ndarray] = False) -> MatrixConstraint:
        """Add an indicator matrix constraint for the linear inequality `cons`.

        The `binvar` argument models the redundancy of the linear constraint. A solution
        for which `binvar` is 1 must satisfy the constraint.

        Parameters
        ----------
        cons : ExprCons or MatrixExprCons
            a linear inequality
        binvar : Variable or MatrixVariable, optional
            binary indicator variable / matrix variable, or None if it should be created. (Default value = None)
        activeone : bool or np.ndarray, optional
            the matrix constraint should be active if binvar is 1 (0 if activeone = False).
        name : str or np.ndarray, optional
            name of the matrix constraint.  (Default value = "")
        initial : bool or np.ndarray, optional
            should the LP relaxation of matrix constraint be in the initial LP?  (Default value = True)
        separate : bool or np.ndarray, optional
            should the matrix constraint be separated during LP processing?  (Default value = True)
        enforce : bool or np.ndarray, optional
            should the matrix constraint be enforced during node processing?  (Default value = True)
        check : bool or np.ndarray, optional
            should the matrix constraint be checked for feasibility?  (Default value = True)
        propagate : bool or np.ndarray, optional
            should the matrix constraint be propagated during node processing? (Default value = True)
        local : bool or np.ndarray, optional
            is the matrix constraint only valid locally? (Default value = False)
        dynamic : bool or np.ndarray, optional
            is the matrix constraint subject to aging? (Default value = False)
        removable : bool or np.ndarray, optional
            should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        stickingatnode : bool or np.ndarray, optional
            should the matrix constraint always be kept at the node where it was added,
            even if it may be moved to a more global node? (Default value = False)

        Returns
        -------
        MatrixConstraint
            The newly created Indicator MatrixConstraint object.
        """

        if not isinstance(cons, (ExprCons, MatrixExprCons)):
            raise TypeError("given constraint is not MatrixExprCons nor ExprCons but %s" % cons.__class__.__name__)

        if isinstance(cons, ExprCons):
            return self.addConsIndicator(cons, binvar=binvar, activeone=activeone, name=name,
                        initial=initial, separate=separate, enforce=enforce, check=check,
                        propagate=propagate, local=local, dynamic=dynamic, removable=removable,
                        stickingatnode=stickingatnode)

        shape = cons.shape

        if isinstance(binvar, MatrixVariable):
            assert binvar.shape == shape
        if isinstance(activeone, np.ndarray):
            assert activeone.shape == shape
        if isinstance(name, np.ndarray):
            assert name.shape == shape
        if isinstance(initial, np.ndarray):
            assert initial.shape == shape
        if isinstance(separate, np.ndarray):
            assert separate.shape == shape
        if isinstance(enforce, np.ndarray):
            assert enforce.shape == shape
        if isinstance(check, np.ndarray):
            assert check.shape == shape
        if isinstance(propagate, np.ndarray):
            assert propagate.shape == shape
        if isinstance(local, np.ndarray):
            assert local.shape == shape
        if isinstance(dynamic, np.ndarray):
            assert dynamic.shape == shape
        if isinstance(removable, np.ndarray):
            assert removable.shape == shape
        if isinstance(stickingatnode, np.ndarray):
            assert stickingatnode.shape == shape

        if not isinstance(binvar, MatrixVariable):
            matrix_binvar = np.full(shape, binvar, dtype=Variable)
        else:
            matrix_binvar = binvar

        if not isinstance(activeone, np.ndarray):
            matrix_activeone = np.full(shape, activeone, dtype=bool)
        else:
            matrix_activeone = activeone

        if isinstance(name, str):
            matrix_names = np.full(shape, name, dtype=object)
            if name != "":
                for idx in np.ndindex(shape):
                    matrix_names[idx] = f"{name}_{'_'.join(map(str, idx))}"
        else:
            matrix_names = name

        if not isinstance(initial, np.ndarray):
            matrix_initial = np.full(shape, initial, dtype=bool)
        else:
            matrix_initial = initial

        if not isinstance(enforce, np.ndarray):
            matrix_enforce = np.full(shape, enforce, dtype=bool)
        else:
            matrix_enforce = enforce

        if not isinstance(separate, np.ndarray):
            matrix_separate = np.full(shape, separate, dtype=bool)
        else:
            matrix_separate = separate

        if not isinstance(check, np.ndarray):
            matrix_check = np.full(shape, check, dtype=bool)
        else:
            matrix_check = check

        if not isinstance(propagate, np.ndarray):
            matrix_propagate = np.full(shape, propagate, dtype=bool)
        else:
            matrix_propagate = propagate

        if not isinstance(local, np.ndarray):
            matrix_local = np.full(shape, local, dtype=bool)
        else:
            matrix_local = local

        if not isinstance(dynamic, np.ndarray):
            matrix_dynamic = np.full(shape, dynamic, dtype=bool)
        else:
            matrix_dynamic = dynamic

        if not isinstance(removable, np.ndarray):
            matrix_removable = np.full(shape, removable, dtype=bool)
        else:
            matrix_removable = removable

        if not isinstance(stickingatnode, np.ndarray):
            matrix_stickingatnode = np.full(shape, stickingatnode, dtype=bool)
        else:
            matrix_stickingatnode = stickingatnode

        matrix_cons = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            matrix_cons[idx] = self.addConsIndicator(
                cons[idx],
                binvar=matrix_binvar[idx],
                activeone=matrix_activeone[idx],
                name=matrix_names[idx],
                initial=matrix_initial[idx],
                separate=matrix_separate[idx],
                enforce=matrix_enforce[idx],
                check=matrix_check[idx],
                propagate=matrix_propagate[idx],
                local=matrix_local[idx],
                dynamic=matrix_dynamic[idx],
                removable=matrix_removable[idx],
                stickingatnode=matrix_stickingatnode[idx],
            )

        return matrix_cons.view(MatrixConstraint)

    def getLinearConsIndicator(self, Constraint cons):
        """
        Get the linear constraint corresponding to the indicator constraint.

        Parameters
        ----------
        cons : Constraint
            The indicator constraint

        Returns
        -------
        Constraint or None
        """

        cdef SCIP_CONS* lincons = SCIPgetLinearConsIndicator(cons.scip_cons)
        if lincons == NULL:
            return None
        return self._getOrCreateCons(lincons)

    def getSlackVarIndicator(self, Constraint cons):
        """
        Get slack variable of an indicator constraint.


        Parameters
        ----------
        cons : Constraint
            The indicator constraint

        Returns
        -------
        Variable

        """
        cdef SCIP_VAR* var = SCIPgetSlackVarIndicator(cons.scip_cons)
        return self._getOrCreateVar(var)

    def addPyCons(self, Constraint cons):
        """
        Adds a customly created cons.

        Parameters
        ----------
        cons : Constraint
            constraint to add

        """
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons.scip_cons))
        Py_INCREF(cons)

    def addVarSOS1(self, Constraint cons, Variable var, weight):
        """
        Add variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            new variable
        weight : weight
            weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, cons.scip_cons, var.scip_var, weight))

    def appendVarSOS1(self, Constraint cons, Variable var):
        """
        Append variable to SOS1 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS1 constraint
        var : Variable
            variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, cons.scip_cons, var.scip_var))

    def addVarSOS2(self, Constraint cons, Variable var, weight):
        """
        Add variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            new variable
        weight : weight
            weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, cons.scip_cons, var.scip_var, weight))

    def appendVarSOS2(self, Constraint cons, Variable var):
        """
        Append variable to SOS2 constraint.

        Parameters
        ----------
        cons : Constraint
            SOS2 constraint
        var : Variable
            variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, cons.scip_cons, var.scip_var))

    def chgRhs(self, Constraint cons, rhs):
        """
        Change right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to change the right-hand side from
        rhs : float or None
            new right-hand side (set to None for +infinity)

        """

        if rhs is None:
           rhs = SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgRhsLinear(self._scip, cons.scip_cons, rhs))
        elif constype == 'nonlinear':
            PY_SCIP_CALL(SCIPchgRhsNonlinear(self._scip, cons.scip_cons, rhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def chgCapacityKnapsack(self, Constraint cons, capacity):
        """
        Change capacity of a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint to change the capacity from
        capacity : float or None
            new capacity (set to None for +infinity)

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'knapsack', "method cannot be called for constraints of type " + constype

        if capacity is None:
           capacity = SCIPinfinity(self._scip)

        PY_SCIP_CALL(SCIPchgCapacityKnapsack(self._scip, cons.scip_cons, capacity))

    def chgLhs(self, Constraint cons, lhs):
        """
        Change left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to change the left-hand side from
        lhs : float or None
            new left-hand side (set to None for -infinity)

        """

        if lhs is None:
           lhs = -SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgLhsLinear(self._scip, cons.scip_cons, lhs))
        elif constype == 'nonlinear':
            PY_SCIP_CALL(SCIPchgLhsNonlinear(self._scip, cons.scip_cons, lhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getRhs(self, Constraint cons):
        """
        Retrieve right-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to get the right-hand side from

        Returns
        -------
        float

        """
        cdef SCIP_Bool success
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')

        if cons.isLinearType():
            rhs = SCIPconsGetRhs(self._scip, cons.scip_cons, &success)
            assert(success)
            return rhs
        elif constype == 'nonlinear':
            return SCIPgetRhsNonlinear(cons.scip_cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getCapacityKnapsack(self, Constraint cons):
        """
        Retrieve capacity of a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint to get the capacity from

        Returns
        -------
        float

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        assert constype == 'knapsack', "method cannot be called for constraints of type " + constype

        return SCIPgetCapacityKnapsack(self._scip, cons.scip_cons)

    def getLhs(self, Constraint cons):
        """
        Retrieve left-hand side value of a constraint.

        Parameters
        ----------
        cons : Constraint
            linear or nonlinear constraint

        Returns
        -------
        float

        """
        cdef SCIP_Bool success
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')

        if cons.isLinearType():
            lhs = SCIPconsGetLhs(self._scip, cons.scip_cons, &success)
            assert(success)
            return lhs
        elif constype == 'nonlinear':
            return SCIPgetLhsNonlinear(cons.scip_cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def chgCoefLinear(self, Constraint cons, Variable var, value):
        """
        Changes coefficient of variable in linear constraint;
        deletes the variable if coefficient is zero; adds variable if not yet contained in the constraint
        This method may only be called during problem creation stage for an original constraint and variable.
        This method requires linear time to search for occurences of the variable in the constraint data.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry
        value : float
            new coefficient of constraint entry

        """
        PY_SCIP_CALL( SCIPchgCoefLinear(self._scip, cons.scip_cons, var.scip_var, value) )

    def delCoefLinear(self, Constraint cons, Variable var):
        """
        Deletes variable from linear constraint
        This method may only be called during problem creation stage for an original constraint and variable.
        This method requires linear time to search for occurrences of the variable in the constraint data.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry

        """

        PY_SCIP_CALL( SCIPdelCoefLinear(self._scip, cons.scip_cons, var.scip_var) )

    def addCoefLinear(self, Constraint cons, Variable var, value):
        """
        Adds coefficient to linear constraint (if it is not zero)

        Parameters
        ----------
        cons : Constraint
            linear constraint
        var : Variable
            variable of constraint entry
        value : float
            coefficient of constraint entry

        """

        PY_SCIP_CALL( SCIPaddCoefLinear(self._scip, cons.scip_cons, var.scip_var, value) )

    def addCoefKnapsack(self, Constraint cons, Variable var, weight):
        """
        Adds coefficient to knapsack constraint (if it is not zero)

        Parameters
        ----------
        cons : Constraint
            knapsack constraint
        var : Variable
            variable of constraint entry
        weight : float
            coefficient of constraint entry

        """

        PY_SCIP_CALL( SCIPaddCoefKnapsack(self._scip, cons.scip_cons, var.scip_var, weight) )

    def getActivity(self, Constraint cons, Solution sol = None):
        """
        Retrieve activity of given constraint.
        Can only be called after solving is completed.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        sol : Solution or None, optional
            solution to compute activity of, None to use current node's solution (Default value = None)

        Returns
        -------
        float

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol

        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            activity = SCIPgetActivityLinear(self._scip, cons.scip_cons, scip_sol)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        return activity

    def getSlack(self, Constraint cons, Solution sol = None, side = None):
        """
        Retrieve slack of given constraint.
        Can only be called after solving is completed.

        Parameters
        ----------
        cons : Constraint
            linear constraint
        sol : Solution or None, optional
            solution to compute slack of, None to use current node's solution (Default value = None)
        side : str or None, optional
            whether to use 'lhs' or 'rhs' for ranged constraints, None to return minimum (Default value = None)

        Returns
        -------
        float

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol


        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if constype == 'linear':
            lhs = SCIPgetLhsLinear(self._scip, cons.scip_cons)
            rhs = SCIPgetRhsLinear(self._scip, cons.scip_cons)
            activity = SCIPgetActivityLinear(self._scip, cons.scip_cons, scip_sol)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        lhsslack = activity - lhs
        rhsslack = rhs - activity

        if side == 'lhs':
            return lhsslack
        elif side == 'rhs':
            return rhsslack
        else:
            return min(lhsslack, rhsslack)

    def isNLPConstructed(self):
        """
        Returns whether SCIP's internal NLP has been constructed.

        Returns
        -------
        bool

        """
        return SCIPisNLPConstructed(self._scip)

    def getNNlRows(self):
        """
        Gets current number of nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        int

        """
        return SCIPgetNNLPNlRows(self._scip)

    def getNlRows(self):
        """
        Returns a list with the nonlinear rows in SCIP's internal NLP.

        Returns
        -------
        list of NLRow

        """
        cdef SCIP_NLROW** nlrows = SCIPgetNLPNlRows(self._scip)
        cdef int i

        return [NLRow.create(nlrows[i]) for i in range(self.getNNlRows())]

    def getNlRowSolActivity(self, NLRow nlrow, Solution sol = None):
        """
        Gives the activity of a nonlinear row for a given primal solution.

        Parameters
        ----------
        nlrow : NLRow
        sol : Solution or None, optional
            a primal solution, if None, then the current LP solution is used

        Returns
        -------
        float

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* solptr

        solptr = sol.sol if not sol is None else NULL
        PY_SCIP_CALL( SCIPgetNlRowSolActivity(self._scip, nlrow.scip_nlrow, solptr, &activity) )
        return activity

    def getNlRowSolFeasibility(self, NLRow nlrow, Solution sol = None):
        """
        Gives the feasibility of a nonlinear row for a given primal solution

        Parameters
        ----------
        nlrow : NLRow
        sol : Solution or None, optional
            a primal solution, if None, then the current LP solution is used

        Returns
        -------
        bool

        """
        cdef SCIP_Real feasibility
        cdef SCIP_SOL* solptr

        solptr = sol.sol if not sol is None else NULL
        PY_SCIP_CALL( SCIPgetNlRowSolFeasibility(self._scip, nlrow.scip_nlrow, solptr, &feasibility) )
        return feasibility

    def getNlRowActivityBounds(self, NLRow nlrow):
        """
        Gives the minimal and maximal activity of a nonlinear row w.r.t. the variable's bounds.

        Parameters
        ----------
        nlrow : NLRow

        Returns
        -------
        tuple of float

        """
        cdef SCIP_Real minactivity
        cdef SCIP_Real maxactivity

        PY_SCIP_CALL( SCIPgetNlRowActivityBounds(self._scip, nlrow.scip_nlrow, &minactivity, &maxactivity) )
        return (minactivity, maxactivity)

    def printNlRow(self, NLRow nlrow):
        """
        Prints nonlinear row.

        Parameters
        ----------
        nlrow : NLRow

        """
        PY_SCIP_CALL( SCIPprintNlRow(self._scip, nlrow.scip_nlrow, NULL) )

    def checkQuadraticNonlinear(self, Constraint cons):
        """
        Returns if the given constraint is quadratic.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bool

        """
        cdef SCIP_Bool isquadratic
        PY_SCIP_CALL( SCIPcheckQuadraticNonlinear(self._scip, cons.scip_cons, &isquadratic) )
        return isquadratic

    def getTermsQuadratic(self, Constraint cons):
        """
        Retrieve bilinear, quadratic, and linear terms of a quadratic constraint.

        Parameters
        ----------
        cons : Constraint

        Returns
        -------
        bilinterms : list of tuple
            Triples ``(var1, var2, coef)`` for terms of the form
            ``coef * var1 * var2`` with ``var1 != var2``.
        quadterms : list of tuple
            Triples ``(var, sqrcoef, lincoef)`` for variables that appear in
            quadratic or bilinear terms. ``sqrcoef`` is the coefficient of
            ``var**2``, and ``lincoef`` is the linear coefficient of ``var``
            if it also appears linearly.
        linterms : list of tuple
            Pairs ``(var, coef)`` for purely linear variables, i.e.,
            variables that do not participate in any quadratic or bilinear term.

        """
        cdef SCIP_EXPR* expr
        cdef int termidx

        # linear terms
        cdef SCIP_EXPR** linexprs
        cdef SCIP_Real* lincoefs
        cdef SCIP_Real lincoef
        cdef int nlinvars

        # bilinear terms
        cdef SCIP_EXPR* bilinterm1
        cdef SCIP_EXPR* bilinterm2
        cdef SCIP_Real bilincoef
        cdef int nbilinterms

        # quadratic terms
        cdef SCIP_EXPR* quadexpr
        cdef SCIP_EXPR* sqrexpr
        cdef SCIP_Real sqrcoef
        cdef int nquadterms

        # variables
        cdef SCIP_VAR* scipvar1
        cdef SCIP_VAR* scipvar2

        assert cons.isNonlinear(), "constraint is not nonlinear"
        assert self.checkQuadraticNonlinear(cons), "constraint is not quadratic"

        expr = SCIPgetExprNonlinear(cons.scip_cons)
        SCIPexprGetQuadraticData(expr, NULL, &nlinvars, &linexprs, &lincoefs,
                                 &nquadterms, &nbilinterms, NULL, NULL)

        linterms   = []
        bilinterms = []

        # Purely linear terms (variables not in any quadratic/bilinear term)
        for termidx in range(nlinvars):
            var = self._getOrCreateVar(SCIPgetVarExprVar(linexprs[termidx]))
            linterms.append((var, lincoefs[termidx]))

        # Collect quadratic terms in a dict so we can merge entries for the same variable.
        quaddict = {}  # var.ptr() -> [var, sqrcoef, lincoef]

        for termidx in range(nbilinterms):
            SCIPexprGetQuadraticBilinTerm(expr, termidx, &bilinterm1, &bilinterm2, &bilincoef, NULL, NULL)
            scipvar1 = SCIPgetVarExprVar(bilinterm1)
            scipvar2 = SCIPgetVarExprVar(bilinterm2)
            var1 = self._getOrCreateVar(scipvar1)
            var2 = self._getOrCreateVar(scipvar2)
            if scipvar1 != scipvar2:
                bilinterms.append((var1, var2, bilincoef))
            else:
                # Squared term reported as bilinear var*var
                key = var1.ptr()
                if key in quaddict:
                    quaddict[key][1] += bilincoef
                else: # TODO: SCIP handles expr like x**2 appropriately, but PySCIPOpt requires this. Need to investigate why.
                    quaddict[key] = [var1, bilincoef, 0.0]

        # Also collect linear coefficients from the quadratic terms
        for termidx in range(nquadterms):
            SCIPexprGetQuadraticQuadTerm(expr, termidx, &quadexpr, &lincoef, &sqrcoef, NULL, NULL, &sqrexpr)
            scipvar1 = SCIPgetVarExprVar(quadexpr)
            var = self._getOrCreateVar(scipvar1)
            key = var.ptr()
            if key in quaddict:
                quaddict[key][1] += sqrcoef
                quaddict[key][2] += lincoef
            else:
                quaddict[key] = [var, sqrcoef, lincoef]

        quadterms = [tuple(entry) for entry in quaddict.values()]

        return (bilinterms, quadterms, linterms)

    def setRelaxSolVal(self, Variable var, val):
        """
        Sets the value of the given variable in the global relaxation solution.

        Parameters
        ----------
        var : Variable
        val : float

        """
        PY_SCIP_CALL(SCIPsetRelaxSolVal(self._scip, NULL, var.scip_var, val))

    def getValsLinear(self, Constraint cons):
        """
        Retrieve the coefficients of a linear constraint

        Parameters
        ----------
        cons : Constraint
            linear constraint to get the coefficients of

        Returns
        -------
        dict of str to float

        """
        cdef SCIP_VAR** vars
        cdef SCIP_Real* vals
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("coefficients not available for constraints of type ", constype)

        vals = SCIPgetValsLinear(self._scip, cons.scip_cons)
        vars = SCIPgetVarsLinear(self._scip, cons.scip_cons)

        valsdict = {}
        for i in range(SCIPgetNVarsLinear(self._scip, cons.scip_cons)):
            valsdict[bytes(SCIPvarGetName(vars[i])).decode('utf-8')] = vals[i]

        return valsdict

    def getWeightsKnapsack(self, Constraint cons):
        """
        Retrieve the coefficients of a knapsack constraint

        Parameters
        ----------
        cons : Constraint
            knapsack constraint to get the coefficients of

        Returns
        -------
        dict of str to float

        """
        cdef SCIP_VAR** vars
        cdef SCIP_Longint* vals
        cdef int nvars
        cdef int i

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'knapsack':
            raise Warning("weights not available for constraints of type ", constype)

        nvars = SCIPgetNVarsKnapsack(self._scip, cons.scip_cons)
        vals = SCIPgetWeightsKnapsack(self._scip, cons.scip_cons)
        vars = SCIPgetVarsKnapsack(self._scip, cons.scip_cons)

        valsdict = {}
        for i in range(nvars):
            var_name = bytes(SCIPvarGetName(vars[i])).decode('utf-8')
            valsdict[var_name] = vals[i]

        return valsdict

    def getRowLinear(self, Constraint cons):
        """
        Retrieve the linear relaxation of the given linear constraint as a row.
        May return NULL if no LP row was yet created; the user must not modify the row!

        Parameters
        ----------
        cons : Constraint
            linear constraint to get the coefficients of

        Returns
        -------
        Row

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.scip_cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("coefficients not available for constraints of type ", constype)

        cdef SCIP_ROW* row = SCIPgetRowLinear(self._scip, cons.scip_cons)
        return Row.create(row)

    def getDualsolLinear(self, Constraint cons):
        """
        Retrieve the dual solution to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float
            Zero if presolve removed the transformed constraint.

        """
        cdef SCIP_CONS* transcons = self._getTransformedCons(cons, "linear")
        if transcons == NULL:
            return 0.0

        return SCIPgetDualsolLinear(self._scip, transcons)

    def getDualsolKnapsack(self, Constraint cons):
        """
        Retrieve the dual solution to a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint

        Returns
        -------
        float

        """
        cdef SCIP_CONS* transcons = self._getTransformedCons(cons, "knapsack")
        if transcons == NULL:
            return 0.0

        return SCIPgetDualsolKnapsack(self._scip, transcons)

    def getDualMultiplier(self, Constraint cons):
        """
        DEPRECATED: Retrieve the dual solution to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float

        """
        raise Warning("model.getDualMultiplier(cons) is deprecated: please use model.getDualsolLinear(cons)")

    def getDualfarkasLinear(self, Constraint cons):
        """
        Retrieve the dual farkas value to a linear constraint.

        Parameters
        ----------
        cons : Constraint
            linear constraint

        Returns
        -------
        float
            Zero if presolve removed the transformed constraint.

        """
        # TODO this should ideally be handled on the SCIP side
        cdef SCIP_CONS* transcons = self._getTransformedCons(cons, "linear")
        if transcons == NULL:
            return 0.0

        return SCIPgetDualfarkasLinear(self._scip, transcons)
    
    def getDualfarkasKnapsack(self, Constraint cons):
        """
        Retrieve the dual farkas value to a knapsack constraint.

        Parameters
        ----------
        cons : Constraint
            knapsack constraint

        Returns
        -------
        float

        """
        # TODO this should ideally be handled on the SCIP side
        cdef SCIP_CONS* transcons = self._getTransformedCons(cons, "knapsack")
        if transcons == NULL:
            return 0.0
            
        return SCIPgetDualfarkasKnapsack(self._scip, transcons)

    def getVarRedcost(self, Variable var):
        """
        Retrieve the reduced cost of a variable.

        Parameters
        ----------
        var : Variable
            variable to get the reduced cost of

        Returns
        -------
        float

        """
        redcost = None
        try:
            redcost = SCIPgetVarRedcost(self._scip, var.scip_var)
            if self.getObjectiveSense() == "maximize":
                redcost = -redcost
        except:
            raise Warning("no reduced cost available for variable " + var.name)
        return redcost

    def getDualSolVal(self, Constraint cons, boundconstraint=False):
        """
        Returns dual solution value of a constraint.

        Parameters
        ----------
        cons : Constraint
            constraint to get the dual solution value of
        boundconstraint : bool, optional
            Decides whether to store a bool if the constraint is a bound constraint
            (default = False)

        Returns
        -------
        float

        """
        cdef SCIP_Real _dualsol
        cdef SCIP_Bool _bounded

        if boundconstraint:
            SCIPgetDualSolVal(self._scip, cons.scip_cons, &_dualsol, &_bounded)
        else:
            SCIPgetDualSolVal(self._scip, cons.scip_cons, &_dualsol, NULL)

        return _dualsol

    def createCons(self, Conshdlr conshdlr, name, initial=True, separate=True, enforce=True, check=True, propagate=True,
                   local=False, modifiable=False, dynamic=False, removable=False, stickingatnode=False):
        """
        Create a constraint of a custom constraint handler.

        Parameters
        ----------
        conshdlr : Conshdlr
            constraint handler
        name : str
            name of constraint handler
        initial : bool, optional
            (Default value = True)
        separate : bool, optional
            (Default value = True)
        enforce : bool, optional
            (Default value = True)
        check : bool, optional
            (Default value = True)
        propagate : bool, optional
            (Default value = True)
        local : bool, optional
            (Default value = False)
        modifiable : bool, optional
            (Default value = False)
        dynamic : bool, optional
            (Default value = False)
        removable : bool, optional
            (Default value = False)
        stickingatnode : bool, optional
            (Default value = False)

        Returns
        -------
        Constraint

        """

        n = str_conversion(name)
        cdef SCIP_CONSHDLR* scip_conshdlr
        scip_conshdlr = SCIPfindConshdlr(self._scip, str_conversion(conshdlr.name))
        constraint = Constraint()
        PY_SCIP_CALL(SCIPcreateCons(self._scip, &(constraint.scip_cons), n, scip_conshdlr, <SCIP_CONSDATA*>constraint,
                                initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))
        return constraint

    def getLPBranchCands(self):
        """
        Gets branching candidates for LP solution branching (fractional variables) along with solution values,
        fractionalities, and number of branching candidates; The number of branching candidates does NOT account
        for fractional implicit integer variables which should not be used for branching decisions. Fractional
        implicit integer variables are stored at the positions nlpcands to nlpcands + nfracimplvars - 1
        branching rules should always select the branching candidate among the first npriolpcands of the candidate list

        Returns
        -------
        list of Variable
            list of variables of LP branching candidates
        list of float
            list of LP candidate solution values
        list of float
            list of LP candidate fractionalities
        int
            number of LP branching candidates
        int
            number of candidates with maximal priority
        int
            number of fractional implicit integer variables

        """
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac
        cdef int ncands
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars
        cdef int i

        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac,
                                          &nlpcands, &npriolpcands, &nfracimplvars))

        return ([self._getOrCreateVar(lpcands[i]) for i in range(nlpcands)], [lpcandssol[i] for i in range(nlpcands)],
                [lpcandsfrac[i] for i in range(nlpcands)], nlpcands, npriolpcands, nfracimplvars)

    def branchVarVal(self, Variable variable, value):
        """
        Branches on variable using a value which separates the domain of the variable.

        Parameters
        ----------
        variable : Variable
            Variable to branch on
        value : float
            value to branch on

        Returns
        -------
        Node
            Node created for the down (left) branch
        Node or None
            Node created for the equal child (middle child). Only exists if the branch variable is integer
        Node
            Node created for the up (right) branch

        """
        cdef SCIP_NODE* downchild
        cdef SCIP_NODE* eqchild
        cdef SCIP_NODE* upchild
        cdef _VarArray wrapper = _VarArray(variable)

        PY_SCIP_CALL(SCIPbranchVarVal(self._scip, wrapper.ptr[0], value, &downchild, &eqchild, &upchild))

        return Node.create(downchild), Node.create(eqchild), Node.create(upchild)

    def calcNodeselPriority(self, Variable variable, branchdir, targetvalue):
        """
        Calculates the node selection priority for moving the given variable's LP value
        to the given target value;
        this node selection priority can be given to the SCIPcreateChild() call.

        Parameters
        ----------
        variable : Variable
            variable on which the branching is applied
        branchdir : PY_SCIP_BRANCHDIR
            type of branching that was performed
        targetvalue : float
            new value of the variable in the child node

        Returns
        -------
        float
            node selection priority for moving the given variable's LP value to the given target value

        """
        return SCIPcalcNodeselPriority(self._scip, variable.scip_var, branchdir, targetvalue)

    def calcChildEstimate(self, Variable variable, targetvalue):
        """
        Calculates an estimate for the objective of the best feasible solution
        contained in the subtree after applying the given branching;
        this estimate can be given to the SCIPcreateChild() call.

        Parameters
        ----------
        variable : Variable
            Variable to compute the estimate for
        targetvalue : float
            new value of the variable in the child node

        Returns
        -------
        float
            objective estimate of the best solution in the subtree after applying the given branching

        """
        return SCIPcalcChildEstimate(self._scip, variable.scip_var, targetvalue)

    # Diving methods (Diving is LP related)
    def startDive(self):
        """Initiates LP diving.
        It allows the user to change the LP in several ways, solve, change again, etc,
        without affecting the actual LP. When endDive() is called,
        SCIP will undo all changes done and recover the LP it had before startDive."""
        PY_SCIP_CALL(SCIPstartDive(self._scip))

    def endDive(self):
        """Quits probing and resets bounds and constraints to the focus node's environment."""
        PY_SCIP_CALL(SCIPendDive(self._scip))

    def chgVarObjDive(self, Variable var, newobj):
        """
        Changes (column) variable's objective value in current dive.

        Parameters
        ----------
        var : Variable
        newobj : float

        """
        PY_SCIP_CALL(SCIPchgVarObjDive(self._scip, var.scip_var, newobj))

    def chgVarLbDive(self, Variable var, newbound):
        """
        Changes variable's current lb in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
        PY_SCIP_CALL(SCIPchgVarLbDive(self._scip, var.scip_var, newbound))

    def chgVarUbDive(self, Variable var, newbound):
        """
        Changes variable's current ub in current dive.

        Parameters
        ----------
        var : Variable
        newbound : float

        """
        PY_SCIP_CALL(SCIPchgVarUbDive(self._scip, var.scip_var, newbound))

    def getVarLbDive(self, Variable var):
        """
        Returns variable's current lb in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
        return SCIPgetVarLbDive(self._scip, var.scip_var)

    def getVarUbDive(self, Variable var):
        """
        Returns variable's current ub in current dive.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
        return SCIPgetVarUbDive(self._scip, var.scip_var)

    def chgRowLhsDive(self, Row row, newlhs):
        """
        Changes row lhs in current dive, change will be undone after diving
        ends, for permanent changes use SCIPchgRowLhs().

        Parameters
        ----------
        row : Row
        newlhs : float

        """
        PY_SCIP_CALL(SCIPchgRowLhsDive(self._scip, row.scip_row, newlhs))

    def chgRowRhsDive(self, Row row, newrhs):
        """
        Changes row rhs in current dive, change will be undone after diving
        ends. For permanent changes use SCIPchgRowRhs().

        Parameters
        ----------
        row : Row
        newrhs : float

        """
        PY_SCIP_CALL(SCIPchgRowRhsDive(self._scip, row.scip_row, newrhs))

    def addRowDive(self, Row row):
        """
        Adds a row to the LP in current dive.

        Parameters
        ----------
        row : Row

        """
        PY_SCIP_CALL(SCIPaddRowDive(self._scip, row.scip_row))

    def solveDiveLP(self, itlim = -1):
        """
        Solves the LP of the current dive. No separation or pricing is applied.

        Parameters
        ----------
        itlim : int, optional
            maximal number of LP iterations to perform (Default value = -1, that is, no limit)

        Returns
        -------
        lperror : bool
            whether an unresolved lp error occured
        cutoff : bool
            whether the LP was infeasible or the objective limit was reached

        """
        cdef SCIP_Bool lperror
        cdef SCIP_Bool cutoff

        PY_SCIP_CALL(SCIPsolveDiveLP(self._scip, itlim, &lperror, &cutoff))
        return lperror, cutoff

    def chgVarObjProbing(self, Variable var, newobj):
        """Changes (column) variable's objective value during probing mode."""
        PY_SCIP_CALL( SCIPchgVarObjProbing(self._scip, var.scip_var, newobj) )

    def chgVarLbProbing(self, Variable var, lb):
        """
        Changes the variable lower bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        lb : float or None
            new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLbProbing(self._scip, var.scip_var, lb))

    def chgVarUbProbing(self, Variable var, ub):
        """
        Changes the variable upper bound during probing mode.

        Parameters
        ----------
        var : Variable
            variable to change bound of
        ub : float or None
            new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUbProbing(self._scip, var.scip_var, ub))

    def fixVarProbing(self, Variable var, fixedval):
        """
        Fixes a variable at the current probing node.

        Parameters
        ----------
        var : Variable
        fixedval : float

        """
        PY_SCIP_CALL( SCIPfixVarProbing(self._scip, var.scip_var, fixedval) )

    def writeLP(self, filename="LP.lp"):
        """
        Writes current LP to a file.

        Parameters
        ----------
        filename : str, optional
            file name (Default value = "LP.lp")

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        absfile = str_conversion(abspath(filename))
        PY_SCIP_CALL( SCIPwriteLP(self._scip, absfile) )

        locale.setlocale(locale.LC_NUMERIC,user_locale)
    
    def writeMIP(self, filename, genericnames=False, origobj=False, lazyconss=False):
        """
        Writes MIP relaxation of the current branch-and-bound node to a file

        Parameters
        ----------
        filename : str
            name of the output file
        genericnames : bool, optional
            should generic names like x_i and row_j be used in order to avoid troubles with reserved symbols? (Default value = False)
        origobj : bool, optional
            should the original objective function be used (Default value = False)
        lazyconss : bool, optional
            output removable rows as lazy constraints? (Default value = False)
        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        absfile = str_conversion(abspath(filename))
        PY_SCIP_CALL(SCIPwriteMIP(self._scip, absfile, genericnames, origobj, lazyconss))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def createSol(self, Heur heur = None, initlp=False):
        """
        Create a new primal solution in the transformed space.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)
        initlp : bool, optional
            Should the created solution be initialised to the current LP solution instead of all zeros

        Returns
        -------
        Solution

        """
        cdef SCIP_SOL* _sol
        cdef SCIP_HEUR* _heur = self._getHeur(heur)

        if not initlp:
            PY_SCIP_CALL(SCIPcreateSol(self._scip, &_sol, _heur))
        else:
            PY_SCIP_CALL(SCIPcreateLPSol(self._scip, &_sol, _heur))
        solution = Solution.create(self._scip, _sol)
        return solution

    def createPartialSol(self, Heur heur = None):
        """
        Create a partial primal solution, initialized to unknown values.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)

        Returns
        -------
        Solution

        """
        cdef SCIP_SOL* _sol
        cdef SCIP_HEUR* _heur = self._getHeur(heur)

        PY_SCIP_CALL(SCIPcreatePartialSol(self._scip, &_sol, _heur))
        partialsolution = Solution.create(self._scip, _sol)
        return partialsolution

    def createOrigSol(self, Heur heur = None):
        """
        Create a new primal solution in the original space.

        Parameters
        ----------
        heur : Heur or None, optional
            heuristic that found the solution (Default value = None)

        Returns
        -------
        Solution

        """
        cdef SCIP_HEUR* _heur
        cdef SCIP_SOL* _sol

        _heur = self._getHeur(heur)

        PY_SCIP_CALL(SCIPcreateOrigSol(self._scip, &_sol, _heur))
        solution = Solution.create(self._scip, _sol)
        return solution

    def printBestSol(self, write_zeros=False):
        """
        Prints the best feasible primal solution.

        Parameters
        ----------
        write_zeros : bool, optional
            include variables that are set to zero (Default = False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPprintBestSol(self._scip, NULL, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def printSol(self, Solution solution=None, write_zeros=False):
        """
        Print the given primal solution.

        Parameters
        ----------
        solution : Solution or None, optional
            solution to print (default = None)
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        if solution is None:
            PY_SCIP_CALL(SCIPprintSol(self._scip, NULL, NULL, write_zeros))
        else:
            PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, NULL, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeBestSol(self, filename="origprob.sol", write_zeros=False):
        """
        Write the best feasible primal solution to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintBestSol(self._scip, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeBestTransSol(self, filename="transprob.sol", write_zeros=False):
        """
        Write the best feasible primal solution for the transformed problem to a file.

        Parameters
        ----------
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this double opening pattern to ensure that IOErrors are
        #   triggered early and in python not in C, Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintBestTransSol(self._scip, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeSol(self, Solution solution, filename="origprob.sol", write_zeros=False):
        """
        Write the given primal solution to a file.

        Parameters
        ----------
        solution : Solution
            solution to write
        filename : str, optional
            name of the output file (Default="origprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    def writeTransSol(self, Solution solution, filename="transprob.sol", write_zeros=False):
        """
        Write the given transformed primal solution to a file.

        Parameters
        ----------
        solution : Solution
            transformed solution to write
        filename : str, optional
            name of the output file (Default="transprob.sol")
        write_zeros : bool, optional
            include variables that are set to zero (Default=False)

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintTransSol(self._scip, solution.sol, cfile, write_zeros))

        locale.setlocale(locale.LC_NUMERIC,user_locale)

    # perhaps this should not be included as it implements duplicated functionality
    #   (as does it's namesake in SCIP)
    def readSol(self, filename):
        """
        Reads a given solution file, problem has to be transformed in advance.

        Parameters
        ----------
        filename : str
            name of the input file

        """
        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        absfile = str_conversion(abspath(filename))
        PY_SCIP_CALL(SCIPreadSol(self._scip, absfile))

        locale.setlocale(locale.LC_NUMERIC, user_locale)

    def readSolFile(self, filename):
        """
        Reads a given solution file.

        Solution is created but not added to storage/the model.
        Use 'addSol' OR 'trySol' to add it.

        Parameters
        ----------
        filename : str
            name of the input file

        Returns
        -------
        Solution

        """
        cdef SCIP_Bool partial
        cdef SCIP_Bool error
        cdef SCIP_Bool stored
        cdef Solution solution

        str_absfile = abspath(filename)
        absfile = str_conversion(str_absfile)
        solution = self.createSol()

        user_locale = locale.getlocale(category=locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "C")

        PY_SCIP_CALL(SCIPreadSolFile(self._scip, absfile, solution.sol, False, &partial, &error))

        locale.setlocale(locale.LC_NUMERIC, user_locale)

        if error:
            raise Exception("SCIP: reading solution from file " + str_absfile + " failed!")

        return solution

    def setSolVal(self, Solution solution, Variable var, val):
        """
        Set a variable in a solution.

        Parameters
        ----------
        solution : Solution
            solution to be modified
        var : Variable
            variable in the solution
        val : float
            value of the specified variable

        """
        cdef SCIP_SOL* _sol
        _sol = <SCIP_SOL*>solution.sol

        assert _sol != NULL, "Cannot set value to a freed solution."
        PY_SCIP_CALL(SCIPsetSolVal(self._scip, _sol, var.scip_var, val))

    def trySol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, free=True):
        """
        Check given primal solution for feasibility and try to add it to the storage.

        Parameters
        ----------
        solution : Solution
            solution to store
        printreason : bool, optional
            should all reasons of violations be printed? (Default value = True)
        completely : bool, optional
            should all violation be checked? (Default value = False)
        checkbounds : bool, optional
            should the bounds of the variables be checked? (Default value = True)
        checkintegrality : bool, optional
            does integrality have to be checked? (Default value = True)
        checklprows : bool, optional
            do current LP rows (both local and global) have to be checked? (Default value = True)
        free : bool, optional
            should solution be freed? (Default value = True)

        Returns
        -------
        stored : bool
            whether given solution was feasible and good enough to keep

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        else:
            PY_SCIP_CALL(SCIPtrySol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        return stored

    def getSolObjVal(self, Solution sol, original=True):
        """
        Retrieve the objective value of the solution.

        Parameters
        ----------
        sol : Solution
        original : bool, optional
            objective value in original space (Default value = True)

        Returns
        -------
        float

        """
        if sol == None:
            sol = Solution.create(self._scip, NULL)

        sol._checkStage("getSolObjVal")

        if original:
            objval = SCIPgetSolOrigObj(self._scip, sol.sol)
        else:
            objval = SCIPgetSolTransObj(self._scip, sol.sol)

        return objval

    def getSolVal(
        self,
        Solution sol,
        expr: Union[Expr, GenExpr, MatrixExpr],
    ) -> Union[float, np.ndarray]:
        """
        Retrieve value of given variable or expression in the given solution.

        Parameters
        ----------
        sol : Solution
            Solution to query the value from. If None, the current LP/pseudo solution is
            used.

        expr : Expr, GenExpr, MatrixExpr
            Expression to query the value of.

        Returns
        -------
        float or np.ndarray

        Notes
        -----
        A variable is also an expression.

        """
        # no need to create a NULL solution wrapper in case we have a variable
        return (sol or Solution.create(self._scip, NULL))[expr]

    def getVal(self, expr: Union[Expr, GenExpr, MatrixExpr]) -> Union[float, np.ndarray]:
        """
        Retrieve the value of the given variable or expression in the best known solution.
        Can only be called after solving is completed.

        Parameters
        ----------
        expr : Expr, GenExpr or MatrixExpr
            Expression to query the value of.

        Returns
        -------
        float or np.ndarray

        Notes
        -----
        A variable is also an expression.

        """
        if SCIPgetStage(self._scip) in {SCIP_STAGE_INIT, SCIP_STAGE_FREE}:
            raise Warning("Method cannot be called in stage ", self.getStage())

        self._ensureFreshBestSol()

        if self._bestSol.sol == NULL and SCIPgetStage(self._scip) != SCIP_STAGE_SOLVING:
            raise Warning("No solution available")

        return self._bestSol[expr]

    def getPrimalRayVal(self, Variable var):
        """
        Gets value of given variable in primal ray causing unboundedness of the LP relaxation.

        Parameters
        ----------
        var : Variable

        Returns
        -------
        float

        """
        assert SCIPhasPrimalRay(self._scip), "The problem does not have a primal ray."

        return SCIPgetPrimalRayVal(self._scip, var.scip_var)

    def getPrimalRay(self):
        """
        Gets primal ray causing unboundedness of the LP relaxation.

        Returns
        -------
        list of float

        """
        assert SCIPhasPrimalRay(self._scip), "The problem does not have a primal ray."
        cdef SCIP_VAR** vars = SCIPgetVars(self._scip)
        cdef int nvars = SCIPgetNVars(self._scip)
        cdef int i

        ray = []
        for i in range(nvars):
            ray.append(float(SCIPgetPrimalRayVal(self._scip, vars[i])))

        return ray

    def getPrimalbound(self):
        """
        Retrieve the best primal bound.

        Returns
        -------
        float

        """
        return SCIPgetPrimalbound(self._scip)

    def getDualbound(self):
        """
        Retrieve the best dual bound.

        Returns
        -------
        float

        """
        return SCIPgetDualbound(self._scip)

    def getDualboundRoot(self):
        """
        Retrieve the best root dual bound.

        Returns
        -------
        float

        """
        return SCIPgetDualboundRoot(self._scip)

    def getLowerboundRoot(self):
        """
        Gets lower (dual) bound in transformed problem of the root node.

        Returns
        -------
        float

        """
        return SCIPgetLowerboundRoot(self._scip)

    def chgReoptObjective(self, coeffs, sense = 'minimize'):
        """
        Change the objective function for reoptimization.

        Parameters
        ----------
        coeffs : Expr
            the coefficients as a linear expression
        sense : str
            the objective sense (Default value = 'minimize')

        """
        cdef SCIP_VAR** vars
        cdef int nvars
        cdef SCIP_Real* _coeffs
        cdef SCIP_OBJSENSE objsense
        cdef int i
        cdef _VarArray wrapper

        if sense == "minimize":
            objsense = SCIP_OBJSENSE_MINIMIZE
        elif sense == "maximize":
            objsense = SCIP_OBJSENSE_MAXIMIZE
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

        assert isinstance(coeffs, Expr), "given coefficients are not Expr but %s" % coeffs.__class__.__name__

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            raise ValueError("Constant offsets in objective are not supported!")

        nvars = len(coeffs.terms) - (CONST in coeffs.terms)

        if nvars == 0:
            PY_SCIP_CALL(SCIPchgReoptObjective(self._scip, objsense, NULL, NULL, 0))
            return

        _coeffs = <SCIP_Real*> malloc(nvars * sizeof(SCIP_Real))
        vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))

        i = 0
        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                wrapper = _VarArray(term[0])
                vars[i] = wrapper.ptr[0]
                _coeffs[i] = coef
                i += 1

        PY_SCIP_CALL(SCIPchgReoptObjective(self._scip, objsense, vars, _coeffs, nvars))

        free(vars)
        free(_coeffs)

    def getVarStrongbranchLast(self, Variable var):
        """
        Get the results of the last strong branching call on this variable (potentially was called
        at another node).

        Parameters
        ----------
        var : Variable
            variable to get the previous strong branching information from

        Returns
        -------
        down : float
            The dual bound of the LP after branching down on the variable
        up : float
            The dual bound of the LP after branchign up on the variable
        downvalid : bool
            stores whether the returned down value is a valid dual bound, or NULL
        upvalid : bool
            stores whether the returned up value is a valid dual bound, or NULL
        solval : float
            The solution value of the variable at the last strong branching call
        lpobjval : float
            The LP objective value at the time of the last strong branching call

        """

        cdef SCIP_Real down
        cdef SCIP_Real up
        cdef SCIP_Real solval
        cdef SCIP_Real lpobjval
        cdef SCIP_Bool downvalid
        cdef SCIP_Bool upvalid

        PY_SCIP_CALL(SCIPgetVarStrongbranchLast(self._scip, var.scip_var, &down, &up, &downvalid, &upvalid, &solval, &lpobjval))

        return down, up, downvalid, upvalid, solval, lpobjval

    def getVarStrongbranch(self, Variable var, itlim, idempotent=False, integral=False):
        """
        Strong branches and gets information on column variable.

        Parameters
        ----------
        var : Variable
            Variable to get strong branching information on
        itlim : int
            LP iteration limit for total strong branching calls
        idempotent : bool, optional
            Should SCIP's state remain the same after the call?
        integral : bool, optional
            Boolean on whether the variable is currently integer.

        Returns
        -------
        down : float
            The dual bound of the LP after branching down on the variable
        up : float
            The dual bound of the LP after branchign up on the variable
        downvalid : bool
            stores whether the returned down value is a valid dual bound, or NULL
        upvalid : bool
            stores whether the returned up value is a valid dual bound, or NULL
        downinf : bool
            store whether the downwards branch is infeasible
        upinf : bool
            store whether the upwards branch is infeasible
        downconflict : bool
            store whether a conflict constraint was created for an infeasible downwards branch
        upconflict : bool
            store whether a conflict constraint was created for an infeasible upwards branch
        lperror : bool
            whether an unresolved LP error occurred in the solving process

        """

        cdef SCIP_Real down
        cdef SCIP_Real up
        cdef SCIP_Bool downvalid
        cdef SCIP_Bool upvalid
        cdef SCIP_Bool downinf
        cdef SCIP_Bool upinf
        cdef SCIP_Bool downconflict
        cdef SCIP_Bool upconflict
        cdef SCIP_Bool lperror

        if integral:
            PY_SCIP_CALL(SCIPgetVarStrongbranchInt(self._scip, var.scip_var, itlim, idempotent, &down, &up, &downvalid,
                                                   &upvalid, &downinf, &upinf, &downconflict, &upconflict, &lperror))
        else:
            PY_SCIP_CALL(SCIPgetVarStrongbranchFrac(self._scip, var.scip_var, itlim, idempotent, &down, &up, &downvalid,
                                                    &upvalid, &downinf, &upinf, &downconflict, &upconflict, &lperror))

        return down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror

    # Exact SCIP methods
    def enableExactSolving(self, SCIP_Bool enable):
        """
        Enables or disables exact solving mode in SCIP.

        Parameters
        ----------
        enable : SCIP_Bool
            Whether to enable exact solving mode (True) or disable it (False).
        """

        PY_SCIP_CALL(SCIPenableExactSolving(self._scip, enable))

    def isExact(self):
        """
        Returns whether exact solving mode is enabled in SCIP.

        Returns
        -------
        bool
        """

        return SCIPisExact(self._scip)

    def allowNegSlackExact(self):
        """
        Returns whether negative slack is allowed in exact solving mode.

        Returns
        -------
        bool
        """

        return SCIPallowNegSlack(self._scip)
    
    def branchLPExact(self):
        """
        Performs exact LP branching.

        Returns
        -------
        SCIP_RESULT
        """
        cdef SCIP_RESULT result
        PY_SCIP_CALL(SCIPbranchLPExact(self._scip, &result))
        return result

    def addRowExact(self, RowExact rowexact):
        """
        Adds an exact row to the LP.

        Parameters
        ----------
        rowexact : RowExact
            The exact row to add.
        """
        PY_SCIP_CALL(SCIPaddRowExact(self._scip, rowexact.scip_row_exact))

    def getBipartiteGraphRepresentation(self, prev_col_features=None, prev_edge_features=None, prev_row_features=None,
                                        static_only=False, suppress_warnings=False):
        """
        This function generates the bipartite graph representation of an LP, which was first used in
        the following paper:
        @inproceedings{conf/nips/GasseCFCL19,
        title={Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
        author={Gasse, Maxime and Chételat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
        booktitle={Advances in Neural Information Processing Systems 32},
        year={2019}
        }
        The exact features have been modified compared to the original implementation.
        This function is used mainly in the machine learning community for MIP.
        A user can only call it during the solving process, when there is an LP object. This means calling it
        from some user defined plugin on the Python side.
        An example plugin is a branching rule or an event handler, which is exclusively created to call this function.
        The user must then make certain to return the appropriate SCIP_RESULT (e.g. DIDNOTRUN)

        Parameters
        ----------
        prev_col_features : list of list or None, optional
            The list of column features previously returned by this function
        prev_edge_features : list of list or None, optional
            The list of edge features previously returned by this function
        prev_row_features : list of list or None, optional
            The list of row features previously returned by this function
        static_only : bool, optional
            Whether exclusively static features should be generated
        suppress_warnings : bool, optional
            Whether warnings should be suppressed

        Returns
        -------
        col_features : list of list
        edge_features : list of list
        row_features : list of list
        dict
            The feature mappings for the columns, edges, and rows

        """
        cdef SCIP* scip = self._scip
        cdef SCIP_VARTYPE vtype
        cdef SCIP_Real sim, prod
        cdef int col_i
        cdef int i
        cdef int j
        cdef int k

        # Check if SCIP is in the correct stage
        if SCIPgetStage(scip) != SCIP_STAGE_SOLVING:
            raise Warning("This functionality can only been called in SCIP_STAGE SOLVING. The row and column"
                          "information is then accessible")

        # Generate column features
        cdef SCIP_COL** cols = SCIPgetLPCols(scip)
        cdef int ncols = SCIPgetNLPCols(scip)

        if static_only:
            n_col_features = 5
            col_feature_map = {"continuous": 0, "binary": 1, "integer": 2, "implicit_integer": 3, "obj_coef": 4}
        else:
            n_col_features = 19
            col_feature_map = {"continuous": 0, "binary": 1, "integer": 2, "implicit_integer": 3, "obj_coef": 4,
                               "has_lb": 5, "has_ub": 6, "sol_at_lb": 7, "sol_at_ub": 8, "sol_val": 9, "sol_frac": 10,
                               "red_cost": 11, "basis_lower": 12, "basis_basic": 13, "basis_upper": 14,
                               "basis_zero": 15, "best_incumbent_val": 16, "avg_incumbent_val": 17, "age": 18}

        if prev_col_features is None:
            col_features = [[0 for i in range(n_col_features)] for j in range(ncols)]
        else:
            assert len(prev_col_features) > 0, "Previous column features is empty"
            col_features = prev_col_features
            if len(prev_col_features) != ncols:
                if not suppress_warnings:
                    raise Warning(f"The number of columns has changed. Previous column data being ignored")
                else:
                    col_features = [[0 for i in range(n_col_features)] for j in range(ncols)]
                    prev_col_features = None
            if len(prev_col_features[0]) != n_col_features:
                raise Warning(f"Dimension mismatch in provided previous features and new features:"
                              f"{len(prev_col_features[0])} != {n_col_features}")

        cdef SCIP_SOL* sol = SCIPgetBestSol(scip)
        cdef SCIP_VAR* var
        cdef SCIP_Real lb
        cdef SCIP_Real ub
        cdef SCIP_Real solval
        cdef SCIP_BASESTAT basis_status

        for i in range(ncols):
            col_i = SCIPcolGetLPPos(cols[i])
            var = SCIPcolGetVar(cols[i])

            lb = SCIPcolGetLb(cols[i])
            ub = SCIPcolGetUb(cols[i])
            solval = SCIPcolGetPrimsol(cols[i])

            # Collect the static features first (don't need to changed if previous features are passed)
            if prev_col_features is None:
                # Variable types
                vtype = SCIPvarGetType(var)
                if vtype == SCIP_VARTYPE_BINARY:
                    col_features[col_i][col_feature_map["binary"]] = 1
                elif vtype == SCIP_VARTYPE_INTEGER:
                    col_features[col_i][col_feature_map["integer"]] = 1
                elif vtype == SCIP_VARTYPE_CONTINUOUS:
                    col_features[col_i][col_feature_map["continuous"]] = 1
                elif vtype == SCIP_DEPRECATED_VARTYPE_IMPLINT:
                    col_features[col_i][col_feature_map["implicit_integer"]] = 1
                # Objective coefficient
                col_features[col_i][col_feature_map["obj_coef"]] = SCIPcolGetObj(cols[i])

            # Collect the dynamic features
            if not static_only:
                # Lower bound
                if not SCIPisInfinity(scip, abs(lb)):
                    col_features[col_i][col_feature_map["has_lb"]] = 1

                # Upper bound
                if not SCIPisInfinity(scip, abs(ub)):
                    col_features[col_i][col_feature_map["has_ub"]] = 1

                # Basis status
                basis_status = SCIPcolGetBasisStatus(cols[i])
                if basis_status == SCIP_BASESTAT_LOWER:
                    col_features[col_i][col_feature_map["basis_lower"]] = 1
                elif basis_status == SCIP_BASESTAT_BASIC:
                    col_features[col_i][col_feature_map["basis_basic"]] = 1
                elif basis_status == SCIP_BASESTAT_UPPER:
                    col_features[col_i][col_feature_map["basis_upper"]] = 1
                elif basis_status == SCIP_BASESTAT_ZERO:
                    col_features[col_i][col_feature_map["basis_zero"]] = 1

                # Reduced cost
                col_features[col_i][col_feature_map["red_cost"]] = SCIPgetColRedcost(scip, cols[i])

                # Age
                col_features[col_i][col_feature_map["age"]] = SCIPcolGetAge(cols[i])

                # LP solution value
                col_features[col_i][col_feature_map["sol_val"]] = solval
                col_features[col_i][col_feature_map["sol_frac"]] = SCIPfeasFrac(scip, solval)
                col_features[col_i][col_feature_map["sol_at_lb"]] = int(SCIPisEQ(scip, solval, lb))
                col_features[col_i][col_feature_map["sol_at_ub"]] = int(SCIPisEQ(scip, solval, ub))

                # Incumbent solution value
                if sol is NULL:
                    col_features[col_i][col_feature_map["best_incumbent_val"]] = None
                    col_features[col_i][col_feature_map["avg_incumbent_val"]] = None
                else:
                    col_features[col_i][col_feature_map["best_incumbent_val"]] = SCIPgetSolVal(scip, sol, var)
                    col_features[col_i][col_feature_map["avg_incumbent_val"]] = SCIPvarGetAvgSol(var)

        # Generate row features
        cdef SCIP_ROW** rows = SCIPgetLPRows(scip)
        cdef int nrows = SCIPgetNLPRows(scip)

        if static_only:
            n_row_features = 6
            row_feature_map = {"has_lhs": 0, "has_rhs": 1, "n_non_zeros": 2, "obj_cosine": 3, "bias": 4, "norm": 5}
        else:
            n_row_features = 14
            row_feature_map = {"has_lhs": 0, "has_rhs": 1, "n_non_zeros": 2, "obj_cosine": 3, "bias": 4, "norm": 5,
                               "sol_at_lhs": 6, "sol_at_rhs": 7, "dual_sol": 8, "age": 9,
                               "basis_lower": 10, "basis_basic": 11, "basis_upper": 12, "basis_zero": 13}

        if prev_row_features is None:
            row_features = [[0 for i in range(n_row_features)] for j in range(nrows)]
        else:
            assert len(prev_row_features) > 0, "Previous row features is empty"
            row_features = prev_row_features
            if len(prev_row_features) != nrows:
                if not suppress_warnings:
                    raise Warning(f"The number of rows has changed. Previous row data being ignored")
                else:
                    row_features = [[0 for i in range(n_row_features)] for j in range(nrows)]
                    prev_row_features = None
            if len(prev_row_features[0]) != n_row_features:
                raise Warning(f"Dimension mismatch in provided previous features and new features:"
                              f"{len(prev_row_features[0])} != {n_row_features}")

        cdef SCIP_Real lhs
        cdef SCIP_Real rhs
        cdef SCIP_Real cst
        cdef int nnzrs = 0

        for i in range(nrows):

            # lhs <= activity + cst <= rhs
            lhs = SCIProwGetLhs(rows[i])
            rhs = SCIProwGetRhs(rows[i])
            cst = SCIProwGetConstant(rows[i])
            activity = SCIPgetRowLPActivity(scip, rows[i])

            if prev_row_features is None:
                # number of coefficients
                row_features[i][row_feature_map["n_non_zeros"]] = SCIProwGetNLPNonz(rows[i])
                nnzrs += row_features[i][row_feature_map["n_non_zeros"]]

                # left-hand-side
                if not SCIPisInfinity(scip, abs(lhs)):
                    row_features[i][row_feature_map["has_lhs"]] = 1

                # right-hand-side
                if not SCIPisInfinity(scip, abs(rhs)):
                    row_features[i][row_feature_map["has_rhs"]] = 1

                # bias
                row_features[i][row_feature_map["bias"]] = cst

                # Objective cosine similarity
                row_features[i][row_feature_map["obj_cosine"]] = SCIPgetRowObjParallelism(scip, rows[i])

                # L2 norm
                row_features[i][row_feature_map["norm"]] = SCIProwGetNorm(rows[i])

            if not static_only:

                # Dual solution
                row_features[i][row_feature_map["dual_sol"]] = SCIProwGetDualsol(rows[i])

                # Basis status
                basis_status = SCIProwGetBasisStatus(rows[i])
                if basis_status == SCIP_BASESTAT_LOWER:
                    row_features[i][row_feature_map["basis_lower"]] = 1
                elif basis_status == SCIP_BASESTAT_BASIC:
                    row_features[i][row_feature_map["basis_basic"]] = 1
                elif basis_status == SCIP_BASESTAT_UPPER:
                    row_features[i][row_feature_map["basis_upper"]] = 1
                elif basis_status == SCIP_BASESTAT_ZERO:
                    row_features[i][row_feature_map["basis_zero"]] = 1

                # Age
                row_features[i][row_feature_map["age"]] = SCIProwGetAge(rows[i])

                # Is tight
                row_features[i][row_feature_map["sol_at_lhs"]] = int(SCIPisEQ(scip, activity, lhs))
                row_features[i][row_feature_map["sol_at_rhs"]] = int(SCIPisEQ(scip, activity, rhs))

        # Generate edge (coefficient) features
        cdef SCIP_COL** row_cols
        cdef SCIP_Real * row_vals

        n_edge_features = 3
        edge_feature_map = {"col_idx": 0, "row_idx": 1, "coef": 2}
        if prev_edge_features is None:
            edge_features = [[0 for i in range(n_edge_features)] for j in range(nnzrs)]
            j = 0
            for i in range(nrows):
                # coefficient indexes and values
                row_cols = SCIProwGetCols(rows[i])
                row_vals = SCIProwGetVals(rows[i])
                for k in range(row_features[i][row_feature_map["n_non_zeros"]]):
                    edge_features[j][edge_feature_map["col_idx"]] = SCIPcolGetLPPos(row_cols[k])
                    edge_features[j][edge_feature_map["row_idx"]] = i
                    edge_features[j][edge_feature_map["coef"]] = row_vals[k]
                    j += 1
        else:
            assert len(prev_edge_features) > 0, "Previous edge features is empty"
            edge_features = prev_edge_features
            if len(prev_edge_features) != nnzrs:
                if not suppress_warnings:
                    raise Warning(f"The number of coefficients in the LP has changed. Previous edge data being ignored")
                else:
                    edge_features = [[0 for i in range(3)] for j in range(nnzrs)]
                    prev_edge_features = None
            if len(prev_edge_features[0]) != 3:
                raise Warning(f"Dimension mismatch in provided previous features and new features:"
                              f"{len(prev_edge_features[0])} != 3")

        return (col_features, edge_features, row_features,
                {"col": col_feature_map, "edge": edge_feature_map, "row": row_feature_map})

    def copyLargeNeighborhoodSearch(self, to_fix, fix_vals) -> Model:
        """
        Creates a configured copy of the transformed problem and applies provided fixings intended for LNS heuristics.

        Parameters
        ----------
        to_fix : List[Variable]
            A list of variables to fix in the copy
        fix_vals : List[Real]
            A list of the values to which to fix the variables in the copy (care their order)

        Returns
        -------
        model : Model
            A model containing the created copy
        """
        cdef SCIP* subscip
        cdef SCIP_HASHMAP* varmap
        cdef SCIP_VAR** orig_vars = SCIPgetVars(self._scip)
        cdef SCIP_VAR** vars = <SCIP_VAR**> malloc(len(to_fix) * sizeof(SCIP_VAR*))
        cdef SCIP_Real* vals = <SCIP_Real*> malloc(len(fix_vals) * sizeof(SCIP_Real))
        cdef SCIP_Real val
        cdef SCIP_Bool valid
        cdef SCIP_Bool success
        cdef int i
        cdef int j = 0

        name_to_val = {var.name: val for var, val in zip(to_fix, fix_vals)}
        for i, var in enumerate(self.getVars()):
            if var.name in name_to_val:
                vars[j] = orig_vars[i]
                vals[j] = <SCIP_Real>name_to_val[var.name]
                j+= 1

        PY_SCIP_CALL(SCIPcreate(&subscip))
        PY_SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(subscip), self.getNVars()) )
        PY_SCIP_CALL( SCIPcopyLargeNeighborhoodSearch(self._scip, subscip, varmap, "LNhS_subscip", vars, vals,
                                                      <int>len(to_fix), False, False, &success, &valid) )
        sub_model = BaseModel.create(subscip)
        sub_model._freescip = True
        free(vars)
        free(vals)
        SCIPhashmapFree(&varmap)

        return sub_model

    def addMatrixVar(self,
                     shape: Union[int, Tuple],
                     name: Union[str, np.ndarray] = '',
                     vtype: Union[str, np.ndarray] = 'C',
                     lb: Union[int, float, np.ndarray, None] = 0.0,
                     ub: Union[int, float, np.ndarray, None] = None,
                     obj: Union[int, float, np.ndarray] = 0.0,
                     pricedVar: Union[bool, np.ndarray] = False,
                     pricedVarScore: Union[int, float, np.ndarray] = 1.0
                     ) -> MatrixVariable:
        """
        Create a new matrix of variable. Default matrix variables are non-negative and continuous.

        Parameters
        ----------
        shape : int or tuple
            the shape of the resultant MatrixVariable
        name : str or np.ndarray, optional
            name of the matrix variable, generic if empty (Default value = '')
        vtype : str or np.ndarray, optional
            type of the matrix variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
            (Default value = 'C')
        lb : float or np.ndarray or None, optional
            lower bound of the matrix variable, use None for -infinity (Default value = 0.0)
        ub : float or np.ndarray or None, optional
            upper bound of the matrix variable, use None for +infinity (Default value = None)
        obj : float or np.ndarray, optional
            objective value of matrix variable (Default value = 0.0)
        pricedVar : bool or np.ndarray, optional
            is the matrix variable a pricing candidate? (Default value = False)
        pricedVarScore : float or np.ndarray, optional
            score of matrix variable in case it is priced, the higher the better (Default value = 1.0)

        Returns
        -------
        MatrixVariable

        """
        # assert has_numpy, "Numpy is not installed. Please install numpy to use matrix variables."

        if isinstance(name, np.ndarray):
            assert name.shape == shape
        if isinstance(vtype, np.ndarray):
            assert vtype.shape == shape
        if isinstance(lb, np.ndarray):
            assert lb.shape == shape
        if isinstance(ub, np.ndarray):
            assert ub.shape == shape
        if isinstance(obj, np.ndarray):
            assert obj.shape == shape
        if isinstance(pricedVar, np.ndarray):
            assert pricedVar.shape == shape
        if isinstance(pricedVarScore, np.ndarray):
            assert pricedVarScore.shape == shape

        if isinstance(shape, int):
            ndim = 1
        else:
            ndim = len(shape)

        matrix_variable = np.empty(shape, dtype=object)

        if isinstance(name, str):
            matrix_names = np.full(shape, name, dtype=object)
            if name != "":
                for idx in np.ndindex(matrix_variable.shape):
                    matrix_names[idx] = f"{name}_{'_'.join(map(str, idx))}"
        else:
            matrix_names = name

        if not isinstance(vtype, np.ndarray):
            matrix_vtypes = np.full(shape, vtype, dtype=str)
        else:
            matrix_vtypes = vtype

        if not isinstance(lb, np.ndarray):
            matrix_lbs = np.full(shape, lb, dtype=object)
        else:
            matrix_lbs = lb

        if not isinstance(ub, np.ndarray):
            matrix_ubs = np.full(shape, ub, dtype=object)
        else:
            matrix_ubs = ub

        if not isinstance(obj, np.ndarray):
            matrix_objs = np.full(shape, obj, dtype=float)
        else:
            matrix_objs = obj

        if not isinstance(pricedVar, np.ndarray):
            matrix_priced_vars = np.full(shape, pricedVar, dtype=bool)
        else:
            matrix_priced_vars = pricedVar

        if not isinstance(pricedVarScore, np.ndarray):
            matrix_priced_var_scores = np.full(shape, pricedVarScore, dtype=float)
        else:
            matrix_priced_var_scores = pricedVarScore

        for idx in np.ndindex(matrix_variable.shape):
            matrix_variable[idx] = self.addVar(name=matrix_names[idx], vtype=matrix_vtypes[idx], lb=matrix_lbs[idx],
                                               ub=matrix_ubs[idx], obj=matrix_objs[idx], pricedVar=matrix_priced_vars[idx],
                                               pricedVarScore=matrix_priced_var_scores[idx])

        return matrix_variable.view(MatrixVariable)
