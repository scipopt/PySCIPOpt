import unittest

from pyscipopt import Model, quicksum, sqrt
from pyscipopt.scip import (
    CONST,
    Expr,
    ExprCons,
    ExprLike,
    GenExpr,
    Term,
    Variable,
)


class VariableTest(unittest.TestCase):
    """Tests for the M4 semantics: Variable is an ExprLike, not an Expr."""

    def setUp(self):
        self.m = Model()
        self.m.hideOutput()
        self.x = self.m.addVar(name="x", lb=0, ub=10)
        self.y = self.m.addVar(name="y", lb=0, ub=10)

    def test_isinstance_semantics(self):
        x = self.x
        self.assertIsInstance(x, ExprLike)
        self.assertNotIsInstance(x, Expr)
        self.assertIsInstance(x, Variable)

        expr = x + 1
        self.assertIsInstance(expr, Expr)
        self.assertNotIsInstance(expr, Variable)

        gen = sqrt(x)
        self.assertIsInstance(gen, GenExpr)
        self.assertIsInstance(gen, ExprLike)

    def test_add_variables(self):
        x, y = self.x, self.y
        expr = x + y
        self.assertIsInstance(expr, Expr)
        self.assertEqual(len(expr.terms), 2)
        self.assertEqual(expr[x], 1.0)
        self.assertEqual(expr[y], 1.0)
        self.assertEqual(expr[CONST], 0.0)

    def test_multiply_variable_with_itself(self):
        x = self.x
        square = x * x
        self.assertIsInstance(square, Expr)
        self.assertEqual(len(square.terms), 1)
        self.assertEqual(square[Term(x) * Term(x)], 1.0)
        self.assertEqual(square[x], 0.0)

    def test_subtract_variable_from_itself(self):
        x = self.x
        diff = x - x
        self.assertIsInstance(diff, Expr)
        self.assertEqual(diff[x], 0.0)
        self.assertEqual(diff[CONST], 0.0)

        # Normalization through ExprCons drops the zero coefficient.
        cons = ExprCons(diff, lhs=0.0)
        self.assertEqual(cons.expr.terms, {})

    def test_quicksum_matches_addition(self):
        x, y = self.x, self.y
        quick = quicksum([x, y])
        added = x + y
        self.assertIsInstance(quick, Expr)
        # quicksum starts from an Expr carrying an explicit zero constant term,
        # so compare the coefficients instead of the raw dict.
        self.assertEqual(quick[CONST], added[CONST])
        self.assertEqual(len(quick.terms) - 1, len(added.terms))
        for var in (x, y):
            self.assertEqual(quick[var], added[var])

    def test_operations_do_not_mutate_variable_view(self):
        x, y = self.x, self.y
        _ = x + y
        _ = 2 * x
        _ = quicksum([x, y])
        # x still represents exactly the variable term x
        self.assertEqual(str(x), "x")
        self.assertEqual(x[x], 1.0)
        self.assertEqual(x[y], 0.0)

    def test_chgReoptObjective_with_variable(self):
        m = self.m
        x, y = self.x, self.y
        m.enableReoptimization()
        m.addCons(x + y <= 15)
        m.setObjective(x + y, sense="maximize")
        m.optimize()
        self.assertAlmostEqual(m.getObjVal(), 15.0)

        m.freeReoptSolve()
        # Regression: Variable used to be an Expr with a .terms attribute.
        m.chgReoptObjective(x, sense="maximize")
        m.optimize()
        self.assertAlmostEqual(m.getVal(x), 10.0)


if __name__ == "__main__":
    unittest.main()
