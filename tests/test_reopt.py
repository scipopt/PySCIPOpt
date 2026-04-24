import unittest
from pyscipopt import Model, quicksum, Expr


class ReoptimizationTest(unittest.TestCase):

    def test_reopt(self):
        """Test basic reoptimization."""
        m = Model()
        m.enableReoptimization()

        x = m.addVar(name="x", ub=5)
        y = m.addVar(name="y", lb=-2, ub=10)

        m.addCons(2 * x + y >= 8)
        m.setObjective(x + y)
        m.optimize()
        print("x", m.getVal(x))
        print("y", m.getVal(y))
        self.assertEqual(m.getVal(x), 5.0)
        self.assertEqual(m.getVal(y), -2.0)

        m.freeReoptSolve()
        m.addCons(y <= 3)
        m.addCons(y + x <= 6)
        m.chgReoptObjective(- x - 2 * y)

        m.optimize()
        print("x", m.getVal(x))
        print("y", m.getVal(y))
        self.assertEqual(m.getVal(x), 3.0)
        self.assertEqual(m.getVal(y), 3.0)

    def test_reopt_maximize(self):
        """Test reoptimization with maximize sense."""
        m = Model()
        m.enableReoptimization()
        m.hideOutput()

        x = m.addVar(name="x", lb=0, ub=10)
        y = m.addVar(name="y", lb=0, ub=10)

        m.addCons(x + y <= 15)
        m.setObjective(x + y, sense="maximize")
        m.optimize()

        self.assertAlmostEqual(m.getObjVal(), 15.0)

        m.freeReoptSolve()
        m.chgReoptObjective(x, sense="maximize")
        m.optimize()

        self.assertAlmostEqual(m.getVal(x), 10.0)

    def test_reopt_many_variables_sparse_objective(self):
        """Test with many variables but only few in the new objective."""
        m = Model()
        m.enableReoptimization()
        m.hideOutput()

        n_vars = 100
        vars = [m.addVar(name=f"x_{i}", lb=0, ub=10) for i in range(n_vars)]

        m.addCons(quicksum(vars) >= 50)
        m.setObjective(quicksum(vars))
        m.optimize()

        m.freeReoptSolve()
        m.chgReoptObjective(vars[0] + vars[50] + vars[99])
        m.optimize()

        self.assertAlmostEqual(m.getVal(vars[0]), 0.0)
        self.assertAlmostEqual(m.getVal(vars[50]), 0.0)
        self.assertAlmostEqual(m.getVal(vars[99]), 0.0)

    def test_freeReoptSolve_requires_reoptimization_enabled(self):
        """freeReoptSolve must raise ValueError instead of crashing when
        reoptimization was not enabled (see issue #624)."""
        m = Model()
        m.hideOutput()
        self.assertFalse(m.isReoptEnabled())

        x = m.addVar(name="x", vtype="I", ub=10)
        m.addCons(x >= 3)
        m.setObjective(x)
        m.optimize()

        with self.assertRaises(ValueError):
            m.freeReoptSolve()

        m2 = Model()
        m2.enableReoptimization()
        self.assertTrue(m2.isReoptEnabled())
        m2.hideOutput()
        y = m2.addVar(name="y", vtype="I", ub=10)
        m2.addCons(y >= 3)
        m2.setObjective(y)
        m2.optimize()
        m2.freeReoptSolve()

    def test_reopt_zero_objective(self):
        """Test reoptimization with zero objective (no variables, all coefficients zero)."""
        m = Model()
        m.enableReoptimization()
        m.hideOutput()

        x = m.addVar(name="x", lb=0, ub=10)
        y = m.addVar(name="y", lb=0, ub=10)

        m.addCons(x + y >= 5)
        m.setObjective(x + y)
        m.optimize()

        self.assertAlmostEqual(m.getObjVal(), 5.0)

        m.freeReoptSolve()
        m.chgReoptObjective(Expr())
        m.optimize()

        self.assertGreaterEqual(m.getVal(x) + m.getVal(y), 5.0)
        self.assertAlmostEqual(m.getObjVal(), 0.0)


if __name__ == '__main__':
    unittest.main()
