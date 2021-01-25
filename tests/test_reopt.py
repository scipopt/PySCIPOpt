import unittest
from pyscipopt import Model

class ReoptimizationTest(unittest.TestCase):

    def test_reopt(self):

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


if __name__ == '__main__':
    unittest.main()
