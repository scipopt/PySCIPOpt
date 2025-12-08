from pyscipopt.scip import Expr, PolynomialExpr, ConstExpr, Term, ExprCons

if __name__ == "__main__":
    from pyscipopt import Model
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    e1 = x + y
    
    # Force ConstExpr
    from pyscipopt.scip import ConstExpr
    ce = ConstExpr(5.0)
    
    print("\n--- Test 1: e1 == 5.0 (Poly == float) ---")
    c1 = e1 == 5.0
    print(f"Result: {c1}")
    
    print("\n--- Test 2: e1 == ConstExpr(5.0) ---")
    c2 = e1 == ce
    print(f"Result: {c2}")
    
    print("\n--- Test 3: ConstExpr(5.0) == e1 ---")
    c3 = ce == e1
    print(f"Result: {c3}")

    print(f"\nType of e1: {type(e1)}")
    print(f"Type of ce: {type(ce)}")
    
    print("\n--- Test 4: Explicit ce.__eq__(e1) ---")
    c4 = ce.__eq__(e1)
    print(f"Result: {c4}")
