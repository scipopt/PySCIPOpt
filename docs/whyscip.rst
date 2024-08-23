###########
Why SCIP?
###########

.. note:: This is written from the perspective user that is primarily MILP focused.

This is an important question, and one that is in general answered by performance claims.
To be clear, SCIP is performant. It is one of the leading open-source solvers.
It manages to be competitive on a huge array of benchmarks, which include but are not limited to,
mixed-integer linear programming, mixed-integer quadratic programming, mixed-integer semidefinite
programming, mixed-integer non-linear programming, and pseudo-boolean optimization.
This page will attempt to answer the question "Why SCIP?" without relying on a performance comparison.
It will convey the scope of SCIP, how the general structure of SCIP works,
and the natural advantages (also weaknesses) SCIP has compared to other mixed-integer optimizers.

So, why SCIP? SCIP (Solving Constraint Integer Programs) is likely much more general than you expect.
It also likely provides easy to use functionality that you didn't know was possible to freely access.

Combining Integer Programming and Constraint Programming
=========================================================

SCIP (Solving Constraint Integer Programs) combines techniques from both the integer programming and
constraint programming communities. For both communities, a common solving technique is
successively dividing the problem into smaller subproblems and solving those subproblems recursively, i.e., branching.
The communities differ however in how they handle those subproblems, with the MIP community using LP relaxations and
cutting planes to provide dual bounds, and the CP community using propagation to tighten variable domains.
Why is this relevant? It is to emphasise that SCIP is more than a branch-and-cut solver.

Differences to Traditional Branch-and-Cut
============================================
Let's expand on this generality of SCIP.
Those from the MIP community are familiar with branch-and-cut based solvers. The algorithm is the backbone of all
modern solvers, and in general are associated with the following problem (we'll stick to linear for now):

.. math::

    &\text{min} & &\mathbf{c}^{t}x \\
    &\text{s.t.} & & \mathbf{A}x \leq \mathbf{b} \\
    & & & x \in \mathbb{Z}^{|\mathcal{J}|} \times \mathbb{R}^{[n] / \mathcal{J}}, \quad \mathcal{J} \subseteq [n]

This is the standard formulation all MIP practitioners are aware of. How would a solver go about solving
this problem? First the solver would perform some presolve. This is maybe the most impactful performance
choice of the solving process. The formulation provided is altered in some way to make the subsequent solving process
faster. This is why in SCIP you have the ``transformed`` problem and the ``original`` problem. This is not
unique SCIP however, it is just that information is more open when using SCIP.

The solver would then solve the root node LP (linear programming) relaxation. The LP relaxation is obtained by
simply removing the integrality requirement on the problem above:

.. math::

    &\text{min} & &\mathbf{c}^{t}x \\
    &\text{s.t.} & & \mathbf{A}x \leq \mathbf{b} \\
    & & & x \in \mathbb{R}^{[n]}

The solver would then generate cuts, add some cuts, resolve the LP, and repeat this process until some termination
condition is hit. Let's pause here and discuss some probable assumptions that were made from a programming perspective.




- Has been used for a long time. Bugs definitely exist, but it is tried and tested.
- Plugin generality (Lp solver etc)


