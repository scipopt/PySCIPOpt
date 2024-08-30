######################
Branching Rule Intro
######################

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, Branchrule, SCIP_RESULT

  scip = Model()

.. contents:: Contents

What is Branching
===================

Branching is when an optimization problem is split into smaller subproblems.
Traditionally this is done on an integer variable with a fractional LP solution, with
two child nodes being created with constraints ``x >= ceil(frac)`` and ``x <= floor(frac)``.
In SCIP, arbitrary amount of children nodes can be created, and the constraints added the
created nodes can also be arbitrary. This is not going to be used in the examples below, but this
should be kept in mind when considering your application of your branching rule.

Example Branching Rule
=======================

Here we will program a most infeasible branching rule. This rule selects the integer variable
whose LP solution is most fractional.

.. code-block:: python

  import numpy as np

  class MostInfBranchRule(Branchrule):

      def __init__(self, scip):
          self.scip = scip

      def branchexeclp(self, allowaddcons):

          # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
          # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
          # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
          branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

          # Find the variable that is most fractional
          best_cand_idx = 0
          best_dist = np.inf
          for i in range(npriocands):
              if abs(branch_cand_fracs[i] - 0.5) <= best_dist:
                  best_dist = abs(branch_cand_fracs[i] - 0.5)
                  best_cand_idx = i

          # Branch on the variable with the largest score
          down_child, eq_child, up_child = self.model.branchVarVal(
              branch_cands[best_cand_idx], branch_cand_sols[best_cand_idx])

          return {"result": SCIP_RESULT.BRANCHED}

Let's talk about some features of this branching rule. Currently we only explicitly programmed
a single function, which is called ``branchexeclp``. This is the function that gets called
when branching on an LP optimal solution. While this is the main case, it is not the only
case that SCIP handles. What if there was an LP error at the node, or you are given a set of external
candidates? For more information on this please read `this page <https://www.scipopt.org/doc/html/BRANCH.php>`_.

Now let's discuss what the function returned. We see that it returned a simple dictionary, with the
``"result"`` key and a ``SCIP_RESULT``. This is because inside the function the child nodes
have already been created, and the solver just needs to be made aware of this with the appropriate
code. In the case of branching when something goes wrong (and you have not made the children nodes),
just simply returned ``SCIP_RESULT:DIDNOTRUN``. This will then move on to the next branching rule with
the next highest priority.

.. note::

  Returning ``SCIP_RESULT:DIDNOTRUN`` for more complicated components of the branching rule
  line in ``branchexecps`` is completely encouraged. It is even strongly suggested if you are doing
  an LP specific branching rule.

Now we will finally see how to include the branching rule.

.. code-block:: python

  scip = Model()

  most_inf_branch_rule = MostInfBranchRule(scip)
  scip.includeBranchrule(most_inf_branch_rule, "mostinf", "custom most infeasible branching rule",
                         priority=10000000, maxdepth=-1, maxbounddist=1)

This function ``includeBranchrule`` takes the branching rule as an argument, the name (which will
be visible in the statistics), a description, the priority (for your rule to be called by default it must
be higher than the current highest, which can be quite large), the maxdepth of the branch and bound tree
for which the rule still works (-1 for unlimited), and the maxbounddist (We recommend using 1 and to see
SCIP documentation for an explanation).

Strong Branching Information
=============================
