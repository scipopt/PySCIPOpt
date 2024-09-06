#########################################
Lazy Constraints (via Constraint Handler)
#########################################

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, quicksum, Conshdlr, SCIP_RESULT

  scip = Model()

.. contents:: Contents

What are Lazy Constraints?
==========================

A lazy constraint is a constraint that the user believes improves computational times when removed from the
optimization problem until it is violated. Generally, lazy constraints exist in an incredibly large amount,
with most of them being trivially satisfied.


Why use a Constraint Handler?
=============================

SCIP does not have a lazy constraint plug-in, rather its definition of constraint is broad enough to
naturally encompass lazy constraints already. Therefore the user must simply create an appropriate
constraint handler.


TSP Subtour Elimination Constraint Example
==========================================

In this example we will examine a basic TSP integer programming formulation, where the exponential
amount of subtour elimination constraints are treated as lazy constraints.

TSP (short for travelling salesman problem) is a classic optimization problem. Let :math:`n \in \mathbb{N}`
be the number of nodes in a graph with vertex set :math:`\mathcal{V} = \{1,...,n\}` and assume that
the graph is complete, i.e., any two vertices are connected by an edge. Each edge :math:`(i,j)` has an associated cost
:math:`c_{i,j} \in \mathbb{R}` and an associated binary variable :math:`x_{i,j}`. A standard
integer programming formulation for the problem is:

.. math::

    &\text{min} & & \sum_{i=1}^{n} \sum_{j=1}^{n} c_{i,j} x_{i,j} \\
    &\text{s.t.} & & \sum_{i=1}^{n} x_{i,j} = 2, \quad \forall j \in \mathcal{V} \\
    & & & \sum_{i,j \in \mathcal{S}} x_{i,j} \leq |\mathcal{S}| - 1, \quad \forall \mathcal{S} \subset \mathcal{V}, |\mathcal{S}| \geq 2 \quad (*) \\
    & & & x_{i,j} \in \{0,1\}, \quad \forall (i,j) \in \mathcal{V} \times \mathcal{V}

In the above formulation, the second set of constraints (marked with an \*) are called subtour elimination constraints.
They are called such as a valid solution in absense of those constraints might consist of a collection
of smaller cycles instead of a single large cycle. As the constraint set requires checking every subset of nodes
there are exponentially many. Moreover, we know that most of the constraints are probably unnecessary,
because it is clear from the objective that a minimum tour does not exist with a mini-cycle of nodes that are
extremely far away from each other. Therefore, we want to model them as lazy constraints!

For modelling these constraints using a constraint handler, the constraint handler needs to
be able to answer the following questions:

- Is a given solution feasible?
- If the given solution is not feasible, can you do something to forbid this solution from further consideration?

We will now create the basic model containing all information aside from the constraint handler

.. code-block:: python

  import numpy as np
  import networkx
  n = 300
  x = {}
  c = {}
  for i in range(n):
      x[i] = {}
      c[i] = {}
      for j in range(i + 1, n):
          x[i][j] = scip.addVar(vtype='B', name=f"x_{i}_{j}")
          c[i][j] = np.random.uniform(10)
  scip.setObjective(quicksum(quicksum(c[i][j]*x[i][j] for j in range(i + 1, n)) for i in range(n)), "minimize")
  for i in range(n):
      scip.addCons(quicksum(x[i][j] for j in range(i + 1, n)) + quicksum(x[j][i] for j in range(i-1, 0, -1)) == 2,
                   name=f"sum_in_out_{i}")


Now we will create the code on how to implement such a constraint handler.

.. code-block:: python

  # subtour elimination constraint handler
  class SEC(Conshdlr):

      # method for creating a constraint of this constraint handler type
      def createCons(self, name, variables):
          model = self.model
          cons = model.createCons(self, name)

          # data relevant for the constraint; in this case we only need to know which
          # variables cannot form a subtour
          cons.data = {}
          cons.data['vars'] = variables
          return cons


      # find subtours in the graph induced by the edges {i,j} for which x[i][j] is positive
      # at the given solution; when solution is None, the LP solution is used
      def find_subtours(self, cons, solution = None):
          edges = []
          x = cons.data['vars']

          for i in list(x.keys()):
              for j in list(x[i].keys()):
                  if self.model.getSolVal(solution, x[i][j]) > 0.5:
                      edges.append((i, j))

          G = networkx.Graph()
          G.add_edges_from(edges)
          components = list(networkx.connected_components(G))

          if len(components) == 1:
              return []
          else:
              return components

      # checks whether solution is feasible
      def conscheck(self, constraints, solution, check_integrality,
                    check_lp_rows, print_reason, completely, **results):

          # check if there is a violated subtour elimination constraint
          for cons in constraints:
              if self.find_subtours(cons, solution):
                  return {"result": SCIP_RESULT.INFEASIBLE}

          # no violated constriant found -> feasible
          return {"result": SCIP_RESULT.FEASIBLE}


      # enforces the LP solution: searches for subtours in the solution and adds
      # adds constraints forbidding all the found subtours
      def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
          consadded = False

          for cons in constraints:
              subtours = self.find_subtours(cons)

              # if there are subtours
              if subtours:
                  x = cons.data['vars']

                  # add subtour elimination constraint for each subtour
                  for S in subtours:
                      print("Constraint added!)
                      self.model.addCons(quicksum(x[i][j] for i in S for j in S if j>i) <= len(S)-1)
                      consadded = True

          if consadded:
              return {"result": SCIP_RESULT.CONSADDED}
          else:
              return {"result": SCIP_RESULT.FEASIBLE}


      # this is rather technical and not relevant for the exercise. to learn more see
      # https://scipopt.org/doc/html/CONS.php#CONS_FUNDAMENTALCALLBACKS
      def conslock(self, constraint, locktype, nlockspos, nlocksneg):
          pass

In the above we've created our problem and custom constraint handler! We now need to actually
add the constraint handler to the problem. After that, we can simply call ``optimize`` whenever we are ready.
To add the costraint handler use something along the lines of the following:

.. code-block:: python

    # create the constraint handler
    conshdlr = SEC()

    # Add the constraint handler to SCIP. We set check priority < 0 so that only integer feasible solutions
    # are passed to the conscheck callback
    scip.includeConshdlr(conshdlr, "TSP", "TSP subtour eliminator", chckpriority = -10, enfopriority = -10)

    # create a subtour elimination constraint
    cons = conshdlr.createCons("no_subtour_cons", x)

    # add constraint to SCIP
    scip.addPyCons(cons)