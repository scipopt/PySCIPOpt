############
Cut Selector
############

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model, SCIP_RESULT
  from pyscipopt.scip import Cutsel

  scip = Model()

.. contents:: Contents

What is a Cut Selector?
========================

A cutting plane (cut) selector is an algorithm that selects which cuts to add to the
optimization problem. It is given a set of candidate cuts, and from the set must decide which
subset to add.

Cut Selector Structure
=======================

A cut selector in PySCIPOpt takes the following structure:

.. code-block:: python

  class DummyCutsel(Cutsel):

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        :param cuts: the cuts which we want to select from. Is a list of scip Rows
        :param forcedcuts: the cuts which we must add. Is a list of scip Rows
        :param root: boolean indicating whether weare at the root node
        :param maxnselectedcuts: int which is the maximum amount of cuts that can be selected
        :return: sorted cuts and forcedcuts
        """

        return {'cuts': sorted_cuts, 'nselectedcuts': n,
                'result': SCIP_RESULT.SUCCESS}

The class ``DummyCutsel`` inherits the necessary ``Cutsel`` class, and then programs
the necessary function ``cutselselect``. The docstrings of the ``cutselselect`` explain
the input to the function. It is then up to the user to create some new ordering ``cuts``,
which we have represented by ``sorted_cuts``, return ``nselectedcuts`` that results in the first
``nselectedcuts`` of the ``sorted_cuts``being added to the optimization problem. The
``SCIP_RESULT`` is there to indicate whether the algorithm was successful. See the
appropriate documentation for more potential result code.

To include a cut selector one would need to do something like the following code:

.. code-block:: python

  cutsel = DummyCutsel()
  scip.includeCutsel(cutsel, 'name', 'description', 5000000)

The final argument of the ``includeCutsel`` function in the example above was the
priority. If the priority is higher than all other cut selectors then it will be called
first. In the case of some failure or non-success return code, then the second highest
priority cut selector is called and so on.

Example Cut Selector
======================

In this example we will program a cut selector that selects the 10 most
efficacious cuts. Efficacy is the standard measure for cut quality and can be calcuated
via SCIP directly.

.. code-block:: python

  class MaxEfficacyCutsel(Cutsel):

      def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
          """
          Selects the 10 cuts with largest efficacy.
          """

          scip = self.model

          scores = [0] * len(cuts)
          for i in range(len(scores)):
              scores[i] = scip.getCutEfficacy(cuts[i])

          rankings = sorted(range(len(cuts)), key=lambda x: scores[x], reverse=True)

          sorted_cuts = [cuts[rank] for rank in rankings]

          assert len(sorted_cuts) == len(cuts)

          return {'cuts': sorted_cuts, 'nselectedcuts': min(maxnselectedcuts, len(cuts), 10),
                  'result': SCIP_RESULT.SUCCESS}


Things to Keep in Mind
=======================

Here are some things to keep in mind when programming your own custom cut selector.

- Do not change any of the actual cut information!
- Do not reorder the ``forcedcuts``. They are provided as reference points to inform
  the selection process. They should not be edited or reordered.
- Only reorder ``cuts``. Do not add any new cuts.
