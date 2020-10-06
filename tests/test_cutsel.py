from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import Cutsel
import itertools

class MaxEfficacyCutsel(Cutsel):

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        Selects the 10 cuts with largest efficacy. Ensures that all forced cuts are passed along.
        Overwrites the base cutselselect of Cutsel.

        :param cuts: the cuts which we want to select from. Is a list of scip Rows
        :param forcedcuts: the cuts which we must add. Is a list of scip Rows
        :return: sorted cuts and forcedcuts
        """

        scip = self.model

        scores = [0] * len(cuts)
        for i in range(len(scores)):
            scores[i] = scip.getCutEfficacy(cuts[i])

        rankings = sorted(range(len(cuts)), key = lambda x: scores[x])

        sorted_cuts = [cuts[rank] for rank in rankings]

        assert len(sorted_cuts) == len(cuts)

        return {'cuts': sorted_cuts, 'forcedcuts': forcedcuts, 'nselectedcuts': min(maxnselectedcuts, len(cuts), 10),
                'result': SCIP_RESULT.SUCCESS}


def test_cut_selector():
    scip = Model()
    scip.setIntParam("presolving/maxrounds", 0)
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)

    cutsel = MaxEfficacyCutsel()
    scip.includeCutsel(cutsel, 'max_efficacy', 'maximises efficacy', 5000000)

    # Make a basic minimum spanning hypertree problem
    # Let's construct a problem with 10 vertices and 20 hyperedges. The hyperedges are our variables.
    v = [0,1,2,3,4,5,6,7,8,9]
    e = {}
    for i in range(20):
        e[i] = scip.addVar(vtype='B', name='hyperedge_{}'.format(i))

    # Construct a dummy incident matrix
    A = [[1, 2, 3], [2, 3, 4, 5], [4, 9], [7, 8, 9], [0, 8, 9],
         [1, 6, 8], [0, 1, 2, 9], [0, 3, 5, 7, 8], [2, 3], [6, 9],
         [5, 8], [1, 9], [2, 7, 8, 9], [3, 8], [2, 4],
         [0, 1], [0, 1, 4], [2, 5], [1, 6, 7, 8], [1, 3, 4, 7, 9]]

    # Create a cost vector for each hyperedge
    c = [2.5, 2.9, 3.2, 7, 1.2, 0.5,
         8.6, 9, 6.7, 0.3, 4,
         0.9, 1.8, 6.7, 3, 2.1,
         1.8, 1.9, 0.5, 4.3, 5.6]

    # Add constraint that your hypertree touches all vertices
    scip.addCons(quicksum((len(A[i]) -1) * e[i] for i in range(len(A))) == len(v) - 1)

    # Now add the sub-tour elimination constraints.
    for i in range(2,len(v) + 1):
        for combination in itertools.combinations(v,i):
            scip.addCons(quicksum(max(len(set(combination) & set(A[j])) -1, 0) * e[j] for j in range(len(A))) <= i -1,
                         name='cons_{}'.format(combination))

    # Add objective to minimise the cost
    scip.setObjective(quicksum(c[i] * e[i] for i in range(len(A))), sense='minimize')

    scip.optimize()

if __name__ == "__main__":
    test_cut_selector()
