import pytest

networkx = pytest.importorskip("networkx")

from pyscipopt import Model, Conshdlr, SCIP_RESULT, SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING

try:
    from types import SimpleNamespace
except:
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))

        def __eq__(self, other):
            return self.__dict__ == other.__dict__


#initial Sudoku values
init = [5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9]

def plot_graph(G):
    plt = pytest.importorskip("matplotlib.pyplot")

    X,Y = networkx.bipartite.sets(G)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    networkx.draw(G, pos=pos, with_labels=False)

    labels = {}
    for node in G.nodes():
        labels[node] = node

    networkx.draw_networkx_labels(G, pos, labels)
    plt.show()

# all different constraint handler
class ALLDIFFconshdlr(Conshdlr):

    # value graph: bipartite graph between variables and the union of their domains
    # an edge connects a variable and a value iff the value is in the variable's domain
    def build_value_graph(self, vars, domains):
        #print(domains)
        vals = set([])
        for var in vars:
            #print("domain of var ", var.name, "is ", domains[var])
            vals.update(domains[var.ptr()]) # vals = vals union domains[var]

        G = networkx.Graph()
        G.add_nodes_from((var.name for var in vars), bipartite = 0) # add vars names as nodes
        G.add_nodes_from(vals, bipartite = 1)                       # add union of values as nodes

        for var in vars:
            for value in domains[var.ptr()]:
                G.add_edge(var.name, value)

        return G, vals

    # propagates single constraint: uses Regin's Algorithm as described in
    # https://www.ps.uni-saarland.de/courses/seminar-ws04/papers/anastasatos.pdf
    # The idea is that every solution of an all different constraint corresponds to a maximal matching in
    # a bipartite graph (see value graph). Furthermore, if an arc of this arc is in no maximal matching, then
    # one can remove it. Removing and arc corresponds to remove a value in the domain of the variable.
    # So what the algorithm does is to determine which arcs can be in a maximal matching. Graph theory help
    # us build fast algorithm so that we don't have to compute all possible maximal matchings ;)
    # That being said, the implementation is pretty naive and brute-force, so there is a lot of room for improvement
    def propagate_cons(self, cons):
        #print("propagating cons %s with id %d"%(cons.name, id(cons)))
        vars = cons.data.vars
        domains = cons.data.domains

        # TODO: would be nice to have a flag to know whether we should propagate the constraint.
        # We would need an event handler to let us know whenever a variable of our constraint changed its domain
        # Currently we can't write event handlers in python.

        G, vals = self.build_value_graph(vars, domains)
        try:
            M = networkx.bipartite.maximum_matching(G) # returns dict between nodes in matching
        except:
            top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
            bottom_nodes = set(G) - top_nodes
            M = networkx.bipartite.maximum_matching(G, top_nodes) # returns dict between nodes in matching

        if( len(M)/2 < len(vars) ):
            #print("it is infeasible: max matching of card ", len(M), " M: ", M)
            #print("Its value graph:\nV = ", G.nodes(), "\nE = ", G.edges())
            plot_graph(G)
            return SCIP_RESULT.CUTOFF

        # build auxiliary directed graph: direct var -> val if [var, val] is in matching, otherwise var <- val
        # note that all vars are matched
        D = networkx.DiGraph()
        D.add_nodes_from(G) ## this seems to work
        for var in vars:
            D.add_edge(var.name, M[var.name])
            for val in domains[var.ptr()]:
                if val != M[var.name]:
                    D.add_edge(val, var.name)

        # find arcs that *do not* need to be removed and *remove* them from G. All remaining edges of G
        # should be use to remove values from the domain of variables
        # get all free vertices
        V = set(G.nodes())
        V_matched = set(M)
        V_free = V.difference(V_matched)
        #print("matched nodes ", V_matched, "\nfree nodes ", V_free)
        # TODO quit() << this produces an assertion

        # no variable should be free!
        for var in vars:
            assert var.name not in V_free

        # perform breadth first search starting from free vertices and mark all visited edges as useful
        for v in V_free:
            visited_edges = networkx.bfs_edges(D, v)
            G.remove_edges_from(visited_edges)

        # compute strongly connected components of D and mark edges on the cc as useful
        for g in networkx.strongly_connected_components(D):
            for e in D.subgraph(g).edges():
                if G.has_edge(*e):
                    G.remove_edge(*e)

        # cannot remove edges in matching!
        for var in vars:
            e = (var.name, M[var.name])
            if G.has_edge(*e):
                G.remove_edge(*e)

        # check that there is something to remove
        if G.size() == 0:
            return SCIP_RESULT.DIDNOTFIND

        #print("Edges to remove!", G.edges())
        # remove values
        for var in vars:
            for val in domains[var.ptr()].copy():
                if G.has_edge(var.name, val):
                    domains[var.ptr()].remove(val) # this asserts if value is not there and we shouldn't delete two times the same value

        # "fix" variable when possible
        for var in vars:
            #print("domain of var ", var.name, "is ", domains[var])
            minval = min(domains[var.ptr()])
            maxval = max(domains[var.ptr()])
            if var.getLbLocal() <  minval:
                self.model.chgVarLb(var, minval)
            if var.getUbLocal() > maxval:
                self.model.chgVarUb(var, maxval)
            #print("bounds of ", var, "are (%d,%d)"%(minval,maxval))

        return SCIP_RESULT.REDUCEDDOM

    # propagator callback
    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming): # I have no idea what to return, documentation?
        result = SCIP_RESULT.DIDNOTFIND
        for cons in constraints:
            prop_result = self.propagate_cons(cons)
            if prop_result == SCIP_RESULT.CUTOFF:
                result = prop_result
                break
            if prop_result == SCIP_RESULT.REDUCEDDOM:
                result = prop_result

        return {"result": result}

    def is_cons_feasible(self, cons, solution = None):
        #print("checking feasibility of constraint %s id: %d"%(cons.name, id(cons)))
        sol_values = set()
        for var in cons.data.vars:
            sol_values.add(round(self.model.getSolVal(solution, var)))
        #print("sol_values = ", sol_values)
        return len(sol_values) == len(cons.data.vars)

    # checks whether solution is feasible, ie, if they are all different
    # since the checkpriority is < 0, we are only called if the integrality
    # constraint handler didn't find infeasibility, so solution is integral
    def conscheck(self, constraints, solution, check_integrality, check_lp_rows, print_reason, completely):
        for cons in constraints:
            if not self.is_cons_feasible(cons, solution):
                return {"result": SCIP_RESULT.INFEASIBLE}
        return {"result": SCIP_RESULT.FEASIBLE}

    # enforces LP solution
    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
        for cons in constraints:
            if not self.is_cons_feasible(cons):
                # TODO: suggest some value to branch on
                return {"result": SCIP_RESULT.INFEASIBLE}
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        for var in constraint.data.vars:
            self.model.addVarLocks(var, nlockspos + nlocksneg , nlockspos + nlocksneg)

    def constrans(self, constraint):
        #print("CONSTRANS BEING CAAAAAAAAAAAAAAAAAAAALLLLLLED")
        return {}


# builds sudoku model; adds variables and all diff constraints
def create_sudoku():
    scip = Model("Sudoku")

    x = {} # values of squares
    for row in range(9):
        for col in range(9):
            # some variables are fix
            if init[row*9 + col] != 0:
                x[row,col] = scip.addVar(vtype = "I", lb = init[row*9 + col], ub = init[row*9 + col], name = "x(%s,%s)" % (row,col))
            else:
                x[row,col] = scip.addVar(vtype = "I", lb = 1, ub = 9, name = "x(%s,%s)" % (row,col))
            var = x[row,col]
            #print("built var ", var.name, " with bounds: (%d,%d)"%(var.getLbLocal(), var.getUbLocal()))

    conshdlr = ALLDIFFconshdlr()

    # hoping to get called when all vars have integer values
    scip.includeConshdlr(conshdlr, "ALLDIFF", "All different constraint", propfreq = 1, enfopriority = -10, chckpriority = -10)

    # row constraints; also we specify the domain of all variables here
    # TODO/QUESTION: in principle domain is of course associated to the var and not the constraint. it should be "var.data"
    # But ideally that information would be handle by SCIP itself... the reason we can't is because domain holes is not implemented, right?
    domains = {}
    for row in range(9):
        vars = []
        for col in range(9):
            var = x[row,col]
            vars.append(var)
            vals = set(range(int(round(var.getLbLocal())), int(round(var.getUbLocal())) + 1))
            domains[var.ptr()] = vals
        # this is kind of ugly, isn't it?
        cons = scip.createCons(conshdlr, "row_%d" % row)
        #print("in test: received a constraint with id ", id(cons)) ### DELETE
        cons.data = SimpleNamespace() # so that data behaves like an instance of a class (ie, cons.data.whatever is allowed)
        cons.data.vars = vars
        cons.data.domains = domains
        scip.addPyCons(cons)

    # col constraints
    for col in range(9):
        vars = []
        for row in range(9):
            var = x[row,col]
            vars.append(var)
        cons = scip.createCons(conshdlr, "col_%d"%col)
        cons.data = SimpleNamespace()
        cons.data.vars = vars
        cons.data.domains = domains
        scip.addPyCons(cons)

    # square constraints
    for idx1 in range(3):
        for idx2 in range(3):
            vars = []
            for row in range(3):
                for col in range(3):
                    var = x[3*idx1 + row, 3*idx2 + col]
                    vars.append(var)
            cons = scip.createCons(conshdlr, "square_%d-%d"%(idx1, idx2))
            cons.data = SimpleNamespace()
            cons.data.vars = vars
            cons.data.domains = domains
            scip.addPyCons(cons)


    #scip.setObjective()

    return scip, x


def test_main():
    scip, x = create_sudoku()

    scip.setBoolParam("misc/allowstrongdualreds", False)
    scip.setEmphasis(SCIP_PARAMEMPHASIS.CPSOLVER)
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.optimize()

    if scip.getStatus() != 'optimal':
        print('Sudoku is not feasible!')
    else:
        print('\nSudoku solution:\n')
        for row in range(9):
            out = ''
            for col in range(9):
                out += str(round(scip.getVal(x[row,col]))) + ' '
            print(out)