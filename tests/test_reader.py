import pytest
import os

from pyscipopt import Model, quicksum, Reader, SCIP_RESULT, readStatistics

class SudokuReader(Reader):

    def readerread(self, filename):
        with open(filename, "r") as f:
            input = f.readline().split()

            for i in range(len(input)):
                input[i] = int(input[i])

            x = {}
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        name = str(i)+','+str(j)+','+str(k)
                        x[i,j,k] = self.model.addVar(name, vtype='B')
            
            # fill in initial values
            for i in range(9):
                for j in range(9):
                    if input[j + 9*i] != 0:
                        self.model.addCons(x[i,j,input[j + 9*i]-1] == 1)

            # only one digit in every field
            for i in range(9):
                for j in range(9):
                    self.model.addCons(quicksum(x[i,j,k] for k in range(9)) == 1)

            # set up row and column constraints
            for ind in range(9):
                for k in range(9):
                    self.model.addCons(quicksum(x[ind,j,k] for j in range(9)) == 1)
                    self.model.addCons(quicksum(x[i,ind,k] for i in range(9)) == 1)

            # set up square constraints
            for row in range(3):
                for col in range(3):
                    for k in range(9):
                        self.model.addCons(quicksum(x[i+3*row, j+3*col, k] for i in range(3) for j in range(3)) == 1)

        return {"result": SCIP_RESULT.SUCCESS}

    def readerwrite(self, file, name, transformed, objsense, objscale, objoffset, binvars, intvars,
                    implvars, contvars, fixedvars, startnvars, conss, maxnconss, startnconss, genericnames):
        with file as f:
            f.write(name)

        return {"result": SCIP_RESULT.SUCCESS}


def createFile(filename):
    with open(filename, "w") as f:
        f.write("5 3 0 0 7 0 0 0 0 6 0 0 1 9 5 0 0 0 0 9 8 0 0 0 0 6 0 8 0 0 0 6 0 0 0 3 4 0 0 8 0 3 0 0 1 7 0 0 0 2 0 0 0 6 0 6 0 0 0 0 2 8 0 0 0 0 4 1 9 0 0 5 0 0 0 0 8 0 0 7 9")

def deleteFile(filename):
    os.remove(filename)

def test_sudoku_reader():
    createFile("tmp.sod")

    m = Model("sudoku")
    reader = SudokuReader()

    m.includeReader(reader, "sudreader", "PyReader for sudoku problem", "sod")

    m.readProblem("tmp.sod")

    m.optimize()

    deleteFile("tmp.sod")

    m.writeProblem("model.sod")
    with open("model.sod", "r") as f:
        input = f.readline()
    assert input == "sudoku"

    deleteFile("model.sod")

@pytest.mark.skip(reason="Test fails on Windows when using cibuildwheel. Cannot find tests/data")
def test_readStatistics():
    m = Model(problemName="readStats")
    x = m.addVar(vtype="I")
    y = m.addVar()

    m.addCons(x+y <= 3)
    m.hideOutput()
    m.optimize()
    m.writeStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    result = readStatistics(os.path.join("tests", "data", "readStatistics.stats"))

    assert result.status == "optimal"
    assert len([k for k, val in result.__dict__.items() if not str(hex(id(val))) in str(val)]) == 20 # number of attributes. See https://stackoverflow.com/a/57431390/9700522
    assert type(result.total_time) == float
    assert result.problem_name == "readStats"
    assert result.presolved_problem_name == "t_readStats"
    assert type(result.primal_dual_integral) == float
    assert result.n_solutions_found == 1
    assert type(result.gap) == float
    assert result._presolved_constraints == {"initial": 1, "maximal": 1}
    assert result._variables == {"total": 2, "binary": 0, "integer": 1, "implicit": 0, "continuous": 1}
    assert result._presolved_variables == {"total": 0, "binary": 0, "integer": 0, "implicit": 0, "continuous": 0}
    assert result.n_vars == 2
    assert result.n_presolved_vars == 0
    assert result.n_binary_vars == 0
    assert result.n_integer_vars == 1

    m = Model()
    x = m.addVar()
    m.setObjective(-x)
    m.hideOutput()
    m.optimize()
    m.writeStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    result = readStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    assert result.status == "unbounded"

    m = Model()
    x = m.addVar()
    m.addCons(x <= -1)
    m.hideOutput()
    m.optimize()
    m.writeStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    result = readStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    assert result.status == "infeasible"
    assert result.gap == None
    assert result.n_solutions_found == 0

    m = Model()
    x = m.addVar()
    m.hideOutput()
    m.setParam("limits/solutions", 0)
    m.optimize()
    m.writeStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    result = readStatistics(os.path.join("tests", "data", "readStatistics.stats"))
    assert result.status == "user_interrupt"
    assert result.gap == None