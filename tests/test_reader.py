import pytest
import os

from pyscipopt import Model, quicksum, Reader, SCIP_RESULT

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

def test():
    createFile("tmp.sod")

    m = Model("soduko")
    reader = SudokuReader()

    m.includeReader(reader, "sodreader", "PyReader for soduko problem", "sod")

    m.readProblem("tmp.sod")

    m.optimize()

    deleteFile("tmp.sod")

    m.writeProblem("model.sod")
    with open("model.sod", "r") as f:
        input = f.readline()
    assert input == "soduko"

    deleteFile("model.sod")


if __name__ == "__main__":
    test()
