##@file sudoku.py
# @brief Simple example of modeling a Sudoku as a binary program

from pyscipopt import Model, quicksum

# initial Sudoku values
init = [5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9]

m = Model()

# create a binary variable for every field and value
x = {}
for i in range(9):
    for j in range(9):
        for k in range(9):
            name = str(i) + ',' + str(j) + ',' + str(k)
            x[i, j, k] = m.addVar(name, vtype='B')

# fill in initial values
for i in range(9):
    for j in range(9):
        if init[j + 9 * i] != 0:
            m.addCons(x[i, j, init[j + 9 * i] - 1] == 1)

# only one digit in every field
for i in range(9):
    for j in range(9):
        m.addCons(quicksum(x[i, j, k] for k in range(9)) == 1)

# set up row and column constraints
for ind in range(9):
    for k in range(9):
        m.addCons(quicksum(x[ind, j, k] for j in range(9)) == 1)
        m.addCons(quicksum(x[i, ind, k] for i in range(9)) == 1)

# set up square constraints
for row in range(3):
    for col in range(3):
        for k in range(9):
            m.addCons(quicksum(x[i + 3 * row, j + 3 * col, k] for i in range(3) for j in range(3)) == 1)

m.hideOutput()
m.optimize()

if m.getStatus() != 'optimal':
    print('Sudoku is not feasible!')
else:
    print('\nSudoku solution:\n')
    sol = {}
    for i in range(9):
        out = ''
        for j in range(9):
            for k in range(9):
                if m.getVal(x[i, j, k]) == 1:
                    sol[i, j] = k + 1
            out += str(sol[i, j]) + ' '
        print(out)
