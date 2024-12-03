"""
This example shows how one can optimize a model with categorical data by converting it into integers.

There are three employees (Alice, Bob, Charlie) and three shifts. Each shift is assigned an integer:

Morning   - 0
Afternoon - 1
Night     - 2

The employees have availabilities (e.g. Alice can only work in the Morning and Afternoon), and different
salary demands. These constraints, and an additional one stipulating that every shift must be covered,
allows us to model a MIP with the objective of minimizing the money spent on salary.
"""

from pyscipopt import Model

# Define categorical data
shift_to_int = {"Morning": 0, "Afternoon": 1, "Night": 2}
employees    = ["Alice", "Bob", "Charlie"]

# Employee availability
availability = {
    "Alice": ["Morning", "Afternoon"],
    "Bob": ["Afternoon", "Night"],
    "Charlie": ["Morning", "Night"]
}

# Transform availability into integer values
availability_int = {}
for emp, available_shifts in availability.items():
    availability_int[emp] = [shift_to_int[shift] for shift in available_shifts] 


# Employees have different salary demands
cost = {
    "Alice":   [2,4,1],
    "Bob":     [3,2,7],
    "Charlie": [3,3,3]
}

# Create the model
model = Model("Shift Assignment")

# x[e, s] = 1 if employee e is assigned to shift s
x = {}
for e in employees:
    for s in shift_to_int.values():
        x[e, s] = model.addVar(vtype="B", name=f"x({e},{s})")

# Each shift must be assigned to exactly one employee
for s in shift_to_int.values():
    model.addCons(sum(x[e, s] for e in employees) == 1)

# Employees can only work shifts they are available for
for e in employees:
    for s in shift_to_int.values():
        if s not in availability_int[e]:
            model.addCons(x[e, s] == 0)

# Minimize shift assignment cost
model.setObjective(
    sum(cost[e][s]*x[e, s] for e in employees for s in shift_to_int.values()), "minimize"
)

# Solve the problem
model.optimize()

# Display the results
print("\nOptimal Shift Assignment:")
for e in employees:
    for s, s_id in shift_to_int.items():
        if model.getVal(x[e, s_id]) > 0.5:
            print("%s is assigned to %s" % (e, s))
