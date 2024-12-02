from pyscipopt import Model

# Define categorical data
shifts = {"Morning": 0, "Afternoon": 1, "Night": 2}
employees = ["Alice", "Bob", "Charlie"]

# Employees have different salary demands
cost = {
    "Alice": [2,4,1],
    "Bob": [3,2,7],
    "Charlie": [3,3,3]
}

# Employee availability
availability = {
    "Alice": ["Morning", "Afternoon"],
    "Bob": ["Afternoon", "Night"],
    "Charlie": ["Morning", "Night"]
}

# Transform availability into integer values
availability_int = {
    emp: [shifts[shift] for shift in available_shifts]
    for emp, available_shifts in availability.items()
}

# Create the model
model = Model("Shift Assignment")

# x[e, s] = 1 if employee e is assigned to shift s
x = {}
for e in employees:
    for s in shifts.values():
        x[e, s] = model.addVar(vtype="B", name=f"x({e},{s})")

# Each shift must be assigned to exactly one employee
for s in shifts.values():
    model.addCons(sum(x[e, s] for e in employees) == 1)

# Employees can only work shifts they are available for
for e in employees:
    for s in shifts.values():
        if s not in availability_int[e]:
            model.addCons(x[e, s] == 0)

# Minimize shift assignment cost
model.setObjective(
    sum(cost[e][s]*x[e, s] for e in employees for s in shifts.values()), "minimize"
)

# Solve the problem
model.optimize()

# Display the results
print("\nOptimal Shift Assignment:")
for e in employees:
    for s, s_id in shifts.items():
        if model.getVal(x[e, s_id]) > 0.5:
            print("%s is assigned to %s" % (e, s))
