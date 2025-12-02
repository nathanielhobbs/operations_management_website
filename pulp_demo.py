import pulp as pl
p = pl.LpProblem("demo", pl.LpMaximize)
x = pl.LpVariable("x", lowBound=0); y = pl.LpVariable("y", lowBound=0)
p += 3*x + 5*y
p += 2*x + 1*y <= 10
p += 1*x + 3*y <= 12
p.solve(pl.GLPK_CMD(msg=True, options=['--ranges','sens.txt']))  # writes Excel-like ranges
print(open('sens.txt').read()[:400])  # should show "Objective Coefficients" / "Right Hand Sides"

