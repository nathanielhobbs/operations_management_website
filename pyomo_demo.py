from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, Constraint, maximize, SolverFactory
m = ConcreteModel()
m.x = Var(domain=NonNegativeReals); m.y = Var(domain=NonNegativeReals)
m.obj = Objective(expr=3*m.x + 5*m.y, sense=maximize)
m.c1 = Constraint(expr=2*m.x + m.y <= 10)
m.c2 = Constraint(expr=m.x + 3*m.y <= 12)
opt = SolverFactory('glpk')
opt.options['ranges'] = 'sens.txt'  # Excel-style ranges
opt.solve(m, keepfiles=True, symbolic_solver_labels=True)

