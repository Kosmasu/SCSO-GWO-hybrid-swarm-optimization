from mealpy import PSO, FFA, ABC, GWO, SCSO

from problems import Squared

problem = Squared(
    lb=-100,
    ub=100,
    dimensions=10,
    minmax="min",
    save_population=True,
    name="Squared_10d",
)

EPOCH = 9999
POPULATION = 30

# Particle Swarm Optimization (PSO)
model_pso = PSO.OriginalPSO(epoch=EPOCH, pop_size=POPULATION)
best_pso = model_pso.solve(problem=problem)

# Firefly Algorithm (FFA)
model_ffa = FFA.OriginalFFA(epoch=EPOCH, pop_size=POPULATION)
best_ffa = model_ffa.solve(problem=problem)

# Artificial Bee Colony (ABC)
model_abc = ABC.OriginalABC(epoch=EPOCH, pop_size=POPULATION)
best_abc = model_abc.solve(problem=problem)

# Grey Wolf Optimizer (GWO)
model_gwo = GWO.OriginalGWO(epoch=EPOCH, pop_size=POPULATION)
best_gwo = model_gwo.solve(problem=problem)

# Sand Cat Swarm Optimization (SCSO)
model_scso = SCSO.OriginalSCSO(epoch=EPOCH, pop_size=POPULATION)
best_scso = model_scso.solve(problem=problem)

# Print the best solutions from each algorithm
print(f"Best solution from PSO: {best_pso}")
print(f"Best solution from FFA: {best_ffa}")
print(f"Best solution from ABC: {best_abc}")
print(f"Best solution from GWO: {best_gwo}")
print(f"Best solution from SCSO: {best_scso}")

