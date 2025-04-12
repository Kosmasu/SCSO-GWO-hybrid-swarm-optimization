from mealpy import PSO, FFA, ABC

from obj_functions import Squared

problem = Squared(
    lb=-100,
    ub=100,
    dimensions=10,
    minmax="min",
    save_population=True,
    name="Squared_10d",
)

EPOCH = 1000
POPULATION = 30

model_pso = PSO.OriginalPSO(epoch=EPOCH, pop_size=POPULATION)
best_pso = model_pso.solve(problem=problem)

model_ffa = FFA.OriginalFFA(epoch=EPOCH, pop_size=POPULATION)
best_ffa = model_ffa.solve(problem=problem)

model_abc = ABC.OriginalABC(epoch=EPOCH, pop_size=POPULATION)
best_abc = model_abc.solve(problem=problem)

print(f"Best solution from PSO: {best_pso}")
print(f"Best solution from FFA: {best_ffa}")
print(f"Best solution from ABC: {best_abc}")
