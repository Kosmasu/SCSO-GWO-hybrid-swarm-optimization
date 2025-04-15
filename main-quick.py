from mealpy import PSO, FFA, ABC, GWO, SCSO

from problems import Squared, Rastrigin, StyblinskiTang, Griewank
from hybrid import HybridGWOSCSO # NORMAL GWO + SCSO
from hybrid_2 import HybridIGWOSCSO # IMPROVED GWO + SCSO
from hybrid_v3 import HybridGWOSCSO3 # HYBRID GWO - SCSO With SCSO starting phase
from hybrid_v3B import HybridGWOSCSO3B # HYBRID GWO - SCSO With SCSO starting phase

EPOCH = 200
POPULATION = 30
DIMENSION = 250
# EARLY STOPPING FOR FAIRER TESTING BCS SCSO USUALLY FOUND 0.0 EARLIER IN ITERATION
# Should we also test via max time? With no iteration limit
# But i might have to modify my hybrid code for the initial run condition
term_dict = {
    "max_early_stop": 20,  # after 20 epochs, if the global best doesn't improve then we stop the program
}

"""
problem_sq = Squared(
    lb=-100,
    ub=100,
    dimensions=DIMENSION,
    minmax="min",
    save_population=True,
    name=f"Squared-{DIMENSION}D",
)

problem = Rastrigin(
        lb=-100,
        ub=100,
        dimensions=DIMENSION,
        minmax="min",
        save_population=True,
        name=f"Rastrigin-{DIMENSION}D",
)

"""
problem = Rastrigin(
        lb=-100,
        ub=100,
        dimensions=DIMENSION,
        minmax="min",
        save_population=True,
        name=f"Rastrigin-{DIMENSION}D",
)

#problem_sta = StyblinskiTang(
#        lb=-100,
#        ub=100,
#        dimensions=DIMENSION,
#        minmax="min",
#        save_population=True,
#        name=f"StyblinskiTang-{DIMENSION}D",
#)

"""
# Particle Swarm Optimization (PSO)
model_pso = PSO.OriginalPSO(epoch=EPOCH, pop_size=POPULATION)
best_pso = model_pso.solve(problem=problem)

# Firefly Algorithm (FFA)
model_ffa = FFA.OriginalFFA(epoch=EPOCH, pop_size=POPULATION)
best_ffa = model_ffa.solve(problem=problem)

# Artificial Bee Colony (ABC)
model_abc = ABC.OriginalABC(epoch=EPOCH, pop_size=POPULATION)
best_abc = model_abc.solve(problem=problem)
"""
# Grey Wolf Optimizer (GWO)
model_gwo = GWO.OriginalGWO(epoch=EPOCH, pop_size=POPULATION)
best_gwo = model_gwo.solve(problem=problem, termination=term_dict)

# Sand Cat Swarm Optimization (SCSO)
model_scso = SCSO.OriginalSCSO(epoch=EPOCH, pop_size=POPULATION)
best_scso = model_scso.solve(problem=problem, termination=term_dict)

STAGNATION_LIMIT = 15
# Normal GWO-SCSO Hybrid
model_gwo_scso = HybridGWOSCSO(epoch=EPOCH, pop_size=POPULATION, stagnation_limit=STAGNATION_LIMIT)
best_gwo_scso = model_gwo_scso.solve(problem=problem, termination=term_dict)

# Improved GWO-SCSO Hybrid
model_igwo_scso = HybridIGWOSCSO(epoch=EPOCH, pop_size=POPULATION, stagnation_limit=STAGNATION_LIMIT)
best_igwo_scso = model_igwo_scso.solve(problem=problem, termination=term_dict)

STAGNATION_LIMIT_2 = 5
INITIAL_SCSO = 0.25 #Run SCSO for 25% of the iteration before switching to GWO
# GWO-SCSO V3
model_gwo_scso3 = HybridGWOSCSO3(epoch=EPOCH, pop_size=POPULATION, stagnation_limit=STAGNATION_LIMIT_2, initial_scso=INITIAL_SCSO) 
best_gwo_scso3 = model_gwo_scso3.solve(problem=problem, termination=term_dict)
# GWO-SCSO V3B
#model_gwo_scso3b = HybridGWOSCSO3B(epoch=EPOCH, pop_size=POPULATION, stagnation_limit=1)
#best_gwo_scso3b = model_gwo_scso3b.solve(problem=problem, termination=term_dict)

# Print the best solutions from each algorithm
#print(f"Best solution from PSO: {best_pso}")
#print(f"Best solution from FFA: {best_ffa}")
#print(f"Best solution from ABC: {best_abc}")
print(f"Best solution from GWO: {best_gwo}")
print(f"Best solution from SCSO: {best_scso}")
print(f"Best solution from GWO-SCSO: {best_gwo_scso}")
print(f"Best solution from IGWO-SCSO: {best_igwo_scso}")
print(f"Best solution from GWO-SCSO V3: {best_gwo_scso3}")

#PSO Graph
#model_pso.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/PSO_GLB")
#model_pso.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/PSO_LOC")
#model_pso.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_PSO_GLB")
#model_pso.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_PSO_LOC")

#FFA Graph
#model_ffa.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/FFA_GLB")
#model_ffa.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/FFA_LOC")
#model_ffa.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_FFA_GLB")
#model_ffa.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_FFA_LOC")

#ABC Graph
#model_abc.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/ABC_GLB")
#model_abc.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/ABC_LOC")
#model_abc.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_ABC_GLB")
#model_abc.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_ABC_LOC")

#GWO Graph
model_gwo.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/GWO_GLB")
model_gwo.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/GWO_LOC")
#model_gwo.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_GWO_GLB")
#model_gwo.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_GWO_LOC")

#SCSO Graph
model_scso.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/SCSO_GLB")
model_scso.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/SCSO_LOC")
#model_scso.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_SCSO_GLB")
#model_scso.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_SCSO_LOC")

#GWO-SCSO Graph
model_gwo_scso.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/GWO_SCSO_GLB")
model_gwo_scso.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/GWO_SCSO_LOC")
#model_gwo_scso.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_GWO_SCSO_GLB")
#model_gwo_scso.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_GWO_SCSO_LOC")

#IGWO-SCSO Graph
model_igwo_scso.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/IGWO_SCSO_GLB")
model_igwo_scso.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/IGWO_SCSO_LOC")
#model_igwo_scso.history.save_global_objectives_chart(filename=f"graph_{problem.name}/OBJ_IGWO_SCSO_GLB")
#model_igwo_scso.history.save_local_objectives_chart(filename=f"graph_{problem.name}/OBJ_IGWO_SCSO_LOC")

#GWO-SCSO V3 Graph
model_gwo_scso3.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/GWO_SCSO_V3_GLB")
model_gwo_scso3.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/GWO_SCSO_V3_LOC")

#GWO-SCSO V3B Graph
#model_gwo_scso3b.history.save_global_best_fitness_chart(filename=f"graph_{problem.name}/GWO_SCSO_V3_GLB")
#model_gwo_scso3b.history.save_local_best_fitness_chart(filename=f"graph_{problem.name}/Local_Best/GWO_SCSO_V3_LOC")

#Writing Result to Txt
f = open(f"graph_{problem.name}/Result_{problem.name}.txt", "a")
f.write(f'''
Test Parameters
--------------------------
Max Iteration(Epoch): {EPOCH}
Population: {POPULATION}
Dimension: {DIMENSION}

Result of {problem.name}
--------------------------
Best Fitness Only
GWO       : {model_gwo.g_best.target.fitness}
SCSO      : {model_scso.g_best.target.fitness}
GWO-SCSO  : {model_gwo_scso.g_best.target.fitness}
IGWO-SCSO : {model_igwo_scso.g_best.target.fitness}
GWO-SCSO3 : {model_gwo_scso3.g_best.target.fitness}
GWO-SCSO3B: -
==========================
RUN TIME
==========================
GWO       : {sum(model_gwo.history.list_epoch_time)}
SCSO      : {sum(model_scso.history.list_epoch_time)}
GWO-SCSO  : {sum(model_gwo_scso.history.list_epoch_time)}
IGWO-SCSO : {sum(model_igwo_scso.history.list_epoch_time)}
GWO-SCSO3 : {sum(model_gwo_scso3.history.list_epoch_time)}
GWO-SCSO3B: -
==========================
BEST FITNESS & SOLUTION
==========================
GWO       : {best_gwo}
SCSO      : {best_scso}
GWO-SCSO  : {best_gwo_scso}
IGWO-SCSO : {best_igwo_scso}
GWO-SCSO3 : {best_gwo_scso3}
GWO-SCSO3B: -
        ''')
f.close()

"""
PSO      : {model_pso.g_best.target.fitness}
FFA      : {model_ffa.g_best.target.fitness}
ABC      : {model_abc.g_best.target.fitness}

PSO      : {sum(model_pso.history.list_epoch_time)}
FFA      : {sum(model_ffa.history.list_epoch_time)}
ABC      : {sum(model_abc.history.list_epoch_time)}

PSO      : {best_pso}
FFA      : {best_ffa}
ABC      : {best_abc}
"""