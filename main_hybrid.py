from problem_MO import MOAssemblyLineBalancingProblem
from hybrid import hybrid_gwo_scso
from data import TASK_TIMES, PRECEDENCE, MAX_EPOCH, POPULATION_SIZE, STAGNATION_LIMIT
from utils import create_output_directory
from visualization import plot_pareto_2d, plot_pareto_front, plot_workstation_tasks

ALGORITHM_NAME = "Hybrid GWO-SCSO"
OUTPUT_DIR = "output/albp/hybrid"

create_output_directory(OUTPUT_DIR)

problem = MOAssemblyLineBalancingProblem(
    TASK_TIMES, PRECEDENCE, cycle_time_upper_bound=120
)
print(f"Running {ALGORITHM_NAME}...")
pareto, all_solutions = hybrid_gwo_scso(
    problem, pop_size=POPULATION_SIZE, max_epoch=MAX_EPOCH, stagnation_limit=STAGNATION_LIMIT, epsilons=(0, 0)
)

print("pareto:", pareto)
print("all_solutions[0:5]:", all_solutions[0:5])

print("len(pareto):", len(pareto))
print("len(all_solutions):", len(all_solutions))

for i, sol in enumerate(pareto):
    obj = sol["fitness"]
    print(
        f"\n🔹 Solution {i + 1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s"
    )
    decoded = problem.decode(sol["position"])
    workstations = problem.assign_tasks(decoded)
    for idx, ws in enumerate(workstations):
        total_time = sum(TASK_TIMES[t] for t in ws)
        print(
            f"  Workstation {idx + 1} [{total_time:.1f}s]: {', '.join(f'Task {t}' for t in ws)}"
        )

plot_pareto_front(pareto, ALGORITHM_NAME, OUTPUT_DIR)
plot_pareto_2d(pareto, all_solutions, ALGORITHM_NAME, OUTPUT_DIR)
for idx, solution in enumerate(pareto, start=1):
    plot_workstation_tasks(
        solution,
        problem,
        algorithm_name=ALGORITHM_NAME,
        output_dir=OUTPUT_DIR,
        filename=f"workstation_tasks_{idx}.png",
    )
