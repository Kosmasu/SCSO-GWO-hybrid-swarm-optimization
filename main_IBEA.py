from platypus import IBEA
from problem_MO import MOAssemblyLineBalancingProblem
from utils import filter_results, create_output_directory
from data import TASK_TIMES, PRECEDENCE, MAX_EPOCH, POPULATION_SIZE
from visualization import plot_pareto_2d, plot_pareto_front, plot_workstation_tasks

ALGORITHM_NAME = "IBEA"
OUTPUT_DIR = f"output/albp/{ALGORITHM_NAME.lower()}"

create_output_directory(OUTPUT_DIR)

# Wrapper class to log all evaluated solutions
class ProblemWithSolutionLogging(MOAssemblyLineBalancingProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_solutions = []

    def evaluate(self, solution):
        super().evaluate(solution) # MOAssemblyLineBalancingProblem.evaluate sets solution.objectives
        if hasattr(solution, 'objectives') and solution.objectives is not None:
            self.logged_solutions.append({
                "position": list(solution.variables),
                "fitness": list(solution.objectives)
            })
        # Platypus Problem's evaluate method doesn't typically return a value
        # It modifies the solution object in place.

problem = ProblemWithSolutionLogging(TASK_TIMES, PRECEDENCE, cycle_time_upper_bound=120)

# TO RUN FAIR BENCHMARKING. BECAUSE ALGORITHM.RUN(N). N IS NOT THE SAME WITH MAX EPOCH.
total_evals = MAX_EPOCH * POPULATION_SIZE

algorithm = IBEA(problem, POPULATION_SIZE)
print(f"Running {ALGORITHM_NAME}...")
algorithm.run(total_evals)

platypus_results = algorithm.result
pareto_solutions_platypus_objects = filter_results(platypus_results, epsilons=(0.0, 0.0))

# Use logged solutions for all_solutions
all_solutions = problem.logged_solutions 
pareto_solutions_formatted = [{"position": s.variables, "fitness": s.objectives} for s in pareto_solutions_platypus_objects]

# Print Pareto Front
for i, solution in enumerate(pareto_solutions_platypus_objects):
    obj = solution.objectives
    print(f"\n🔹 {ALGORITHM_NAME} Solution {i+1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s")
    decoded = problem.decode(solution.variables)
    workstations = problem.assign_tasks(decoded)
    for idx, ws in enumerate(workstations):
        total_time = sum(TASK_TIMES[t] for t in ws)
        print(f"  Workstation {idx+1} [{total_time:.1f}s]: {', '.join(f'Task {t}' for t in ws)}")

plot_pareto_front(pareto_solutions_formatted, ALGORITHM_NAME, OUTPUT_DIR)
plot_pareto_2d(pareto_solutions_formatted, all_solutions, ALGORITHM_NAME, OUTPUT_DIR)
for idx, sol_dict in enumerate(pareto_solutions_formatted, start=1):
    plot_workstation_tasks(
        sol_dict,
        problem,
        algorithm_name=ALGORITHM_NAME,
        output_dir=OUTPUT_DIR,
        filename=f"workstation_tasks_{idx}.png",
    )
