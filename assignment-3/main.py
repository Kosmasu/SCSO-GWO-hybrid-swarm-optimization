import time
import csv
import os

from problem_MO import MOAssemblyLineBalancingProblem
from hybrid import hybrid_gwo_scso
from data import TASK_TIMES, PRECEDENCE, MAX_EPOCH, POPULATION_SIZE, STAGNATION_LIMIT
from utils import create_output_directory, filter_results
from visualization import plot_pareto_2d, plot_pareto_front, plot_workstation_tasks

from platypus import IBEA, NSGAII, SPEA2

# Wrapper class to log all evaluated solutions for Platypus algorithms
class ProblemWithSolutionLogging(MOAssemblyLineBalancingProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_solutions = []

    def evaluate(self, solution):
        super().evaluate(solution) 
        if hasattr(solution, 'objectives') and solution.objectives is not None:
            self.logged_solutions.append({
                "position": list(solution.variables),
                "fitness": list(solution.objectives)
            })

def main():
    timing_results = []
    base_csv_output_dir = "output/albp"
    create_output_directory(base_csv_output_dir) # Ensure base directory for CSV exists

    # --- Run Hybrid GWO-SCSO ---
    algo_name_hybrid = "Hybrid GWO-SCSO"
    output_dir_hybrid = os.path.join(base_csv_output_dir, "hybrid")
    create_output_directory(output_dir_hybrid)

    print(f"\nRunning {algo_name_hybrid}...")
    problem_hybrid = MOAssemblyLineBalancingProblem(
        TASK_TIMES, PRECEDENCE, cycle_time_upper_bound=120
    )
    
    start_time = time.perf_counter()
    pareto_hybrid, all_solutions_hybrid = hybrid_gwo_scso(
        problem_hybrid, pop_size=POPULATION_SIZE, max_epoch=MAX_EPOCH, stagnation_limit=STAGNATION_LIMIT, epsilons=(0, 0)
    )
    end_time = time.perf_counter()
    duration_hybrid = end_time - start_time
    timing_results.append({"Algorithm": algo_name_hybrid, "Time (s)": duration_hybrid})
    print(f"{algo_name_hybrid} finished in {duration_hybrid:.2f} seconds.")

    print(f"\n--- Results for {algo_name_hybrid} ---")
    for i, sol in enumerate(pareto_hybrid):
        obj = sol["fitness"]
        print(
            f"\n🔹 Solution {i + 1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s"
        )
        decoded = problem_hybrid.decode(sol["position"])
        workstations = problem_hybrid.assign_tasks(decoded)
        for idx, ws in enumerate(workstations):
            total_time_ws = sum(TASK_TIMES[t] for t in ws)
            print(
                f"  Workstation {idx + 1} [{total_time_ws:.1f}s]: {', '.join(f'Task {t}' for t in ws)}"
            )
    
    plot_pareto_front(pareto_hybrid, algo_name_hybrid, output_dir_hybrid)
    plot_pareto_2d(pareto_hybrid, all_solutions_hybrid, algo_name_hybrid, output_dir_hybrid)
    for idx, solution in enumerate(pareto_hybrid, start=1):
        plot_workstation_tasks(
            solution,
            problem_hybrid,
            algorithm_name=algo_name_hybrid,
            output_dir=output_dir_hybrid,
            filename=f"workstation_tasks_{idx}.png",
        )

    # --- Run Platypus Algorithms ---
    platypus_algorithms = [
        ("IBEA", IBEA),
        ("NSGA-II", NSGAII),
        ("SPEA2", SPEA2)
    ]

    for algo_name_platypus, algorithm_class in platypus_algorithms:
        output_dir_platypus = os.path.join(base_csv_output_dir, algo_name_platypus.lower().replace('-', ''))
        create_output_directory(output_dir_platypus)

        print(f"\nRunning {algo_name_platypus}...")
        problem_platypus = ProblemWithSolutionLogging(
            TASK_TIMES, PRECEDENCE, cycle_time_upper_bound=120
        )
        
        total_evals = MAX_EPOCH * POPULATION_SIZE
        platypus_algo_instance = algorithm_class(problem_platypus, POPULATION_SIZE)
        
        start_time = time.perf_counter()
        platypus_algo_instance.run(total_evals)
        end_time = time.perf_counter()
        duration_platypus = end_time - start_time
        timing_results.append({"Algorithm": algo_name_platypus, "Time (s)": duration_platypus})
        print(f"{algo_name_platypus} finished in {duration_platypus:.2f} seconds.")

        platypus_results_raw = platypus_algo_instance.result
        pareto_platypus_objects = filter_results(platypus_results_raw, epsilons=(0.0, 0.0))
        
        all_solutions_platypus = problem_platypus.logged_solutions
        pareto_solutions_platypus_formatted = [{"position": s.variables, "fitness": s.objectives} for s in pareto_platypus_objects]

        print(f"\n--- Results for {algo_name_platypus} ---")
        for i, sol_obj in enumerate(pareto_platypus_objects):
            obj = sol_obj.objectives
            print(f"\n🔹 {algo_name_platypus} Solution {i+1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s")
            decoded = problem_platypus.decode(sol_obj.variables)
            workstations = problem_platypus.assign_tasks(decoded)
            for idx, ws in enumerate(workstations):
                total_time_ws = sum(TASK_TIMES[t] for t in ws)
                print(f"  Workstation {idx+1} [{total_time_ws:.1f}s]: {', '.join(f'Task {t}' for t in ws)}")

        plot_pareto_front(pareto_solutions_platypus_formatted, algo_name_platypus, output_dir_platypus)
        plot_pareto_2d(pareto_solutions_platypus_formatted, all_solutions_platypus, algo_name_platypus, output_dir_platypus)
        for idx, sol_dict in enumerate(pareto_solutions_platypus_formatted, start=1):
            plot_workstation_tasks(
                sol_dict,
                problem_platypus, # Pass the problem instance used by this algorithm
                algorithm_name=algo_name_platypus,
                output_dir=output_dir_platypus,
                filename=f"workstation_tasks_{idx}.png",
            )

    # --- Write timing results to CSV ---
    csv_file_path = os.path.join(base_csv_output_dir, "time.csv")
    print(f"\nWriting timing results to {csv_file_path}")
    try:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Algorithm", "Time (s)"])
            writer.writeheader()
            writer.writerows(timing_results)
        print("Timing results successfully written.")
    except IOError:
        print(f"Error writing timing results to {csv_file_path}.")

if __name__ == "__main__":
    main()
