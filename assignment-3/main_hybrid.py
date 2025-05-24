from problem_MO import MOAssemblyLineBalancingProblem
from hybrid import hybrid_gwo_scso
from data import TASK_TIMES, PRECEDENCE, STAGNATION_LIMIT # MAX_EPOCH and POPULATION_SIZE will be from configs
from utils import create_output_directory
from visualization import plot_pareto_2d, plot_pareto_front, plot_workstation_tasks
import os
import csv
import time

# Original ALGORITHM_NAME and OUTPUT_DIR can be used as a base or for a default run if needed
BASE_ALGORITHM_NAME = "Hybrid GWO-SCSO"
BASE_OUTPUT_DIR = "output/albp/hybrid" # Base directory for benchmarks

# Benchmark configurations: (pop_size, max_epoch)
BENCHMARK_CONFIGS = [
    (50,  100),
    (25,  150),
    (50,  150),
    (100, 150),
    (200, 150),
    (25,  300),
    (50,  300),
    (100, 300),
    (200, 300),
    (25,  500),
    (50,  500),
    (100, 500),
    (200, 500),
    (25,  1000),
    (50,  1000),
    (100, 1000),
    (200, 1000),
]

# Ensure the base directory for CSV and benchmark outputs exists
create_output_directory(BASE_OUTPUT_DIR)

benchmark_results_list = []

problem_instance = MOAssemblyLineBalancingProblem(
    TASK_TIMES, PRECEDENCE, cycle_time_upper_bound=120
)

for pop_size_config, max_epoch_config in BENCHMARK_CONFIGS:
    benchmark_run_name = f"pop{pop_size_config}_epoch{max_epoch_config}"
    current_algo_name = f"{BASE_ALGORITHM_NAME} ({benchmark_run_name})"
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, benchmark_run_name)

    create_output_directory(current_output_dir)

    print(f"\nRunning Benchmark: {current_algo_name}")
    print(f"Parameters: Population Size = {pop_size_config}, Max Epoch = {max_epoch_config}")
    print(f"Output will be saved to: {current_output_dir}")

    start_time = time.perf_counter()
    pareto, all_solutions = hybrid_gwo_scso(
        problem_instance, 
        pop_size=pop_size_config, 
        max_epoch=max_epoch_config, 
        stagnation_limit=STAGNATION_LIMIT, # Using STAGNATION_LIMIT from data.py
        epsilons=(0, 0)
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Benchmark {current_algo_name} finished in {duration:.2f} seconds.")

    # Store results for CSV
    pareto_objectives_str = str([list(sol["fitness"]) for sol in pareto])
    benchmark_results_list.append({
        "Algorithm": BASE_ALGORITHM_NAME, # Store base name for easier grouping
        "Population Size": pop_size_config,
        "Max Epoch": max_epoch_config,
        "Time (s)": f"{duration:.2f}",
        "Num Pareto Solutions": len(pareto),
        "Pareto Objectives": pareto_objectives_str
    })

    print(f"\n--- Results for {current_algo_name} ---")
    # print("pareto:", pareto) # Optional: print full pareto details
    # print("all_solutions[0:5]:", all_solutions[0:5]) # Optional: print some all_solutions
    print(f"len(pareto): {len(pareto)}")
    print(f"len(all_solutions): {len(all_solutions)}")


    for i, sol in enumerate(pareto):
        obj = sol["fitness"]
        print(
            f"\n🔹 Solution {i + 1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s"
        )
        # Decoding and printing workstations can be verbose for many benchmarks,
        # but kept for consistency with original script's output per run.
        decoded = problem_instance.decode(sol["position"])
        workstations = problem_instance.assign_tasks(decoded)
        for idx, ws in enumerate(workstations):
            total_time = sum(TASK_TIMES[t] for t in ws)
            print(
                f"  Workstation {idx + 1} [{total_time:.1f}s]: {', '.join(f'Task {t}' for t in ws)}"
            )

    plot_pareto_front(pareto, current_algo_name, current_output_dir)
    plot_pareto_2d(pareto, all_solutions, current_algo_name, current_output_dir)
    for idx, solution in enumerate(pareto, start=1):
        plot_workstation_tasks(
            solution,
            problem_instance,
            algorithm_name=current_algo_name,
            output_dir=current_output_dir,
            filename=f"workstation_tasks_{idx}.png",
        )
    print(f"Plots for {current_algo_name} saved to {current_output_dir}")


# Write benchmark results to CSV
csv_file_path = os.path.join(BASE_OUTPUT_DIR, "hybrid_benchmarks.csv")
print(f"\nWriting benchmark results to {csv_file_path}")
if benchmark_results_list:
    fieldnames = benchmark_results_list[0].keys()
    try:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(benchmark_results_list)
        print("Benchmark results successfully written to CSV.")
    except IOError:
        print(f"Error writing benchmark results to {csv_file_path}.")
else:
    print("No benchmark results to write.")
