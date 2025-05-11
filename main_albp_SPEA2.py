from platypus import SPEA2
from problem_mo_albp import MOAssemblyLineBalancingProblem
from pareto_utils import filter_results

task_times = {
    1: 44.4,  2: 54.8,  3: 39.56,  4: 15.6,  5: 32.3, 6: 28,    7: 28.7,
    8: 44.4,  9: 58.8, 10: 73.1,  11: 85.6, 12: 89.8, 13: 30.2, 14: 82.7,
   15: 57.3, 16: 78.5, 17: 41.4,  18: 55.1, 19: 72.7, 20: 40.4, 21: 34.4,
   22: 29.4, 23: 108,  24: 93.2, 25: 39.8, 26: 34.7, 27: 40.5, 28: 110,
   29: 55.6
}

precedence = {
    1: [], 2: [], 3: [1, 2], 4: [], 5: [3, 4], 6: [5], 7: [6], 8: [7], 9: [8], 10: [9],
    11: [10], 12: [11], 13: [12], 14: [13], 15: [14], 16: [15], 17: [], 18: [17],
    19: [], 20: [19], 21: [18, 20], 22: [21], 23: [16, 22], 24: [23], 25: [24],
    26: [25], 27: [26], 28: [27], 29: [28]
}

problem = MOAssemblyLineBalancingProblem(task_times, precedence, cycle_time_upper_bound=120)

algorithm = SPEA2(problem, population_size=100)
algorithm.run(9999)  # or adjust iteration count

filtered = filter_results(algorithm.result, epsilons=(0.0, 0.0))

# Print Pareto Front
for i, solution in enumerate(filtered):
    obj = solution.objectives
    print(f"\n🔹 SPEA2 Solution {i+1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s")
    decoded = problem.decode(solution.variables)
    workstations = problem.assign_tasks(decoded)
    for idx, ws in enumerate(workstations):
        total_time = sum(task_times[t] for t in ws)
        print(f"  Workstation {idx+1} [{total_time:.1f}s]: {', '.join(f'Task {t}' for t in ws)}")
