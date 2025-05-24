from platypus import Problem, Real
import numpy as np

class MOAssemblyLineBalancingProblem(Problem):
    def __init__(self, task_times, precedence, cycle_time_upper_bound=999.0):
        self.task_times = task_times
        self.precedence = precedence
        self.num_tasks = len(task_times)
        self.cycle_time_upper_bound = cycle_time_upper_bound

        super().__init__(self.num_tasks, 2)  # num_vars, num_objectives
        self.types[:] = [Real(0.0, 1.0) for _ in range(self.num_tasks)]
        self.directions[:] = [self.MINIMIZE, self.MINIMIZE]

    def evaluate(self, solution):
        weights = solution.variables
        permutation = self.decode(weights)
        workstations = self.assign_tasks(permutation)
        num_ws = len(workstations)
        bottleneck = max(sum(self.task_times[t] for t in ws) for ws in workstations)
        solution.objectives[:] = [num_ws, bottleneck]

    def decode(self, weights):
        """Turn float vector [0-1] into task ID permutation"""
        indices = list(range(1, self.num_tasks + 1))
        sorted_tasks = [x for _, x in sorted(zip(weights, indices))]
        return sorted_tasks

    def assign_tasks(self, permutation):
        assigned = set()
        workstations = []
        current_ws = []
        current_time = 0.0

        while len(assigned) < self.num_tasks:
            progress = False
            for task in permutation:
                if task in assigned:
                    continue
                if all(pred in assigned for pred in self.precedence[task]):
                    t_time = self.task_times[task]
                    if current_time + t_time <= self.cycle_time_upper_bound:
                        current_ws.append(task)
                        current_time += t_time
                        assigned.add(task)
                        progress = True
            if not progress or current_ws:
                workstations.append(current_ws)
                current_ws = []
                current_time = 0.0
        return workstations