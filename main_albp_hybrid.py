from problem_mo_albp import MOAssemblyLineBalancingProblem
from hybrid_gwoscso_albp import Archive, Individual, hybrid_gwo_scso
import numpy as np
import os
import shutil

# Create output directories if they don't exist
os.makedirs("output/albp", exist_ok=True)

# Clear all files in output/albp directory
for filename in os.listdir("output/albp"):
    file_path = os.path.join("output/albp", filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {e}")

# Task data
task_times = {
    1: 44.4,  2: 54.8,  3: 39.56,  4: 15.6,  5: 32.3,
    6: 28,    7: 28.7,  8: 44.4,   9: 58.8, 10: 73.1,
11: 85.6, 12: 89.8, 13: 30.2,  14: 82.7, 15: 57.3,
16: 78.5, 17: 41.4, 18: 55.1,  19: 72.7, 20: 40.4,
21: 34.4, 22: 29.4, 23: 108,   24: 93.2, 25: 39.8,
26: 34.7, 27: 40.5, 28: 110,   29: 55.6
}

precedence = {
    1: [], 2: [], 3: [1, 2], 4: [], 5: [3, 4], 6: [5], 7: [6], 8: [7],
    9: [8], 10: [9], 11: [10], 12: [11], 13: [12], 14: [13], 15: [14],
    16: [15], 17: [], 18: [17], 19: [], 20: [19], 21: [18, 20],
    22: [21], 23: [16, 22], 24: [23], 25: [24], 26: [25], 27: [26],
    28: [27], 29: [28]
}

problem = MOAssemblyLineBalancingProblem(task_times, precedence, cycle_time_upper_bound=120)
pareto, all_solutions = hybrid_gwo_scso(problem, pop_size=100, max_epoch=9999, stagnation_limit=15, epsilons=(0, 0))

for i, sol in enumerate(pareto):
    obj = sol["fitness"]
    print(f"\n🔹 Solution {i+1}: {int(obj[0])} Workstations, Cycle Time = {obj[1]:.2f}s")
    decoded = problem.decode(sol["position"])
    workstations = problem.assign_tasks(decoded)
    for idx, ws in enumerate(workstations):
        total_time = sum(task_times[t] for t in ws)
        print(f"  Workstation {idx+1} [{total_time:.1f}s]: {', '.join(f'Task {t}' for t in ws)}")

# Plotting the Pareto front
import matplotlib.pyplot as plt

def plot_pareto_front(pareto: Archive):
    # Extract objective values
    workstations = [int(sol["fitness"][0]) for sol in pareto]
    cycle_times = [sol["fitness"][1] for sol in pareto]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(workstations, cycle_times, s=100, color='blue', edgecolor='black', alpha=0.7)

    # Add labels to points
    for i, (ws, ct) in enumerate(zip(workstations, cycle_times)):
        plt.annotate(f"{i+1}", (ws, ct), fontsize=9)

    # Sort points for connecting line
    sorted_points = sorted(zip(workstations, cycle_times), key=lambda x: x[0])
    sorted_ws = [p[0] for p in sorted_points]
    sorted_ct = [p[1] for p in sorted_points]

    # Connect points to show Pareto front
    plt.plot(sorted_ws, sorted_ct, 'r--', alpha=0.5)

    # Add labels and title
    plt.xlabel('Number of Workstations', fontsize=12)
    plt.ylabel('Cycle Time (seconds)', fontsize=12)
    plt.title('Pareto Front - Assembly Line Balancing', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Highlight extreme solutions
    min_ws_idx = workstations.index(min(workstations))
    min_ct_idx = cycle_times.index(min(cycle_times))
    plt.scatter([workstations[min_ws_idx]], [cycle_times[min_ws_idx]], s=150, color='green', 
                edgecolor='black', label='Min Workstations')
    plt.scatter([workstations[min_ct_idx]], [cycle_times[min_ct_idx]], s=150, color='red', 
                edgecolor='black', label='Min Cycle Time')

    plt.legend()
    plt.tight_layout()
    plt.savefig("output/albp/pareto_front.png", dpi=300)
    plt.show()

# Call the function
plot_pareto_front(pareto)

def plot_pareto_2d(pareto: Archive, all_solutions=None):
    """
    Plot the Pareto front and all evaluated solutions.
    
    Args:
        pareto: Archive object containing Pareto optimal solutions
        all_solutions: List of all evaluated solutions during optimization
    """
    plt.figure(figsize=(10, 6))
    
    # Extract Pareto front objective values
    pareto_ws = [int(sol["fitness"][0]) for sol in pareto]
    pareto_ct = [sol["fitness"][1] for sol in pareto]
    
    # If all solutions are provided, plot them first as gray dots
    if all_solutions:
        all_ws = [int(sol["fitness"][0]) for sol in all_solutions]
        all_ct = [sol["fitness"][1] for sol in all_solutions]
        plt.scatter(all_ws, all_ct, color='gray', s=30, alpha=0.7, label='Solutions')
    
    # Plot Pareto front as blue dots
    plt.scatter(pareto_ws, pareto_ct, color='blue', s=50, alpha=0.9, label='Pareto Front')
    
    # Add labels and title
    plt.xlabel('Workstations', fontsize=12)
    plt.ylabel('Cycle Time', fontsize=12)
    plt.title('2D Pareto Front', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/albp/pareto_2d.png', dpi=300)
    plt.show()

plot_pareto_2d(pareto, all_solutions)

def plot_workstation_tasks(solution: Individual, problem: MOAssemblyLineBalancingProblem, filename: str = "workstation_tasks.png"):
    """
    Visualize the tasks assigned to each workstation in a horizontal bar chart.
    
    Args:
        solution: A solution dictionary containing position and fitness information
        problem: The MOAssemblyLineBalancingProblem instance
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Decode the solution
    decoded = problem.decode(solution["position"])
    workstations = problem.assign_tasks(decoded)
    cycle_time = solution["fitness"][1]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate colors for tasks
    np.random.seed(42)  # For consistent colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    task_colors = {task_id: colors[i % 20] for i, task_id in enumerate(problem.task_times.keys())}
    
    # Plot tasks for each workstation
    for ws_idx, ws_tasks in enumerate(workstations):
        y_pos = len(workstations) - ws_idx  # Workstation 1 at the bottom
        current_time = 0
        
        for task_id in ws_tasks:
            task_time = problem.task_times[task_id]
            
            # Create a rectangle for the task
            rect = patches.Rectangle(
                (current_time, y_pos - 0.4),  # (x, y)
                task_time,  # width
                0.8,  # height
                linewidth=1,
                edgecolor='black',
                facecolor=task_colors[task_id],
                alpha=0.7
            )
            
            # Add the rectangle to the plot
            ax.add_patch(rect)
            
            # Add task label
            ax.text(
                current_time + task_time/2,  # x-center of the bar
                y_pos,  # y-center of the bar
                f"{task_id} ({task_time:.1f}s)",
                ha='center', va='center',
                fontsize=9, color='black'
            )
            
            # Update current time position
            current_time += task_time
    
    # Set plot limits and labels
    ax.set_xlim(0, cycle_time * 1.05)  # Add 5% margin
    ax.set_ylim(0, len(workstations) + 1)
    ax.set_yticks(range(1, len(workstations) + 1))
    ax.set_yticklabels([f"Workstation {len(workstations) - i + 1}" for i in range(1, len(workstations) + 1)])
    ax.set_xlabel('Task Time')
    ax.set_title(f'Tasks in Each Workstation at Cycle Time:{cycle_time:.1f}')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"output/albp/{filename}", dpi=300)
    plt.show()

for idx, solution in enumerate(pareto, start=1):
    plot_workstation_tasks(solution, problem, filename=f"albp/workstation_tasks_{idx}.png")