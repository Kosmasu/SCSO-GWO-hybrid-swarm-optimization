import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_pareto_front(solutions, algorithm_name, output_dir):
    """
    Plot the Pareto front for a given algorithm
    
    Args:
        solutions: List of solutions (format adapted for each algorithm)
        algorithm_name: Name of the algorithm for display
        output_dir: Directory to save the plot
    """
    # Extract objective values (handling different solution formats)
    workstations = [int(sol["fitness"][0]) for sol in solutions]
    cycle_times = [sol["fitness"][1] for sol in solutions]

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
    plt.title(f'Pareto Front - {algorithm_name}', fontsize=14)
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
    save_path = f"{output_dir}/pareto_front.png"
    print(f"Saving Pareto front plot to {save_path}...")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pareto_2d(pareto, all_solutions, algorithm_name, output_dir):
    """
    Plot the Pareto front and all evaluated solutions.
    
    Args:
        pareto: Pareto optimal solutions
        all_solutions: All evaluated solutions (if available)
        algorithm_name: Name of the algorithm
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract Pareto front objective values based on algorithm
    pareto_ws = [int(sol["fitness"][0]) for sol in pareto]
    pareto_ct = [sol["fitness"][1] for sol in pareto]
    
    # If all solutions are provided, plot them first as gray dots
    if all_solutions is not None:
        all_ws = [int(sol["fitness"][0]) for sol in all_solutions]
        all_ct = [sol["fitness"][1] for sol in all_solutions]
        plt.scatter(all_ws, all_ct, color='gray', s=30, alpha=0.5, label='Solutions')
    
    # Plot Pareto front as blue dots
    plt.scatter(pareto_ws, pareto_ct, color='blue', s=50, alpha=0.9, label='Pareto Front')
    
    # Add labels and title
    plt.xlabel('Workstations', fontsize=12)
    plt.ylabel('Cycle Time', fontsize=12)
    plt.title(f'2D Pareto Front - {algorithm_name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    save_path = f'{output_dir}/pareto_2d.png'
    print(f"Saving 2D Pareto plot to {save_path}...")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_workstation_tasks(solution, problem, algorithm_name, output_dir, filename):
    """
    Visualize the tasks assigned to each workstation in a horizontal bar chart.
    
    Args:
        solution: A solution (format depends on the algorithm)
        problem: The MOAssemblyLineBalancingProblem instance
        algorithm_name: Name of the algorithm
        output_dir: Directory to save the plot
        filename: Filename to save the plot
    """
    # Decode the solution based on algorithm
    decoded = problem.decode(solution["position"])
    cycle_time = solution["fitness"][1]
    
    workstations = problem.assign_tasks(decoded)
    
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
    ax.set_title(f'{algorithm_name} - Tasks in Each Workstation at Cycle Time:{cycle_time:.1f}')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = f"{output_dir}/{filename}"
    print(f"Saving workstation tasks plot to {save_path}...")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparative_pareto(hybrid_pareto, nsga2_pareto, spea2_pareto, output_dir):
    """
    Generate a comparative plot of Pareto fronts from all three algorithms.
    
    Args:
        hybrid_pareto: Pareto front from Hybrid GWO-SCSO
        nsga2_pareto: Pareto front from NSGA-II
        spea2_pareto: Pareto front from SPEA2
        output_dir: Directory to save the output
    """
    plt.figure(figsize=(12, 8))
    
    # Extract objectives from each algorithm
    hybrid_ws = [int(sol["fitness"][0]) for sol in hybrid_pareto]
    hybrid_ct = [sol["fitness"][1] for sol in hybrid_pareto]
    
    nsga2_ws = [int(sol.objectives[0]) for sol in nsga2_pareto]
    nsga2_ct = [sol.objectives[1] for sol in nsga2_pareto]
    
    spea2_ws = [int(sol.objectives[0]) for sol in spea2_pareto]
    spea2_ct = [sol.objectives[1] for sol in spea2_pareto]
    
    # Plot all Pareto fronts
    plt.scatter(hybrid_ws, hybrid_ct, s=80, color='blue', edgecolor='black', alpha=0.7, label='Hybrid GWO-SCSO')
    plt.scatter(nsga2_ws, nsga2_ct, s=80, color='red', edgecolor='black', alpha=0.7, label='NSGA-II')
    plt.scatter(spea2_ws, spea2_ct, s=80, color='green', edgecolor='black', alpha=0.7, label='SPEA2')
    
    # Create connecting lines for each
    for ws, ct, color, name in [(hybrid_ws, hybrid_ct, 'blue', 'Hybrid'), 
                                (nsga2_ws, nsga2_ct, 'red', 'NSGA-II'), 
                                (spea2_ws, spea2_ct, 'green', 'SPEA2')]:
        if len(ws) > 1:  # Only if there are at least 2 points
            sorted_points = sorted(zip(ws, ct), key=lambda x: x[0])
            sorted_ws = [p[0] for p in sorted_points]
            sorted_ct = [p[1] for p in sorted_points]
            plt.plot(sorted_ws, sorted_ct, '--', color=color, alpha=0.5, label=f'{name} Pareto')
    
    # Add labels and title
    plt.xlabel('Number of Workstations', fontsize=14)
    plt.ylabel('Cycle Time (seconds)', fontsize=14)
    plt.title('Comparison of Pareto Fronts', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    save_path = f"{output_dir}/comparative_pareto.png"
    print(f"Saving comparative Pareto front plot to {save_path}...")
    plt.savefig(save_path, dpi=300)
    plt.close()

def convert_platypus_to_dict(platypus_solutions):
    """Convert Platypus solutions to hybrid-like dictionary format"""
    return [
        {"position": sol.variables, "fitness": sol.objectives}
        for sol in platypus_solutions
    ]
