from mealpy import PSO, GA, DE, GWO, SCSO

from mealpy.optimizer import Optimizer
import csv
import os
import pickle

from problems import (
    Ackley,
    CustomProblem,
    ExpandedSchaffer,
    Griewank,
    HappyCat,
    ModifiedSchwefel,
    Rosenbrock,
    Squared,
    Rastrigin,
    Weierstrass,
)
from hybrid import HybridGWOSCSO  # NORMAL GWO + SCSO
from hybrid_2 import HybridIGWOSCSO  # IMPROVED GWO + SCSO
from hybrid_v3 import HybridGWOSCSO3  # HYBRID GWO - SCSO With SCSO starting phase

EPOCH = 99999
POPULATION = [25, 50, 75]
DIMENSION = [2, 5, 10, 20, 30, 50, 100]
ITERATIONS = 30
PROBLEMS: list[type[CustomProblem]] = [
    Squared,
    Rastrigin,
    Griewank,
    Weierstrass,
    ModifiedSchwefel,
    ExpandedSchaffer,
    HappyCat,
    Ackley,
    Rosenbrock,
]


def initialize_problem(
    problem_class: type[CustomProblem], dimension: int = 10
) -> CustomProblem:
    problem = problem_class(dimension=dimension)
    return problem


MODELS: list[type[Optimizer]] = [
    PSO.OriginalPSO,
    GA.BaseGA,
    DE.OriginalDE,
    GWO.OriginalGWO,
    SCSO.OriginalSCSO,
    HybridGWOSCSO,
    HybridIGWOSCSO,
    HybridGWOSCSO3,
]

results_dir = "output/experiment_results"
os.makedirs(results_dir, exist_ok=True)
csv_file = os.path.join(results_dir, "experiment_log.csv")

# Define CSV header
header = [
    "problem",
    "dimension",
    "model",
    "run",
    "best_fitness",
    "runtime",
    "epochs",
    "fitness_diff_to_gt"
]

term_dict = {
    "max_early_stop": 50,
    "epsilon": 1e-10
}

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for model in MODELS:
        for problem_class in PROBLEMS:
            for dimension in DIMENSION:
                for pop_size in POPULATION:
                    for i in range(ITERATIONS):
                        problem = initialize_problem(problem_class, dimension)
                        # Nested folder structure: model/problem/dimension/population_size/iteration
                        checkpoint_dir = os.path.join(
                            "output",
                            "checkpoints",
                            model.__name__,
                            problem.name,
                            f"dim{dimension}",
                            f"pop{pop_size}",
                            f"run_{i+1}"
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_file = os.path.join(checkpoint_dir, "history.pkl")

                        # Skip iteration if checkpoint exists
                        if os.path.exists(checkpoint_file):
                            print(f"Checkpoint exists for {checkpoint_file}, skipping iteration.")
                            continue

                        model_instance = model(epoch=EPOCH, pop_size=pop_size)
                        g_best = model_instance.solve(problem=problem, seed=i, termination=term_dict)
                        if not model_instance.problem or not model_instance.history:
                            raise ValueError("Problem or history not set in the model.")

                        # Save history to pickle file as checkpoint
                        with open(checkpoint_file, "wb") as cp_file:
                            pickle.dump(model_instance.history, cp_file)

                        # Prepare output directory for this run (same as checkpoint_dir for clarity)
                        run_dir = checkpoint_dir
                        os.makedirs(run_dir, exist_ok=True)

                        # Visualization
                        vis_files = []
                        vis_files.append(model_instance.history.save_global_objectives_chart(filename=os.path.join(run_dir, "goc")))
                        vis_files.append(model_instance.history.save_local_objectives_chart(filename=os.path.join(run_dir, "loc")))
                        vis_files.append(model_instance.history.save_global_best_fitness_chart(filename=os.path.join(run_dir, "gbfc")))
                        vis_files.append(model_instance.history.save_local_best_fitness_chart(filename=os.path.join(run_dir, "lbfc")))
                        vis_files.append(model_instance.history.save_runtime_chart(filename=os.path.join(run_dir, "rtc")))
                        vis_files.append(model_instance.history.save_exploration_exploitation_chart(filename=os.path.join(run_dir, "eec")))
                        vis_files.append(model_instance.history.save_diversity_chart(filename=os.path.join(run_dir, "dc")))
                        vis_files.append(
                            model_instance.history.save_trajectory_chart(
                                list_agent_idx=[3, 5, 6, 7],
                                selected_dimensions=[3, 4],
                                filename=os.path.join(run_dir, "tc"),
                            )
                        )

                        # Calculate difference to ground truth (which is 0)
                        fitness_diff_to_gt = abs(model_instance.g_best.target.fitness - 0)

                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    problem.name,
                                    dimension,
                                    model.__name__,
                                    i + 1,
                                    model_instance.g_best.target.fitness,
                                    sum(model_instance.history.list_epoch_time),
                                    model_instance.history.epoch,
                                    fitness_diff_to_gt
                                ]
                            )
