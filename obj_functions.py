"""
Custom benchmark objective functions for optimization algorithms.

This module contains implementations of various benchmark functions commonly used
to test optimization algorithms, such as Rastrigin, Styblinski-Tang, and others.
These functions can be used with the mealpy optimization library.
"""

import os
from typing import Union
from mealpy import Problem, FloatVar, PSO
import numpy as np
from mealpy.utils.space import BaseVar


class CustomProblem(Problem):
    def __init__(
        self,
        lb: float,
        ub: float,
        dimensions: int,
        minmax: str = "min",
        save_population=True,
        **kwargs,
    ):
        bounds = FloatVar(ub=(ub,) * dimensions, lb=(lb,) * dimensions, name="bounds")
        super().__init__(bounds, minmax, save_population=save_population, **kwargs)
        self.name = kwargs.get("name", "CustomProblem")


class Squared(CustomProblem):
    def __init__(
        self,
        lb: float,
        ub: float,
        dimensions: int,
        minmax: str = "min",
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            minmax=minmax,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Squared_{dimensions}d")

    def obj_func(self, x):
        return np.sum(x**2)


class Rastrigin(CustomProblem):
    def __init__(
        self,
        lb: float,
        ub: float,
        dimensions: int,
        minmax: str = "min",
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            minmax=minmax,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Rastrigin_{dimensions}d")

    def obj_func(self, x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class StyblinskiTang(Problem):
    def __init__(
        self,
        lb: float,
        ub: float,
        dimensions: int,
        minmax: str = "min",
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            minmax=minmax,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Rastrigin_{dimensions}d")

    def obj_func(self, x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


if __name__ == "__main__":
    problem = Squared(-10, 10, 5, minmax="min", name="Squared", data="Amazing")
    model = PSO.OriginalPSO(epoch=300, pop_size=20)
    g_best = model.solve(problem)

    if not model.problem or not model.history:
        raise ValueError("Problem or history not set in the model.")

    print("Solution:", g_best.solution)
    print("Target fitness:", g_best.target.fitness)
    print("Target objectives:", g_best.target.objectives)
    print("Best solution (g_best):", g_best)
    print("Model parameters:", model.get_parameters())
    print("Model name:", model.get_name())
    print("Global best from attributes:", model.get_attributes()["g_best"])
    print("Problem name:", model.problem.get_name())
    print("Problem dimensions:", model.problem.n_dims)
    print("Problem bounds:", model.problem.bounds)
    print("Problem lower bounds:", model.problem.lb)
    print("Problem upper bounds:", model.problem.ub)

    os.makedirs("output/test", exist_ok=True)

    # Visualization
    model.history.save_global_objectives_chart(filename="output/test/goc")
    model.history.save_local_objectives_chart(filename="output/test/loc")
    model.history.save_global_best_fitness_chart(filename="output/test/gbfc")
    model.history.save_local_best_fitness_chart(filename="output/test/lbfc")
    model.history.save_runtime_chart(filename="output/test/rtc")
    model.history.save_exploration_exploitation_chart(filename="output/test/eec")
    model.history.save_diversity_chart(filename="output/test/dc")
    model.history.save_trajectory_chart(
        list_agent_idx=[
            3,
            5,
            6,
            7,
        ],
        selected_dimensions=[3, 4],
        filename="output/test/tc",
    )
