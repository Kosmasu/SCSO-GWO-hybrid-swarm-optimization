"""
Custom benchmark objective functions for optimization algorithms.

This module contains implementations of various benchmark functions commonly used
to test optimization algorithms, such as Rastrigin, Styblinski-Tang, and others.
These functions can be used with the mealpy optimization library.

Most functions are from the CEC 2019 The 100-Digit Challenge
Price, K. V., Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2018). Problem definitions and evaluation criteria for the 100-digit challenge special session and competition on single objective numerical optimization. In Technical report. Singapore: Nanyang Technological University.

All functions are defined as classes inheriting from the `CustomProblem` class,
which provides a standard interface for optimization problems.

All functions globally minimize with a value of 0
"""

import os
from mealpy import Problem, FloatVar, GWO
import numpy as np


class CustomProblem(Problem):
    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        minmax="min",
        **kwargs,
    ):
        bounds = FloatVar(ub=(ub,) * dimensions, lb=(lb,) * dimensions, name="bounds")
        super().__init__(
            bounds, minmax=minmax, save_population=save_population, **kwargs
        )
        self.name = kwargs.get("name", "CustomProblem")


class Squared(CustomProblem):
    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Squared_{dimensions}d")

    def obj_func(self, x):
        return np.sum(x**2)


class InverseHilbert(CustomProblem):
    """
    Inverse Hilbert Matrix Problem (Function 2)
    Note: Only defined for D = 16 and other perfect square dimensions
    """

    def __init__(
        self,
        lb: float = -16384,
        ub: float = 16384,
        dimensions: int = 16,
        save_population=True,
        **kwargs,
    ):
        # Check if dimensions are valid (D should be a perfect square)
        n = int(np.sqrt(dimensions))
        if n * n != dimensions:
            raise ValueError(
                f"InverseHilbert function requires D to be a perfect square. Got D={dimensions}"
            )

        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"InverseHilbert_{dimensions}d")

    def obj_func(self, x):
        D = len(x)
        n = int(np.sqrt(D))

        # Create Hilbert matrix H
        H = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                H[i, k] = 1.0 / (i + k + 1)

        # Create matrix Z from x
        Z = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                Z[i, k] = x[i + n * k]

        # Calculate W = HZ - I
        I = np.eye(n)
        W = np.dot(H, Z) - I

        # Calculate sum of absolute values
        return np.sum(np.abs(W))


class Rastrigin(CustomProblem):
    """
    Rastrigin's Function (Function 4)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Rastrigin_{dimensions}d")

    def obj_func(self, x):
        return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)


class Griewank(CustomProblem):
    """
    Griewank's Function (Function 5)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Griewank_{dimensions}d")

    def obj_func(self, x):
        sum_part = np.sum(x**2) / 4000
        prod_part = 1
        for i in range(len(x)):
            prod_part *= np.cos(x[i] / np.sqrt(i + 1))
        return sum_part - prod_part + 1


class Weierstrass(CustomProblem):
    """
    Weierstrass Function (Function 6)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Weierstrass_{dimensions}d")

    def obj_func(self, x):
        D = len(x)
        a = 0.5
        b = 3
        kmax = 20

        # First sum term
        sum1 = 0
        for i in range(D):
            inner_sum = 0
            for k in range(kmax + 1):
                inner_sum += a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5))
            sum1 += inner_sum

        # Second sum term (constant)
        sum2 = 0
        for k in range(kmax + 1):
            sum2 += a**k * np.cos(np.pi * b**k)
        sum2 *= D

        return sum1 - sum2


class ModifiedSchwefel(CustomProblem):
    """
    Modified Schwefel's Function (Function 7)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"ModifiedSchwefel_{dimensions}d")

    def g(self, z_i, D):
        if abs(z_i) <= 500:
            return z_i * np.sin(np.sqrt(abs(z_i)))
        elif z_i > 500:
            return (500 - z_i % 500) * np.sin(np.sqrt(abs(500 - z_i % 500))) - (
                z_i - 500
            ) ** 2 / (10000 * D)
        else:  # z_i < -500
            return (abs(z_i) % 500 - 500) * np.sin(
                np.sqrt(abs(abs(z_i) % 500 - 500))
            ) - (z_i + 500) ** 2 / (10000 * D)

    def obj_func(self, x):
        D = len(x)
        z = x + 420.9687462275036  # Shifted

        # Calculate the sum of g(z_i)
        sum_g = 0
        for i in range(D):
            sum_g += self.g(z[i], D)

        return 418.9829 * D - sum_g


class ExpandedSchaffer(CustomProblem):
    """
    Expanded Schaffer's F6 Function (Function 8)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"ExpandedSchaffer_{dimensions}d")

    def g(self, x, y):
        numer = np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5
        denom = (1 + 0.001 * (x**2 + y**2)) ** 2
        return 0.5 + numer / denom

    def obj_func(self, x):
        D = len(x)
        result = 0

        # Calculate g(x_i, x_{i+1}) for i=1,...,D-1
        for i in range(D - 1):
            result += self.g(x[i], x[i + 1])

        # Add g(x_D, x_1) to complete the cycle
        result += self.g(x[D - 1], x[0])

        return result


class HappyCat(CustomProblem):
    """
    Happy Cat Function (Function 9)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"HappyCat_{dimensions}d")

    def obj_func(self, x):
        D = len(x)
        sum_squared = np.sum(x**2)
        sum_x = np.sum(x)

        term1 = abs(sum_squared - D) ** (1 / 4)
        term2 = (0.5 * sum_squared + sum_x) / D + 0.5

        return term1 + term2


class Ackley(CustomProblem):
    """
    Ackley Function (Function 10)
    """

    def __init__(
        self,
        lb: float = -100,
        ub: float = 100,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Ackley_{dimensions}d")

    def obj_func(self, x):
        D = len(x)

        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))

        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / D))
        term2 = -np.exp(sum2 / D)

        return term1 + term2 + 20 + np.e


class Rosenbrock(CustomProblem):
    """
    Rosenbrock Function (Banana Function)

    This is the multidimensional Rosenbrock function:
    f(x) = sum_{i=1}^{N-1} [100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Global minimum at (1,1,...,1) with a value of 0
    """

    def __init__(
        self,
        lb: float = -5,
        ub: float = 10,
        dimensions: int = 10,
        save_population=True,
        **kwargs,
    ):
        super().__init__(
            ub=ub,
            lb=lb,
            dimensions=dimensions,
            save_population=save_population,
            **kwargs,
        )
        self.name = kwargs.get("name", f"Rosenbrock_{dimensions}d")

    def obj_func(self, x):
        n = len(x)
        result = 0

        for i in range(n - 1):
            result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

        return result


if __name__ == "__main__":
    problem = Rastrigin(-10, 10, 10)
    model = GWO.OriginalGWO(epoch=100, pop_size=100)
    g_best = model.solve(problem, seed=10)

    if not model.problem or not model.history:
        raise ValueError("Problem or history not set in the model.")

    print("Solution:", g_best.solution)
    print("Target fitness:", g_best.target.fitness)
    print("Target objectives:", g_best.target.objectives)
    print("Best solution (g_best):", g_best)
    print("Model parameters:", model.get_parameters())
    print("Model name:", model.get_name())
    print("Global best from attributes:", model.get_attributes()["g_best"])
    print("Global best from attributes:", model.get_attributes()["g_best"])
    print("Problem name:", model.problem.get_name())
    print("Problem dimensions:", model.problem.n_dims)
    print("Problem bounds:", model.problem.bounds)
    print("Problem lower bounds:", model.problem.lb)
    print("Problem upper bounds:", model.problem.ub)
    print("model.history.epoch:", model.history.epoch)

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
