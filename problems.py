"""
Custom benchmark objective functions for optimization algorithms.

This module contains implementations of various benchmark functions commonly used
to test optimization algorithms, such as Rastrigin, Styblinski-Tang, and others.
These functions can be used with the mealpy optimization library.
"""

import os
from typing import Union
from mealpy import Problem, FloatVar, GWO
import numpy as np
from mealpy.utils.space import BaseVar


"""
Yes, all of these benchmark functions are designed for minimization problems where the goal is to find the global minimum.
In the context of the 100-Digit Challenge and optimization testing in general, these functions serve as difficult test cases because they have challenging characteristics that make finding the global minimum difficult:

Most are multimodal (have multiple local minima)
Many are non-separable (variables are interdependent)
Several have numerous local optima that can trap optimization algorithms
Some have deceptive landscapes where the global minimum is surrounded by areas that might lead algorithms astray

According to the document you shared, all the functions in the 100-Digit Challenge have their global minimum at a function value of 1.000000000 (to 10 digits of accuracy). The challenge is to locate this minimum as precisely as possible.
For the Rosenbrock function specifically, its standard form has a global minimum of 0 at the point (1,1,...,1), but the version used in the challenge would likely be shifted to have a minimum value of 1.000000000 to match the other functions.
Different optimization algorithms (like the swarm algorithm you mentioned) can be benchmarked based on how accurately and efficiently they can find these global minima across various challenging landscapes.
"""


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


# class StyblinskiTang(CustomProblem):
#     def __init__(
#         self,
#         lb: float = -100,
#         ub: float = 100,
#         dimensions: int = 10,
#         save_population=True,
#         **kwargs,
#     ):
#         super().__init__(
#             ub=ub,
#             lb=lb,
#             dimensions=dimensions,
#             save_population=save_population,
#             **kwargs,
#         )
#         self.name = kwargs.get("name", f"Styblinski_Tang{dimensions}d")

#     def obj_func(self, x):
#         return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


# class StornChebyshev(CustomProblem):
#     """
#     Storn's Chebyshev Polynomial Fitting Problem (Function 1)
#     Note: Only defined for D = 9 and other specific dimensions
#     """

#     def __init__(
#         self,
#         lb: float = -8192,
#         ub: float = 8192,
#         dimensions: int = 9,
#         save_population=True,
#         **kwargs,
#     ):
#         # Check if dimensions are valid (D = 9 for the document)
#         if dimensions != 9:
#             raise ValueError(
#                 "StornChebyshev function is specifically defined for D = 9 in this challenge"
#             )

#         super().__init__(
#             ub=ub,
#             lb=lb,
#             dimensions=dimensions,
#             save_population=save_population,
#             **kwargs,
#         )
#         self.name = kwargs.get("name", f"StornChebyshev_{dimensions}d")
#         self.d = 72.661  # Specific for D = 9

#     def obj_func(self, x):
#         D = len(x)
#         m = 32 * D

#         # Calculate p1
#         u = sum(x[j] * (1.2) ** (D - j - 1) for j in range(D))
#         p1 = (u - self.d) ** 2 if u < self.d else 0

#         # Calculate p2
#         v = sum(x[j] * (-1.2) ** (D - j - 1) for j in range(D))
#         p2 = (v - self.d) ** 2 if v < self.d else 0

#         # Calculate p3
#         p3 = 0
#         for k in range(m + 1):
#             w_k = sum(x[j] * (2 * k / m - 1) ** (D - j - 1) for j in range(D))
#             if w_k > 1:
#                 p3 += (w_k - 1) ** 2
#             elif w_k < -1:
#                 p3 += (w_k + 1) ** 2

#         return p1 + p2 + p3


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


# class LennardJones(CustomProblem):
#     """
#     Lennard-Jones Minimum Energy Cluster Problem (Function 3)
#     Note: Only defined for D = 3n where n is the number of atoms
#     """

#     def __init__(
#         self,
#         lb: float = -4,
#         ub: float = 4,
#         dimensions: int = 18,
#         save_population=True,
#         **kwargs,
#     ):
#         # Check if dimensions are valid (D should be divisible by 3)
#         if dimensions % 3 != 0:
#             raise ValueError(
#                 f"LennardJones function requires D to be divisible by 3. Got D={dimensions}"
#             )

#         super().__init__(
#             ub=ub,
#             lb=lb,
#             dimensions=dimensions,
#             save_population=save_population,
#             **kwargs,
#         )
#         self.name = kwargs.get("name", f"LennardJones_{dimensions}d")

#     def obj_func(self, x):
#         D = len(x)
#         n = D // 3  # Number of atoms

#         # Add the constant 12.7120622568 as per the document
#         energy = 12.7120622568

#         # Calculate pairwise distances and energies
#         for i in range(n - 1):
#             for j in range(i + 1, n):
#                 # Calculate squared distance between atoms i and j
#                 d_squared = 0
#                 for k in range(3):
#                     d_squared += (x[3 * i + k] - x[3 * j + k]) ** 2

#                 # Calculate distance to the power of 6 (d^6)
#                 d6 = d_squared**3

#                 # Add potential energy term
#                 if d6 > 0:  # Prevent division by zero
#                     energy += 1 / d6 - 2 / np.sqrt(d6)

#         return energy


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
        self.a = 0.5
        self.b = 3
        self.kmax = 20

    def obj_func(self, x):
        D = len(x)
        a = self.a
        b = self.b
        kmax = self.kmax

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
