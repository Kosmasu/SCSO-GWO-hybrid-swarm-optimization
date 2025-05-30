import numpy as np
from mealpy.optimizer import Optimizer

from problems import Rastrigin


class HybridIGWOSCSO(Optimizer):
    """
    A hybrid optimizer combining Improved Grey Wolf Optimizer (IGWO) and Sand Cat Swarm Optimization (SCSO).

    The algorithm starts with IGWO to search for good solutions, but switches to SCSO when
    improvement stagnates to help escape local optima. It dynamically switches between phases
    based on stagnation detection.

    References:
    - IGWO: Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures.
        Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
    - SCSO: Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization:
            a nature-inspired algorithm to solve global optimization problems.
    """

    def __init__(
        self,
        epoch=10000,
        pop_size=100,
        stagnation_limit=10,
        a_min: float = 0.02,
        a_max: float = 2.2,
        **kwargs,
    ):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            stagnation_limit (int): number of iterations without improvement before switching to recovery phase
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.stagnation_limit = self.validator.check_int(
            "stagnation_limit", stagnation_limit, [1, 100]
        )
        self.sort_flag = False

        self.a_min = self.validator.check_float("a_min", a_min, (0.0, 1.6))
        self.a_max = self.validator.check_float("a_max", a_max, [1.0, 4.0])
        self.set_parameters(["epoch", "pop_size", "stagnation_limit", "a_min", "a_max"])

    def initialize_variables(self):
        """Initialize support variables for the algorithm"""
        # IGWO variables
        self.growth_alpha = 2
        self.growth_delta = 3

        # SCSO variables
        self.ss = 2  # maximum Sensitivity range
        self.pp = np.arange(1, 361)

        # Variables for phase tracking
        self.stagnation_count = 0
        self.best_fitness_history = []
        self.current_phase = "gwo"  # Start with GWO phase

    def get_index_roulette_wheel_selection__(self, p):
        """Helper method for SCSO phase"""
        p = p / np.sum(p)
        c = np.cumsum(p)
        return np.argwhere(self.generator.random() < c)[0][0]

    def evolve(self, epoch):
        """
        The main operations of the algorithm, combining GWO and SCSO with dynamic phase switching

        Args:
            epoch (int): The current iteration
        """
        # Track the best fitness and check for stagnation
        current_best = self.g_best.target.fitness
        if len(self.best_fitness_history) > 0:
            last_best = self.best_fitness_history[-1]
            # If we're minimizing, improvement means lower fitness value
            # If we're maximizing, improvement means higher fitness value
            if self.problem.minmax == "min":
                improved = current_best < last_best
            else:
                improved = current_best > last_best

            if improved:
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Print when change from scso to gwo
            if (
                self.current_phase == "scso"
                and self.stagnation_count < self.stagnation_limit
            ):
                print(f"Switching from SCSO to GWO at epoch {epoch}")
                print("Because of improvement in fitness")
            elif (
                self.current_phase == "gwo"
                and self.stagnation_count >= self.stagnation_limit
            ):
                print(
                    f"Stagnation count: {self.stagnation_count}, current phase: {self.current_phase}"
                )
                print("Because of no improvement in fitness")

            # Dynamic phase switching based on stagnation
            if self.stagnation_count >= self.stagnation_limit:
                self.current_phase = "scso"  # Switch to recovery phase
            else:
                self.current_phase = "gwo"  # Stay in exploration phase

        # Store current best fitness for next iteration comparison
        self.best_fitness_history.append(current_best)

        # Execute the current phase's evolution strategy
        if self.current_phase == "gwo":
            self._evolve_igwo(epoch)
        else:  # SCSO phase
            self._evolve_scso(epoch)

    def _evolve_igwo(self, epoch):
        """
        Improved Grey Wolf Optimizer phase

        References
        ~~~~~~~~~~
        [1] Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures.
        Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
        """
        _, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        pop_new = []

        for idx in range(0, self.pop_size):
            # IGWO functions
            a_alpha = self.a_max * np.exp(
                (epoch / self.epoch) ** self.growth_alpha * np.log(self.a_min / self.a_max)
            )
            a_delta = self.a_max * np.exp(
                (epoch / self.epoch) ** self.growth_delta * np.log(self.a_min / self.a_max)
            )
            a_beta = (a_alpha + a_delta) * 0.5

            # Calculate hunting factors
            A1 = a_alpha * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a_beta * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a_delta * (2 * self.generator.random(self.problem.n_dims) - 1)

            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)

            # Update positions based on alpha, beta, and delta wolves
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - self.pop[idx].solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.pop[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.pop[idx].solution
            )

            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

    def _evolve_scso(self, epoch):
        """Sand Cat Swarm Optimization phase (recovery mode)"""
        guides_r = self.ss - (self.ss * epoch / self.epoch)
        pop_new = []

        for idx in range(0, self.pop_size):
            r = self.generator.random() * guides_r
            R = (
                2 * guides_r
            ) * self.generator.random() - guides_r  # controls transition phases
            pos_new = self.pop[idx].solution.copy()

            for jdx in range(0, self.problem.n_dims):
                teta = self.get_index_roulette_wheel_selection__(self.pp)
                if -1 <= R <= 1:
                    rand_pos = np.abs(
                        self.generator.random() * self.g_best.solution[jdx]
                        - self.pop[idx].solution[jdx]
                    )
                    pos_new[jdx] = self.g_best.solution[jdx] - r * rand_pos * np.cos(
                        teta
                    )
                else:
                    cp = int(self.generator.random() * self.pop_size)
                    pos_new[jdx] = r * (
                        self.pop[cp].solution[jdx]
                        - self.generator.random() * self.pop[idx].solution[jdx]
                    )

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)

        self.pop = self.update_target_for_population(pop_new)


# Example usage
if __name__ == "__main__":
    epoch = 99999
    pop_size = 50
    stagnation_limit = 15  # Switch to SCSO after 15 iterations without improvement

    problem = Rastrigin(
        lb=-100,
        ub=100,
        dimensions=100,
        minmax="min",
        save_population=True,
        name="Rastrigin_10d",
    )

    model = HybridIGWOSCSO(
        epoch=epoch, pop_size=pop_size, stagnation_limit=stagnation_limit
    )
    best_solution = model.solve(problem)

    print(f"Solution: {best_solution.solution}")
    print(f"Fitness: {best_solution.target.fitness}")
