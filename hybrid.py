"""
Our algorithm goes here
https://mealpy.readthedocs.io/en/latest/pages/general/build_new_optimizer.html
"""

import numpy as np
from mealpy import Optimizer, FloatVar


class GWO_SCSO_Hybrid(Optimizer):
    """
    This is an example how to build new optimizer
    """

    def __init__(self, epoch=10000, pop_size=100, m_clusters=2, p1=0.75, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.m_clusters = self.validator.check_int("m_clusters", m_clusters, [2, 5])
        self.p1 = self.validator.check_float("p1", p1, (0, 1.0))

        self.sort_flag = True
        # Determine to sort the problem or not in each epoch
        ## if True, the problem always sorted with fitness value increase
        ## if False, the problem is not sorted

    def initialize_variables(self):
        """
        This is method is called before initialization() method.
        """
        ## Support variables
        self.n_agents = int(self.pop_size / self.m_clusters)
        self.space = self.problem.ub - self.problem.lb

    def initialization(self):
        """
        Override this method if needed. But the first 2 lines of code is required.
        """
        ### Required code
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        ### Your additional code can be implemented here
        self.mean_pos = np.mean([agent[self.ID_POS] for agent in self.pop])

    def evolve(self, epoch):
        """
        You can do everything in this function (i.e., Loop through the population multiple times)

        Args:
            epoch (int): The current iteration
        """
        epsilon = 1.0 - epoch / self.epoch      # The epsilon in each epoch is changing based on this equation

        ## 1. Replace the almost worst agent by random agent
        if self.generator.uniform() < self.p1:
            idx = self.generator.integers(self.n_agents, self.pop_size)
            self.pop[idx] = self.generate_agent()

        ## 2. Replace all bad solutions by current_best + noise
        for idx in range(self.n_agents, self.pop_size):
            pos_new = self.pop[0].solution + epsilon * self.space * self.generator.normal(0, 1)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

        ## 3. Move all good solutions toward current best solution
        for idx in range(0, self.n_agents):
            if idx == 0:
                pos_new = self.pop[idx].solution + epsilon * self.space * self.generator.uniform(0, 1)
            else:
                pos_new = self.pop[idx].solution + epsilon * self.space * (self.pop[0].solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent

        ## Do additional works here if needed.

if __name__ == "__main__":
    # Example usage
    from obj_functions import Squared

    problem = Squared(
        lb=-100,
        ub=100,
        dimensions=10,
        minmax="min",
        save_population=True,
        name="Squared_10d",
    )

    EPOCH = 1000
    POPULATION = 30

    model = GWO_SCSO_Hybrid(epoch=EPOCH, pop_size=POPULATION)
    best_solution = model.solve(problem=problem)
    print(f"Best solution: {best_solution}")