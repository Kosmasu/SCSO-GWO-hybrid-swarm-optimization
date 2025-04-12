import numpy as np
from mealpy.optimizer import Optimizer

class HybridGWOSCSO(Optimizer):
    """
    A hybrid optimizer combining Grey Wolf Optimizer (GWO) and Sand Cat Swarm Optimization (SCSO).
    
    The algorithm starts with GWO to search for good solutions, but switches to SCSO when
    improvement stagnates to help escape local optima. It dynamically switches between phases
    based on stagnation detection.
    
    References:
    - GWO: Mirjalili, S., Mirjalili, S.M. and Lewis, A., 2014. Grey wolf optimizer. 
           Advances in engineering software, 69, pp.46-61.
    - SCSO: Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization: 
            a nature-inspired algorithm to solve global optimization problems.
    """

    def __init__(self, epoch=10000, pop_size=100, stagnation_limit=10, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            stagnation_limit (int): number of iterations without improvement before switching to recovery phase
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.stagnation_limit = self.validator.check_int("stagnation_limit", stagnation_limit, [1, 100])
        self.set_parameters(["epoch", "pop_size", "stagnation_limit"])
        self.sort_flag = False
    
    def initialize_variables(self):
        """Initialize support variables for the algorithm"""
        # GWO doesn't need special initialization
        
        # SCSO variables
        self.ss = 2      # maximum Sensitivity range
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
                
            # Dynamic phase switching based on stagnation
            if self.stagnation_count >= self.stagnation_limit:
                self.current_phase = "scso"  # Switch to recovery phase
            else:
                self.current_phase = "gwo"   # Stay in exploration phase
        
        # Store current best fitness for next iteration comparison
        self.best_fitness_history.append(current_best)
        
        # Execute the current phase's evolution strategy
        if self.current_phase == "gwo":
            self._evolve_gwo(epoch)
        else:  # SCSO phase
            self._evolve_scso(epoch)
    
    def _evolve_gwo(self, epoch):
        """Grey Wolf Optimizer phase"""
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            
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
            R = (2*guides_r)*self.generator.random() - guides_r  # controls transition phases
            pos_new = self.pop[idx].solution.copy()
            
            for jdx in range(0, self.problem.n_dims):
                teta = self.get_index_roulette_wheel_selection__(self.pp)
                if -1 <= R <= 1:
                    rand_pos = np.abs(self.generator.random() * self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = self.g_best.solution[jdx] - r * rand_pos * np.cos(teta)
                else:
                    cp = int(self.generator.random() * self.pop_size)
                    pos_new[jdx] = r * (self.pop[cp].solution[jdx] - self.generator.random() * self.pop[idx].solution[jdx])
            
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                
        self.pop = self.update_target_for_population(pop_new)


# Example usage
if __name__ == "__main__":
    from mealpy import FloatVar
    
    def objective_function(solution):
        return np.sum(solution**2)
    
    problem_dict = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-100, ]*30, ub=[100, ]*30),
        "minmax": "min",
    }
    
    epoch = 1000
    pop_size = 50
    stagnation_limit = 15  # Switch to SCSO after 15 iterations without improvement
    
    model = HybridGWOSCSO(epoch=epoch, pop_size=pop_size, stagnation_limit=stagnation_limit)
    best_solution = model.solve(problem_dict)
    
    print(f"Solution: {best_solution.solution}")
    print(f"Fitness: {best_solution.target.fitness}")