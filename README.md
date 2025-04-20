# SCSO-GWO Hybrid Swarm Optimization

This repository provides implementations of hybrid swarm optimization algorithms, combining Grey Wolf Optimizer (GWO), Improved GWO (IGWO), and Sand Cat Swarm Optimization (SCSO). The codebase is built on top of the [mealpy](https://github.com/thieu1995/mealpy) library and includes custom benchmark problems for testing and experimentation.

## Features

- **Hybrid Algorithms**: Switch dynamically between GWO/IGWO and SCSO to escape local optima.
- **Custom Benchmark Problems**: Includes Rastrigin, Griewank, Ackley, Rosenbrock, and more.
- **Experiment Automation**: Easily run large-scale experiments and log results.
- **Jupyter Notebook Example**: Quick demonstration of the adaptive hybrid optimizer.

## Installation

### Using [uv](https://github.com/astral-sh/uv)
```bash
uv venv
source .venv/bin/activate
uv sync
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run a Quick Example

To run a simple optimization using the hybrid GWO-SCSO algorithm on the Rastrigin function:

```bash
python hybrid_1.py
```

You can also try the improved hybrid or the SCSO-starting hybrid:

```bash
python hybrid_2.py
python hybrid_3.py
```

### 2. Run Full Experiments

To run all algorithms on all benchmark problems and log results:

```bash
python main.py
```

Results and checkpoints will be saved in the `output/` directory.


## Project Structure

- `main.py` — Runs experiments across all models and problems.
- `hybrid_1.py`, `hybrid_2.py`, `hybrid_3.py` — Hybrid optimizer implementations.
- `problems.py` — Custom benchmark problem definitions.
- `output/` — Results, checkpoints, and visualizations.

## Customization

- To add new benchmark problems, edit `problems.py`.
- To adjust optimizer parameters (population, epochs, etc.), modify the arguments in the scripts.

## Requirements

- Python 3.13+
- See `requirements.txt` or `pyproject.toml` for dependencies.


## Pseudocodes
### HybridGWOSCSO
Algorithm: Hybrid Grey Wolf Optimizer and Sand Cat Swarm Optimization (HybridGWOSCSO)
```
Input: Objective function f(x), population size (pop_size), max iterations (epoch), stagnation limit (stagnation_limit), problem bounds (lb, ub), dimensions (n_dims)

Output: Best solution found (g_best)

BEGIN
  // Initialization
  Initialize population (pop) with pop_size random solutions within bounds [lb, ub]
  Evaluate fitness for each agent in pop
  Find the global best solution (g_best)
  Initialize SCSO sensitivity range (ss = 2)
  Initialize SCSO roulette wheel probabilities (pp = array from 1 to 360)
  Initialize stagnation_count = 0
  Initialize best_fitness_history = []
  Initialize current_phase = "gwo"

  // Main Loop
  FOR t = 1 TO epoch DO
    current_best_fitness = g_best.fitness

    // Stagnation Detection and Phase Switching
    IF best_fitness_history is not empty THEN
      last_best_fitness = last element of best_fitness_history
      IF (problem is minimization AND current_best_fitness < last_best_fitness) OR (problem is maximization AND current_best_fitness > last_best_fitness) THEN
        stagnation_count = 0  // Improvement detected
      ELSE
        stagnation_count = stagnation_count + 1 // No improvement
      END IF

      // Switch phase based on stagnation
      IF stagnation_count >= stagnation_limit THEN
        current_phase = "scso"
      ELSE
        current_phase = "gwo"
      END IF
    END IF

    Append current_best_fitness to best_fitness_history

    // Execute Evolution based on Current Phase
    IF current_phase == "gwo" THEN
      CALL _evolve_gwo(t, pop, g_best)
    ELSE // current_phase == "scso"
      CALL _evolve_scso(t, pop, g_best)
    END IF

    // Update Global Best
    Update g_best based on the new population (pop)

  END FOR

  RETURN g_best

END

// GWO Evolution Phase
FUNCTION _evolve_gwo(t, pop, g_best)
  Calculate a = 2 - 2 * t / epoch
  Identify alpha, beta, delta wolves (best 3 solutions in pop)
  Initialize pop_new = empty list

  FOR each agent i in pop DO
    Calculate A1, A2, A3 using 'a' and random numbers
    Calculate C1, C2, C3 using random numbers
    Calculate X1 = alpha.position - A1 * |C1 * alpha.position - agent_i.position|
    Calculate X2 = beta.position - A2 * |C2 * beta.position - agent_i.position|
    Calculate X3 = delta.position - A3 * |C3 * delta.position - agent_i.position|
    pos_new = (X1 + X2 + X3) / 3
    Correct pos_new to stay within bounds [lb, ub]
    Create new_agent with pos_new
    Evaluate fitness of new_agent
    Append new_agent to pop_new
  END FOR

  Update pop based on pop_new (e.g., greedy selection)
END FUNCTION

// SCSO Evolution Phase
FUNCTION _evolve_scso(t, pop, g_best)
  Calculate guides_r = ss - (ss * t / epoch)
  Initialize pop_new = empty list

  FOR each agent i in pop DO
    Calculate r = random() * guides_r
    Calculate R = (2 * guides_r) * random() - guides_r
    pos_new = copy of agent_i.position

    FOR each dimension j DO
      Select teta using roulette wheel selection on pp
      IF -1 <= R <= 1 THEN // Exploitation
        rand_pos = |random() * g_best.position[j] - agent_i.position[j]|
        pos_new[j] = g_best.position[j] - r * rand_pos * cos(teta)
      ELSE // Exploration
        Select random agent cp from pop
        pos_new[j] = r * (pop[cp].position[j] - random() * agent_i.position[j])
      END IF
    END FOR

    Correct pos_new to stay within bounds [lb, ub]
    Create new_agent with pos_new
    Evaluate fitness of new_agent
    Append new_agent to pop_new
  END FOR

  Replace pop with pop_new
END FUNCTION
```

### HybridIGWOSCSO
Algorithm: Hybrid Improved Grey Wolf Optimizer and Sand Cat Swarm Optimization (HybridIGWOSCSO)
```
Input: Objective function f(x), population size (pop_size), max iterations (epoch), stagnation limit (stagnation_limit), IGWO parameters (a_min, a_max, growth_alpha, growth_delta), problem bounds (lb, ub), dimensions (n_dims)

Output: Best solution found (g_best)

BEGIN
  // Initialization
  Initialize population (pop) with pop_size random solutions within bounds [lb, ub]
  Evaluate fitness for each agent in pop
  Find the global best solution (g_best)
  Initialize IGWO parameters: a_min, a_max, growth_alpha, growth_delta
  Initialize SCSO sensitivity range (ss = 2)
  Initialize SCSO roulette wheel probabilities (pp = array from 1 to 360)
  Initialize stagnation_count = 0
  Initialize best_fitness_history = []
  Initialize current_phase = "gwo" // Start with IGWO phase

  // Main Loop
  FOR t = 1 TO epoch DO
    current_best_fitness = g_best.fitness

    // Stagnation Detection and Phase Switching
    IF best_fitness_history is not empty THEN
      last_best_fitness = last element of best_fitness_history
      IF (problem is minimization AND current_best_fitness < last_best_fitness) OR (problem is maximization AND current_best_fitness > last_best_fitness) THEN
        stagnation_count = 0  // Improvement detected
      ELSE
        stagnation_count = stagnation_count + 1 // No improvement
      END IF

      // Switch phase based on stagnation
      IF stagnation_count >= stagnation_limit THEN
        current_phase = "scso"
      ELSE
        current_phase = "gwo"
      END IF
    END IF

    Append current_best_fitness to best_fitness_history

    // Execute Evolution based on Current Phase
    IF current_phase == "gwo" THEN
      CALL _evolve_igwo(t, pop, g_best, a_min, a_max, growth_alpha, growth_delta)
    ELSE // current_phase == "scso"
      CALL _evolve_scso(t, pop, g_best)
    END IF

    // Update Global Best
    Update g_best based on the new population (pop)

  END FOR

  RETURN g_best

END

// Improved GWO (IGWO) Evolution Phase
FUNCTION _evolve_igwo(t, pop, g_best, a_min, a_max, growth_alpha, growth_delta)
  Identify alpha, beta, delta wolves (best 3 solutions in pop)
  Initialize pop_new = empty list

  // Calculate IGWO 'a' parameters based on epoch 't'
  a_alpha = a_max * exp( (t / epoch)^growth_alpha * log(a_min / a_max) )
  a_delta = a_max * exp( (t / epoch)^growth_delta * log(a_min / a_max) )
  a_beta = (a_alpha + a_delta) * 0.5

  FOR each agent i in pop DO
    // Calculate hunting factors using IGWO 'a' values
    A1 = a_alpha * (2 * random_vector(n_dims) - 1)
    A2 = a_beta  * (2 * random_vector(n_dims) - 1)
    A3 = a_delta * (2 * random_vector(n_dims) - 1)

    C1 = 2 * random_vector(n_dims)
    C2 = 2 * random_vector(n_dims)
    C3 = 2 * random_vector(n_dims)

    // Update position based on alpha, beta, delta
    X1 = alpha.position - A1 * |C1 * alpha.position - agent_i.position|
    X2 = beta.position  - A2 * |C2 * beta.position  - agent_i.position|
    X3 = delta.position - A3 * |C3 * delta.position - agent_i.position|
    pos_new = (X1 + X2 + X3) / 3

    Correct pos_new to stay within bounds [lb, ub]
    Create new_agent with pos_new
    Evaluate fitness of new_agent
    Append new_agent to pop_new
  END FOR

  Update pop based on pop_new (e.g., greedy selection)
END FUNCTION

// SCSO Evolution Phase
FUNCTION _evolve_scso(t, pop, g_best)
  Calculate guides_r = ss - (ss * t / epoch)
  Initialize pop_new = empty list

  FOR each agent i in pop DO
    Calculate r = random() * guides_r
    Calculate R = (2 * guides_r) * random() - guides_r
    pos_new = copy of agent_i.position

    FOR each dimension j DO
      Select teta using roulette wheel selection on pp
      IF -1 <= R <= 1 THEN // Exploitation (Attack prey)
        rand_pos = |random() * g_best.position[j] - agent_i.position[j]|
        pos_new[j] = g_best.position[j] - r * rand_pos * cos(teta)
      ELSE // Exploration (Search for prey)
        Select random agent cp from pop
        pos_new[j] = r * (pop[cp].position[j] - random() * agent_i.position[j])
      END IF
    END FOR

    Correct pos_new to stay within bounds [lb, ub]
    Create new_agent with pos_new
    Evaluate fitness of new_agent
    Append new_agent to pop_new
  END FOR

  Replace pop with pop_new
END FUNCTION
```

### HybridGWOSCSO3
Algorithm: Hybrid Grey Wolf Optimizer and Sand Cat Swarm Optimization (Version 3 - HybridGWOSCSO3)
```
Input: Objective function f(x), population size (pop_size), max iterations (epoch), stagnation limit (stagnation_limit), initial SCSO ratio (initial_scso), SCSO recovery limit (recovery_limit), problem bounds (lb, ub), dimensions (n_dims)

Output: Best solution found (g_best)

BEGIN
  // Initialization
  Initialize population (pop) with pop_size random solutions within bounds [lb, ub]
  Evaluate fitness for each agent in pop
  Find the global best solution (g_best)
  Initialize SCSO sensitivity range (ss = 2)
  Initialize SCSO roulette wheel probabilities (pp = array from 1 to 360)
  Initialize stagnation_count = 0
  Initialize best_fitness_history = []
  Initialize current_phase = "scso" // Start with SCSO
  Initialize startwithscso = true
  Calculate scso_run_limit = epoch * initial_scso
  Initialize recovery_iter = 0

  // Main Loop
  FOR t = 1 TO epoch DO
    current_best_fitness = g_best.fitness

    // Stagnation Detection (after first iteration)
    IF best_fitness_history is not empty THEN
      last_best_fitness = last element of best_fitness_history
      IF (problem is minimization AND current_best_fitness < last_best_fitness) OR (problem is maximization AND current_best_fitness > last_best_fitness) THEN
        stagnation_count = 0  // Improvement detected
      ELSE
        stagnation_count = stagnation_count + 1 // No improvement
      END IF
    END IF

    // Phase Switching Logic
    IF startwithscso is true THEN
      IF t > scso_run_limit THEN
        // End of initial SCSO phase
        current_phase = "gwo"
        startwithscso = false
        stagnation_count = 0 // Reset stagnation after switching
    ELSE // After initial SCSO phase
      IF current_phase == "gwo" AND stagnation_count >= stagnation_limit THEN
        // Switch from GWO to SCSO due to stagnation
        current_phase = "scso"
        recovery_iter = 0 // Start recovery phase
      ELSE IF current_phase == "scso" THEN
        IF recovery_iter < recovery_limit THEN
          // Continue SCSO recovery phase
          recovery_iter = recovery_iter + 1
        ELSE
          // End of SCSO recovery phase iteration limit
          IF stagnation_count >= stagnation_limit THEN
            // Still stagnated, restart SCSO recovery
            recovery_iter = 0
          ELSE
            // Improvement occurred during SCSO recovery, switch back to GWO
            current_phase = "gwo"
          END IF
        END IF
      END IF
    END IF

    Append current_best_fitness to best_fitness_history

    // Execute Evolution based on Current Phase
    IF current_phase == "gwo" THEN
      CALL _evolve_gwo(t, pop, g_best)
    ELSE // current_phase == "scso"
      CALL _evolve_scso(t, pop, g_best)
    END IF

    // Update Global Best
    Update g_best based on the new population (pop)

  END FOR

  RETURN g_best

END

// GWO Evolution Phase
FUNCTION _evolve_gwo(t, pop, g_best)
  Calculate a = 2 - 2 * t / epoch
  Identify alpha, beta, delta wolves (best 3 solutions in pop)
  Initialize pop_new = empty list

  FOR each agent i in pop DO
    Calculate A1, A2, A3 using 'a' and random numbers
    Calculate C1, C2, C3 using random numbers
    Calculate X1 = alpha.position - A1 * |C1 * alpha.position - agent_i.position|
    Calculate X2 = beta.position - A2 * |C2 * beta.position - agent_i.position|
    Calculate X3 = delta.position - A3 * |C3 * delta.position - agent_i.position|
    pos_new = (X1 + X2 + X3) / 3
    Correct pos_new to stay within bounds [lb, ub]
    Create new_agent with pos_new
    Evaluate fitness of new_agent
    Append new_agent to pop_new
  END FOR

  Update pop based on pop_new (e.g., greedy selection)
END FUNCTION

// SCSO Evolution Phase
FUNCTION _evolve_scso(t, pop, g_best)
  Calculate guides_r = ss - (ss * t / epoch)
  Initialize pop_new = empty list

  FOR each agent i in pop DO
    Calculate r = random() * guides_r
    Calculate R = (2 * guides_r) * random() - guides_r
    pos_new = copy of agent_i.position

    FOR each dimension j DO
      Select teta using roulette wheel selection on pp
      IF -1 <= R <= 1 THEN // Exploitation (Attack prey)
        rand_pos = |random() * g_best.position[j] - agent_i.position[j]|
        pos_new[j] = g_best.position[j] - r * rand_pos * cos(teta)
      ELSE // Exploration (Search for prey)
        Select random agent cp from pop
        pos_new[j] = r * (pop[cp].position[j] - random() * agent_i.position[j])
      END IF
    END FOR

    Correct pos_new to stay within bounds [lb, ub]
    Create new_agent with pos_new
    Evaluate fitness of new_agent
    Append new_agent to pop_new
  END FOR

  Replace pop with pop_new
END FUNCTION
```

## References

If you use this codebase in your research, please cite the relevant papers for GWO, IGWO, SCSO, and [mealpy](https://github.com/thieu1995/mealpy).
- OriginalGWO: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.
- IGWO: Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures. Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
- OriginalSCSO: Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization: a nature-inspired algorithm to solve global optimization problems. Engineering with Computers, 1-25.
