from typing import List, Dict, Tuple, Any, Union, Optional, TypeVar
import numpy as np
import math

# Type definitions
Fitness = List[float]
Position = np.ndarray
Individual = Dict[str, Union[Position, Fitness]]
Archive = List[Individual]
Problem = Any  # This should be replaced with actual problem type if available

# === Pareto Utilities ===
def dominates(f1: Fitness, f2: Fitness) -> bool:
    return all(x <= y for x, y in zip(f1, f2)) and any(x < y for x, y in zip(f1, f2))

def within_eps(f1: Fitness, f2: Fitness, epsilons: Tuple[float, ...]) -> bool:
    return all(abs(x - y) <= e for x, y, e in zip(f1, f2, epsilons))

def pareto_filter(candidates: Archive) -> Archive:
    pareto = []
    for c in candidates:
        dominated = False
        to_remove = []
        for i, p in enumerate(pareto):
            if dominates(p["fitness"], c["fitness"]):
                dominated = True
                break
            elif dominates(c["fitness"], p["fitness"]):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                del pareto[i]
            pareto.append(c)
    return pareto

def epsilon_filter(pareto: Archive, epsilons: Tuple[float, ...] = (1.0, 5.0)) -> Archive:
    final = []
    for c in pareto:
        if not any(within_eps(c["fitness"], p["fitness"], epsilons) for p in final):
            final.append(c)
    return final

def update_archive(pop: Archive, archive: Archive, epsilons: Tuple[float, ...] = (1.0, 5.0)) -> Archive:
    combined = archive + pop
    pareto = pareto_filter(combined)
    filtered = epsilon_filter(pareto, epsilons)
    return filtered

# === GWO Step ===
def gwo_step(pop: Archive, archive: Archive, a: float) -> Archive:
    if len(archive) < 3:
        sorted_pop = sorted(pop, key=lambda x: x["fitness"])
    else:
        sorted_pop = sorted(archive, key=lambda x: x["fitness"])
    if len(sorted_pop) < 3:
        sorted_pop += [sorted_pop[-1]] * (3 - len(sorted_pop))

    alpha, beta, delta = sorted_pop[:3]

    new_pop = []
    for wolf in pop:
        A = [a * (2 * np.random.rand(len(wolf["position"])) - 1) for _ in range(3)]
        C = [2 * np.random.rand(len(wolf["position"])) for _ in range(3)]
        X = [
            alpha["position"] - A[0] * np.abs(C[0] * alpha["position"] - wolf["position"]),
            beta["position"]  - A[1] * np.abs(C[1] * beta["position"]  - wolf["position"]),
            delta["position"] - A[2] * np.abs(C[2] * delta["position"] - wolf["position"])
        ]
        new_pos = np.clip((X[0] + X[1] + X[2]) / 3.0, 0, 1)
        new_pop.append({"position": new_pos})
    return new_pop

# === SCSO Step ===
def scso_step(pop: Archive, best: Individual, epoch: int, max_epoch: int) -> Archive:
    ss = 2
    guides_r = ss - (ss * epoch / max_epoch)
    new_pop = []

    for agent in pop:
        r = np.random.rand() * guides_r
        R = (2 * guides_r) * np.random.rand() - guides_r
        new_pos = agent["position"].copy()

        for j in range(len(new_pos)):
            teta = np.random.randint(1, 361)
            if -1 <= R <= 1:
                rand_diff = abs(np.random.rand() * best["position"][j] - agent["position"][j])
                new_pos[j] = best["position"][j] - r * rand_diff * math.cos(math.radians(teta))
            else:
                cp = np.random.randint(len(pop))
                new_pos[j] = r * (pop[cp]["position"][j] - np.random.rand() * agent["position"][j])

        new_pos = np.clip(new_pos, 0, 1)
        new_pop.append({"position": new_pos})
    return new_pop

# === Hybrid Optimizer ===
def hybrid_gwo_scso(problem: Problem, pop_size: int = 60, max_epoch: int = 500, 
                    stagnation_limit: int = 15, epsilons: Tuple[float, ...] = (0.5, 0.5)) -> Tuple[Archive, List[Individual]]:
    pop = [{"position": np.random.uniform(0, 1, problem.nvars), "fitness": None} for _ in range(pop_size)]
    archive: Archive = []
    all_solutions: List[Individual] = []  # Store all evaluated solutions

    # Initial fitness eval
    for agent in pop:
        solution = type("Sol", (), {
            "variables": agent["position"],
            "objectives": [None, None]
        })()
        problem.evaluate(solution)
        agent["fitness"] = solution.objectives
        all_solutions.append(agent.copy())  # Store a copy of the evaluated solution

    archive = update_archive(pop, [], epsilons)

    stagnation: int = 0
    phase: str = "gwo"
    best_archive_size: int = len(archive)

    for epoch in range(1, max_epoch + 1):
        if phase == "gwo":
            a: float = 2 - 2 * (epoch / max_epoch)
            new_pop = gwo_step(pop, archive, a)
        else:
            if archive:
                best = sorted(archive, key=lambda x: x["fitness"])[0]
            else:
                best = min(pop, key=lambda x: x["fitness"])
            new_pop = scso_step(pop, best, epoch, max_epoch)

        for agent in new_pop:
            solution = type("Sol", (), {
                "variables": agent["position"],
                "objectives": [None, None]
            })()
            problem.evaluate(solution)
            agent["fitness"] = solution.objectives
            all_solutions.append(agent.copy())  # Store a copy of the evaluated solution

        archive = update_archive(new_pop, archive, epsilons)
        pop = new_pop

        # Stagnation logic based on archive size change
        if len(archive) == best_archive_size:
            stagnation += 1
        else:
            stagnation = 0
            best_archive_size = len(archive)

        if stagnation >= stagnation_limit:
            phase = "scso"
        else:
            phase = "gwo"

        print(f"Epoch {epoch}: Archive size = {len(archive)} | Phase: {phase}")

    return archive, all_solutions