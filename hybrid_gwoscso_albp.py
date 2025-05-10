import numpy as np
import math

# === Pareto Utilities ===
def dominates(f1, f2):
    return all(x <= y for x, y in zip(f1, f2)) and any(x < y for x, y in zip(f1, f2))
    #return all(x <= y for x, y in zip(f1, f2)) and any(x < y for x in f1)

def update_archive(pop, archive):
    for agent in pop:
        dominated = False
        new_archive = []
        for arch in archive:
            if dominates(agent["fitness"], arch["fitness"]):
                continue
            elif dominates(arch["fitness"], agent["fitness"]):
                dominated = True
                break
            else:
                new_archive.append(arch)
        if not dominated:
            new_archive.append(agent)
        archive[:] = new_archive
    return archive

def gwo_step(pop, archive, a):
    if len(archive) < 3:
        # Use best agents from population instead
        sorted_pop = sorted(pop, key=lambda x: x["fitness"])
    else:
        sorted_pop = sorted(archive, key=lambda x: x["fitness"])
    
    # Ensure at least 3 agents exist
    if len(sorted_pop) < 3:
        sorted_pop += [sorted_pop[-1]] * (3 - len(sorted_pop))  # duplicate last if needed

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
def scso_step(pop, best, epoch, max_epoch):
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
def hybrid_gwo_scso(problem, pop_size=50, max_epoch=200, stagnation_limit=10):
    pop = [{"position": np.random.uniform(0, 1, problem.nvars), "fitness": None} for _ in range(pop_size)]
    archive = []

    for agent in pop:
        #solution = type("Sol", (), {"variables": agent["position"]})()
        solution = type("Sol", (), {
            "variables": agent["position"],
            "objectives": [None, None]
        })()
        problem.evaluate(solution)
        agent["fitness"] = solution.objectives
    archive = update_archive(pop, [])

    stagnation = 0
    phase = "gwo"
    best_fitnesses = []

    for epoch in range(1, max_epoch + 1):
        if phase == "gwo":
            a = 2 - 2 * (epoch / max_epoch)
            new_pop = gwo_step(pop, archive, a)
        else:
            best = sorted(archive, key=lambda x: x["fitness"])[0]
            new_pop = scso_step(pop, best, epoch, max_epoch)

        for agent in new_pop:
            #solution = type("Sol", (), {"variables": agent["position"]})()
            solution = type("Sol", (), {
                "variables": agent["position"],
                "objectives": [None, None]
            })()
            problem.evaluate(solution)
            agent["fitness"] = solution.objectives

        archive = update_archive(new_pop, archive)
        pop = new_pop

        if len(best_fitnesses) > 0 and all(agent["fitness"] == best_fitnesses[-1] for agent in archive):
            stagnation += 1
        else:
            stagnation = 0

        best_fitnesses.append([agent["fitness"] for agent in archive])

        if stagnation >= stagnation_limit:
            phase = "scso"
        else:
            phase = "gwo"

    return archive