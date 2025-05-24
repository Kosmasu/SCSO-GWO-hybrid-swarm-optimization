import os
import shutil


def dominates(f1, f2):
    return all(x <= y for x, y in zip(f1, f2)) and any(x < y for x, y in zip(f1, f2))


def within_eps(f1, f2, epsilons):
    return all(abs(x - y) <= e for x, y, e in zip(f1, f2, epsilons))


def pareto_filter(candidates):
    pareto = []
    for c in candidates:
        dominated = False
        to_remove = []
        for i, p in enumerate(pareto):
            if dominates(p.objectives, c.objectives):
                dominated = True
                break
            elif dominates(c.objectives, p.objectives):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                del pareto[i]
            pareto.append(c)
    return pareto


def epsilon_filter(pareto, epsilons=(1.0, 5.0)):
    final = []
    for c in pareto:
        if not any(within_eps(c.objectives, p.objectives, epsilons) for p in final):
            final.append(c)
    return final


def filter_results(results, epsilons=(1.0, 5.0)):
    return epsilon_filter(pareto_filter(results), epsilons)


# create and clear output directory
def create_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
