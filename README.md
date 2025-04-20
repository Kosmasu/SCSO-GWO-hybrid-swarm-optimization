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

### 3. Jupyter Notebook Demo

Open the notebook for a step-by-step example:

```bash
jupyter notebook "GWO-SCSO Hybrid.ipynb"
```

## Project Structure

- `main.py` — Runs experiments across all models and problems.
- `hybrid_1.py`, `hybrid_2.py`, `hybrid_3.py` — Hybrid optimizer implementations.
- `problems.py` — Custom benchmark problem definitions.
- `GWO-SCSO Hybrid.ipynb` — Notebook demo.
- `output/` — Results, checkpoints, and visualizations.

## Customization

- To add new benchmark problems, edit `problems.py`.
- To adjust optimizer parameters (population, epochs, etc.), modify the arguments in the scripts.

## Requirements

- Python 3.13+
- See `requirements.txt` or `pyproject.toml` for dependencies.

## References

If you use this codebase in your research, please cite the relevant papers for GWO, IGWO, SCSO, and [mealpy](https://github.com/thieu1995/mealpy).
- OriginalGWO: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.
- IGWO: Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures. Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
- OriginalSCSO: Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization: a nature-inspired algorithm to solve global optimization problems. Engineering with Computers, 1-25.
