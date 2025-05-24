# SCSO-GWO Hybrid Swarm Optimization (Multi-Objective)

This repository demonstrates multi-objective optimization using a hybrid approach combining Grey Wolf Optimizer (GWO) and Sand Cat Swarm Optimization (SCSO). It leverages the power of [Platypus](https://platypus.readthedocs.io) for Pareto-based evolutionary optimization.

## Features

- **Hybrid Algorithm**: Merges GWO with SCSO to tackle multi-objective assembly line balancing.
- **Pareto Optimization**: Maintains a Pareto archive for non-dominated solutions.
- **Utility Scripts**: Plots Pareto fronts, saves solutions, and logs outcomes.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Edit parameter settings (population size, epochs, etc.) in the code.
2. Run experiments using:
    ```bash
    python main.py
    ```
3. View results and plots in the `output/` directory.

## Project Structure

- **main.py**: Entry point to run the main benchmark across various evolutionary algorithms.
- **main_hybrid.py**: Dedicated to executing large-scale benchmarks with different configurations of the hybrid optimizer.
- **hybrid.py**: Implements the two-phase (GWO + SCSO) hybrid swarm optimization approach with Pareto-based archiving.
- **problem_MO.py**: Defines a multi-objective assembly line balancing problem using Platypus’s customizable `Problem` class.
- **utils.py** (optional): Contains helper functions for directory creation and result filtering.
- **visualization.py** (optional): Generates 2D Pareto plots and workstation task distributions.
- **data.py** (optional): Stores task time data, precedence constraints, and hyperparameters for optimization.

## Customization

- To add new constraints, modify `problem_MO.py`.
- To change algorithm settings (epochs, stagnation limits), edit parameters in `main.py` or `main_hybrid.py`.
- Incorporate additional logging or plotting in the `visualization.py` scripts.

## Requirements

- Python 3.10+
- NumPy, Platypus, Matplotlib, and other libraries listed in `requirements.txt`.
