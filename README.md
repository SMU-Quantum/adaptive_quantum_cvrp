# Adaptive Quantum CVRP Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the research paper "[Hybrid Learning and Optimization methods for solving Capacitated Vehicle Routing Problem](https://arxiv.org/abs/2509.15262v1)." It provides a novel hybrid framework that uses Reinforcement Learning (RL) to automate parameter tuning for an Augmented Lagrangian Method (ALM) solver for the Capacitated Vehicle Routing Problem (CVRP), with support for both classical and quantum subproblem solvers.

## 📜 Overview

The Capacitated Vehicle Routing Problem (CVRP) is a classic NP-hard optimization problem. This project tackles it using an Augmented Lagrangian Method (ALM), where the traditionally difficult task of tuning penalty parameters is automated by a Soft Actor-Critic (SAC) reinforcement learning agent.

The framework is designed to be flexible and supports two distinct backends for solving the ALM subproblems:
* **Classical Solver**: A fast, heuristic-based classical solver.
* **Quantum Solver**: A quantum approach that formulates the subproblem as a QUBO and solves it using Qiskit's Variational Quantum Eigensolver (VQE).

## ✨ Features

* **Modular Architecture**: The code is organized into distinct, reusable modules for data handling, optimization, reinforcement learning, and quantum computing.
* **Configuration-Driven**: All experiments are controlled via simple YAML configuration files. No code changes are needed to change parameters, solvers, or problem instances.
* **Hybrid Classical-Quantum**: Seamlessly switch between a classical and a quantum subproblem solver via the configuration.
* **RL-Powered Automation**: Leverages a Soft Actor-Critic (SAC) agent to intelligently learn and set the penalty parameters in the ALM framework.
* **Reproducible**: With a single entry point and version-controlled dependencies, experiments are easy to reproduce.

## 🏗️ Project Structure

The project is organized into a clean and maintainable structure:

```
adaptive_quantum_cvrp/
├── configs/                # Experiment configuration files
├── data/                   # CVRP instance files (.vrp)
├── results/                # Output directory for logs, models, and solutions
├── src/
│   └── adaptive_quantum_cvrp/ # Main source code package
│       ├── alm/             # Augmented Lagrangian Method core
│       ├── common/          # Shared data structures (Instance, Solution)
│       ├── quantum/         # Quantum components (QUBO, VQE)
│       ├── rl/              # Reinforcement Learning (Agent, Environment)
│       └── utils/           # Helper utilities (logging, config loading)
├── tests/                  # Unit tests for the modules
├── requirements.txt        # Project dependencies
└── run_experiment.py       # Single entry point to run all experiments
```

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* `pip` and `venv`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SMU-Quantum/adaptive_quantum_cvrp
    cd adaptive-quantum-cvrp
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

All experiments are launched from the `run_experiment.py` script, which takes a single argument: the path to a configuration file.

### Running a Vanilla ALM Experiment

This runs the ALM solver with a fixed penalty parameter, using the classical backend.

```bash
python run_experiment.py --config configs/experiments/classical_alm.yaml
```

### Training the RL Agent (Classical Backend)

This trains the SAC agent to learn penalty parameters using the classical solver as its backend.

```bash
python run_experiment.py --config configs/experiments/rl_classical_alm.yaml
```

### Training the RL Agent (Quantum Backend)

This trains the SAC agent using the VQE-based quantum solver. **Note:** This is computationally intensive and is best suited for small CVRP instances.

```bash
python run_experiment.py --config configs/experiments/rl_quantum_alm.yaml
```

### Customizing Experiments

To run on a different instance or change parameters, simply copy one of the example config files in `configs/experiments/` and modify it. You can change the instance path, solver type, RL hyperparameters, and more.

## 📄 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{sharma2025hybridlearningoptimizationmethods,
      title={Hybrid Learning and Optimization methods for solving Capacitated Vehicle Routing Problem}, 
      author={Monit Sharma and Hoong Chuin Lau},
      year={2025},
      eprint={2509.15262},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph},
      url={https://arxiv.org/abs/2509.15262}, 
}
```

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
