# Adaptive Quantum CVRP

Implementation of the paper:

**Hybrid Learning and Optimization Methods for Solving Capacitated Vehicle Routing Problem**

Citation:
Sharma, M., & Lau, H. C. (2025). Hybrid Learning and Optimization Methods for Solving Capacitated Vehicle Routing Problem. Zenodo. https://doi.org/10.5281/zenodo.15621110

This repository provides an adaptive framework for solving the Capacitated Vehicle Routing Problem (CVRP) using quantum algorithms and reinforcement learning techniques. The project includes classical and quantum solvers, RL-based penalty learners, and experiment scripts for benchmarking and evaluation.

## Features

- Quantum and classical solvers for CVRP
- Reinforcement learning penalty learner for constraint handling
- Batch experiment scripts for automated evaluation
- Pre-trained models and sample results
- Support for standard CVRP instance formats

## Project Structure

```
adaptive_quantum_cvrp/
├── data/                # CVRP instance files and solutions
├── dummy_tiny_instances/# Tiny test instances
├── experiments/         # Experiment scripts
├── models/              # Pre-trained models and RL outputs
├── notes/               # Project notes
├── results/             # Experiment results
├── src/                 # Source code (solvers, RL, quantum modules)
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: virtual environment (venv, conda, etc.)

### Installation

```bash
git clone https://github.com/SMU-Quantum/adaptive_quantum_cvrp.git
cd adaptive_quantum_cvrp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

- Run experiments:
	```bash
	python experiments/run_alm_experiment.py
	python experiments/run_rl_penalty_learner_experiment.py
	```
- Evaluate on provided CVRP instances in `data/`
- Pre-trained models are available in `models/`

### Example

```python
from src.alm import quantum_solver
# ...existing code...
solution = quantum_solver.solve(instance_path)
print(solution)
```

## Data

- Standard CVRP instances from CVRPLIB in `data/cvrplib_instances_*`
- Tiny test instances in `dummy_tiny_instances/`

## Results

- Experiment outputs and solution files are stored in `results/`


## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration, please contact the maintainers or open an issue.
