# optimize-anything-assignment8

A comprehensive laboratory assignment repository exploring continuous, discrete, submodular, parallel, distributed, and quantum optimization techniques across diverse applied tracks.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Dependencies](#dependencies)
- [How to Run Each Track](#how-to-run-each-track)
- [Reproducibility](#reproducibility)
- [AI Tools Register](#ai-tools-register)
- [Commit Convention](#commit-convention)
- [Acknowledgements of External Libraries](#acknowledgements-of-external-libraries)

---

## Project Overview
This repository contains implementations, experimental setups, benchmarks, and analyses for university optimization coursework covering:
- Classical continuous and discrete optimization algorithms
- Submodular optimization, distributed learning, and multi-task optimization
- Quantum optimization (QAOA, VQE, QNNs, quantum annealing, quantum algorithms)
- Systems and applied tracks (compilers, operating systems, DBMS, software engineering)

All experiments emphasize mathematical formulation, empirical benchmarking, rigorous error analysis, and reproducibility.

---

## Repository Structure
```
optimize-anything-assignment8/
├── .gitignore
├── requirements.txt
├── README.md
├── CHECKLIST.md
├── FAILURE_LOG.md
├── TOOL_COMPARISON.md
├── common/                          # Shared Python utilities package
│   ├── __init__.py
│   ├── seeding.py                   # Deterministic global seed management
│   ├── timing.py                    # Wall-clock profiling (decorator & context manager)
│   ├── plotting.py                  # Standardized convergence curves & bar charts
│   ├── stats.py                     # Multi-seed stochastic experiment statistics
│   └── io_utils.py                  # JSON/CSV persistence helpers
├── results/                         # Output artifacts and figures
│   ├── track_a/                     # Continuous optimization outputs
│   ├── track_b/                     # Discrete optimization outputs
│   ├── track_c/                     # Quantum optimization outputs
│   ├── track_d/                     # Submodular & distributed outputs
│   ├── track_e/                     # Optimization case study outputs
│   ├── track_f/                     # Compiler optimization outputs
│   ├── track_g/                     # xv6 kernel optimization outputs
│   ├── track_h/                     # DBMS query optimization outputs
│   ├── track_i/                     # Quantum algorithms & QML outputs
│   └── track_j/                     # Software engineering optimization outputs
├── tracks/                          # Track implementation modules
│   ├── track_a_continuous/
│   ├── track_b_discrete/
│   ├── track_c_quantum_opt/
│   ├── track_d_submodular_parallel_distributed/
│   ├── track_e_case_study/
│   ├── track_f_compiler/
│   ├── track_g_xv6/
│   ├── track_h_dbms/
│   ├── track_i_quantum_algos_qml/
│   └── track_j_software_eng/
└── prompts/                         # AI collaboration logs per track
    ├── track_a_continuous/
    ├── track_b_discrete/
    ├── track_c_quantum_opt/
    ├── track_d_submodular_parallel_distributed/
    ├── track_e_case_study/
    ├── track_f_compiler/
    ├── track_g_xv6/
    ├── track_h_dbms/
    ├── track_i_quantum_algos_qml/
    └── track_j_software_eng/
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.15)
- Git
- Virtual environment manager (`venv` or `conda`)

### Environment Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd optimize-anything-assignment8
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows (Command Prompt / PowerShell):
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   pytest
   ```

---

## Dependencies
The project dependencies are specified in `requirements.txt`:
- **Core Numerical & Scientific:** `numpy`, `scipy`, `pandas`
- **Visualization:** `matplotlib`
- **Machine Learning & Deep Learning:** `scikit-learn`, `torch`
- **Combinatorial & Discrete Optimization:** `pulp`, `networkx`
- **Quantum Computing & Optimization:** `qiskit`, `qiskit-aer`, `pennylane`
- **Testing:** `pytest`

---

## How to Run Each Track
Each track folder under `tracks/` provides its own standalone scripts, benchmarks, and unit tests. Run the corresponding entrypoints or tests as follows:

| Track | Description | Command / Execution Example |
| :--- | :--- | :--- |
| **Track A** | Continuous Optimization | `pytest tracks/track_a_continuous/tests/` |
| **Track B** | Discrete Optimization | `pytest tracks/track_b_discrete/tests/` |
| **Track C** | Quantum Optimization | `pytest tracks/track_c_quantum_opt/tests/` |
| **Track D** | Submodular, Parallel & Distributed | `pytest tracks/track_d_submodular_parallel_distributed/tests/` |
| **Track E** | Case Study | `pytest tracks/track_e_case_study/tests/` |
| **Track F** | Compiler Optimization | `pytest tracks/track_f_compiler/tests/` |
| **Track G** | xv6 Kernel Optimization | `pytest tracks/track_g_xv6/tests/` |
| **Track H** | DBMS Query Optimization | `pytest tracks/track_h_dbms/tests/` |
| **Track I** | Quantum Algorithms & QML | `pytest tracks/track_i_quantum_algos_qml/tests/` |
| **Track J** | Software Engineering Optimization | `pytest tracks/track_j_software_eng/tests/` |

---

## Reproducibility
To ensure consistent, reproducible results across runs and platforms:
- **Global Seeds:** Always invoke `common.seeding.set_global_seed(seed)` at the start of any script or experiment. Default benchmark seed: `42`.
- **Stochastic Evaluation:** For algorithms with random initialization or stochastic updates, run benchmarks across multiple seeds using `common.stats.run_multiseed_experiment(...)`.
- **System & Hardware Details:**
  - OS: [e.g., Windows 11 / Ubuntu 22.04 LTS]
  - CPU: [e.g., Intel Core i7 / AMD Ryzen 9 / Apple M-series]
  - GPU (if applicable): [e.g., NVIDIA RTX 3080 / CPU only]
  - Python Version: [e.g., 3.10.x / 3.11.x]

---

## AI Tools Register
All AI tool interactions used throughout the design, formulation, implementation, and debugging must be logged below and recorded in detail in `prompts/<track>/PROMPT_LOG.md`.

| Tool | Model version | Tracks | Dates | Kind of work |
| :--- | :--- | :--- | :--- | :--- |
| Antigravity IDE | Claude Sonnet 5.5 | All (setup), A, B, C, D, F, G, H, I | 2026-10-28, 2026-10-29, 2026-10-30, 2026-10-31 | Planning, code generation, unit-test writing, experiment scripting, debugging, documentation (repository skeleton, templates, shared utilities in common/, Track A continuous optimization, Track B discrete optimization, Track C quantum optimization, Track D submodular, parallel and distributed optimization, Track F compiler optimization, Track H DBMS query optimization, Track I quantum algorithms and QML); test correction; experiment scripting (exp1–exp6, Track B discrete benchmarks, Track C quantum experiments C1–C5, Track D experiments D1–D4, Track F experiments F1–F3, Track H DBMS index selection experiments, Track I quantum algorithms and QML experiments I1–I5, run_all, smoke tests); code generation (kernel and user programs), documentation |
| Antigravity IDE | Gemini 3.8 Flash | All Tracks | 2026-10-28 | Repository skeleton, templates, shared utilities |
| [e.g., Claude] | [e.g., Claude 3.7 Sonnet] | [e.g., Track A, Track C] | [YYYY-MM-DD] | [e.g., Algorithm formulation, test generation] |
| [e.g., ChatGPT] | [e.g., GPT-4o] | [e.g., Track B] | [YYYY-MM-DD] | [e.g., Debugging ILP formulations] |

---

## Commit Convention
To maintain attribution and transparent traceability between manual edits and AI generation:
- AI-assisted commits must be prefixed with `[AI:tool-name]`, e.g.:
  ```
  [AI:antigravity] Implement AdaGrad and RMSProp optimizers in Track A
  ```
- Manual student edits must be prefixed with `[MANUAL]`, e.g.:
  ```
  [MANUAL] Correct boundary condition in Branch-and-Bound pruning
  ```

---

## Acknowledgements of External Libraries
We gratefully acknowledge the developers and communities behind the following open-source libraries utilized in this project:
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/): Fundamental algorithms for scientific computing.
- [Matplotlib](https://matplotlib.org/): Comprehensive visualization library.
- [Pandas](https://pandas.pydata.org/): Data manipulation and tabular analysis.
- [Scikit-Learn](https://scikit-learn.org/): Machine learning primitives and benchmarks.
- [PyTorch](https://pytorch.org/): Tensors and dynamic neural network optimization.
- [PuLP](https://coin-or.github.io/pulp/): Linear and integer programming modeler.
- [NetworkX](https://networkx.org/): Network and graph optimization algorithms.
- [Qiskit](https://qiskit.org/) & [Qiskit Aer](https://github.com/Qiskit/qiskit-aer): Quantum information processing and simulation.
- [PennyLane](https://pennylane.ai/): Differentiable quantum computing and QML framework.
- [pytest](https://docs.pytest.org/): Python testing framework.
