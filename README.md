# SI-CL-SDEO: HDFS Block Placement Simulation

This project contains a Python-based simulation environment for computing and evaluating Hadoop Distributed File System (HDFS) block placement strategies. It implements a novel hybrid metaheuristic approach called **SI-CL-SDEO** (Swarm Intelligence with Chaotic Local Search and Differential Evolution Optimization) and compares it against a baseline **MBFOA** (Modified Bacterial Foraging Optimization Algorithm).

## Key Features

- **Genuine HDFS Simulation Model**: Simulates NameNodes, DataNodes, Racks, caching, and network topological constraints.
- **SI-CL-SDEO Implementation**: A mathematically robust hybrid algorithm combining Swarm Intelligence (Salp Swarm Optimization) and Differential Evolution to optimize block placement.
- **MBFOA Baseline**: Implements a genuine $O(N^2)$ complex swarming model representing the Modified Bacterial Foraging Optimization Algorithm.
- **Comprehensive Metrics Suite**: 
  - **Latency**: Average Read Latency (ARL), Average Write Latency (AWL) accounting for network pipelines.
  - **Reliability**: Data Availability (DA), Fault Tolerance under different Replication Factors ($\psi$).
  - **Network**: Packet Delivery Ratio (PDR) and Bandwidth Utilization.
- **Hardware & Congestion Calibration**: Maps raw Python CPU math execution time realistically to Hadoop operational speeds.

## Project Structure

```text
si-cl-sdeo/
├── main.py                     # Main simulation orchestrator and entry point
├── requirements.txt            # Python dependencies
├── notebooks/
├── src/
│   ├── algorithms/             # Core Optimization Algorithms
│   │   ├── baselines.py        # MBFOA baseline logic
│   │   ├── fitness.py          # Multiobjective fitness evaluation function
│   │   └── si_cl_sdeo.py       # SI-CL-SDEO algorithm
│   ├── hdfs_env/               # Hadoop Environment Simulators
│   │   ├── cluster.py          # Cluster topology modeling
│   │   ├── datanode.py         # Hardware and metric modeling for DataNodes
│   │   └── workload.py         # Simulated HDFS write workloads and stress tests
│   └── metrics/                # Evaluation Metrics Evaluators
│       ├── availability.py     # DA and Fault Tolerance logic
│       ├── latency.py          # Read and Write Latency estimators
│       └── network.py          # PDR and bandwidth calculation logic
└── tests/                      # Unit testing framework (WIP)
    └── test_fitness.py
```

## Setup & Installation

It is recommended to use a virtual environment. The codebase primarily requires `numpy` and `matplotlib` for the mathematical evaluations and visual reporting.

```bash
# Clone the repository and navigate into it
cd si-cl-sdeo

# Create and activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

*(Note: `pandas` and `scipy` are listed in requirements but aren't strictly necessary for the core logic, which is tightly optimized using only `numpy`.)*

## Running the Simulation

The execution is driven entirely by `main.py`. This script will initialize the simulated DataNodes, warm up the CPU cache, simulate the network topology, execute both the SI-CL-SDEO and MBFOA algorithms under different stress loads, and finally produce comparative visual graphs.

```bash
python main.py
```

### Outputs

Upon running the orchestration script, two primary visual graphs will be automatically generated in the root directory:
1. `calibrated_execution_time.png`: Execution delay (ms) for both algorithms across Low, Medium, and High HDFS cluster load environments.
2. `calibrated_da_pdr.png`: Line plots comparing the probability curves of Data Availability and Packet Delivery Ratios across varying Rack counts and Replication Factors ($\psi={2, 3, 4}$).

## Testing

The project is structured to support the `pytest` testing framework. Currently, the test suite is empty and undergoing development. You can trigger the suite with:

```bash
pytest tests/
```

## Known Technical Considerations

- The multi-objective `evaluate_fitness` assigns an infinite penalty if the simulated cluster crashes completely.
- The `CR` (Crossover rate) and `F` (Mutation factor) used in the Differential Evolution phase of the SI-CL-SDEO algorithm are hardcoded inside `src/algorithms/si_cl_sdeo.py` to `0.7` and `0.5` respectively, to recreate the paper's default conditions.
- Running operations under High Load severely impacts the MBFOA algorithm due to its inherent Quadratic $O(N^2)$ network iteration penalty, deliberately demonstrating its non-suitability for massive topologies in the benchmarks.
