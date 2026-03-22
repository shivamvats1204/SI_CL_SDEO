from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.baselines import optimize_mbfoa
from src.algorithms.fitness import OptimizationResult
from src.algorithms.si_cl_sdeo import si_cl_sdeo_optimize
from src.hdfs_env.cluster import HDFSCluster
from src.hdfs_env.workload import WorkloadGenerator


RESULTS_PATH = Path("simulation_results.json")
EXECUTION_FIGURE = Path("execution_time_comparison.png")
AVAILABILITY_FIGURE = Path("availability_pdr_vs_racks.png")
METRICS_FIGURE = Path("latency_fault_network.png")


@dataclass(frozen=True)
class Scenario:
    load_condition: str
    dataset_size_gb: int
    failure_rate: float


@dataclass(frozen=True)
class ExperimentSummary:
    algorithm: str
    load_condition: str
    dataset_size_gb: int
    failure_rate: float
    placements_completed: int
    wall_clock_time_ms: float
    optimizer_time_ms: float
    average_optimizer_time_ms: float
    average_replication_factor: float
    average_read_latency_ms: float
    average_write_latency_ms: float
    data_availability_pct: float
    fault_tolerance_pct: float
    packet_delivery_ratio_pct: float
    network_utilization_pct: float
    dataset_block_count: int = 0
    sampled_block_count: int = 0
    sampled_data_gb: float = 0.0


SCENARIOS = {
    "Low": Scenario(load_condition="Low", dataset_size_gb=100, failure_rate=0.01),
    "Medium": Scenario(load_condition="Medium", dataset_size_gb=500, failure_rate=0.05),
    "High": Scenario(load_condition="High", dataset_size_gb=1000, failure_rate=0.10),
}


def default_sampled_block_count(load_condition: str, dataset_block_count: int) -> int:
    if dataset_block_count <= 0:
        return 0

    baseline_samples = {"Low": 18, "Medium": 24, "High": 30}
    if load_condition not in baseline_samples:
        raise ValueError(f"unsupported load condition: {load_condition}")

    scaled_samples = int(round(np.sqrt(dataset_block_count) * 0.45))
    return min(dataset_block_count, max(baseline_samples[load_condition], scaled_samples))


def select_blocks_for_simulation(dataset: Sequence, sample_count: int) -> list:
    if not dataset or sample_count <= 0:
        return []

    if sample_count >= len(dataset):
        return list(dataset)

    indices = np.linspace(0, len(dataset) - 1, num=sample_count)
    sampled_indices: list[int] = []
    seen_indices: set[int] = set()

    for index in np.rint(indices).astype(int):
        normalized_index = int(index)
        if normalized_index in seen_indices:
            continue
        sampled_indices.append(normalized_index)
        seen_indices.add(normalized_index)

    if len(sampled_indices) < sample_count:
        for index in range(len(dataset)):
            if index in seen_indices:
                continue
            sampled_indices.append(index)
            seen_indices.add(index)
            if len(sampled_indices) == sample_count:
                break

    sampled_indices.sort()
    return [dataset[index] for index in sampled_indices]


def run_algorithm_simulation(
    optimizer,
    algorithm_name: str,
    scenario: Scenario,
    *,
    num_nodes: int = 50,
    num_racks: int = 10,
    fixed_replication_factor: int | None = None,
    cluster_seed: int = 7,
    workload_seed: int = 19,
    optimizer_seed: int = 101,
    placement_request_limit: int | None = None,
) -> ExperimentSummary:
    cluster = HDFSCluster(num_nodes=num_nodes, num_racks=num_racks, seed=cluster_seed)
    workload = WorkloadGenerator(
        load_condition=scenario.load_condition,
        request_count=6000,
        seed=workload_seed,
    )
    dataset = workload.generate_dataset(scenario.dataset_size_gb)
    request_limit = placement_request_limit or default_sampled_block_count(
        scenario.load_condition,
        len(dataset),
    )
    sampled_blocks = select_blocks_for_simulation(dataset, request_limit)
    sampled_data_gb = sum(block.size_mb for block in sampled_blocks) / 1024.0

    cluster.reset_loads()
    workload.apply_load_to_cluster(cluster)

    placement_results: list[OptimizationResult] = []
    wall_clock_start = perf_counter()

    for index, block in enumerate(sampled_blocks):
        active_nodes = cluster.simulate_cluster_state(scenario.failure_rate)
        if not active_nodes:
            cluster.cool_down(0.02)
            continue

        result = optimizer(
            active_nodes,
            block_size_mb=block.size_mb,
            fixed_replication_factor=fixed_replication_factor,
            failure_rate=scenario.failure_rate,
            seed=optimizer_seed + index,
        )
        if not result.selected_node_ids:
            cluster.cool_down(0.01)
            continue

        cluster.place_block(result.selected_node_ids, block.size_mb)
        cluster.cool_down(0.004)
        placement_results.append(result)

    wall_clock_time_ms = (perf_counter() - wall_clock_start) * 1000.0

    if not placement_results:
        return ExperimentSummary(
            algorithm=algorithm_name,
            load_condition=scenario.load_condition,
            dataset_size_gb=scenario.dataset_size_gb,
            failure_rate=scenario.failure_rate,
            placements_completed=0,
            wall_clock_time_ms=wall_clock_time_ms,
            optimizer_time_ms=0.0,
            average_optimizer_time_ms=0.0,
            average_replication_factor=0.0,
            average_read_latency_ms=float("inf"),
            average_write_latency_ms=float("inf"),
            data_availability_pct=0.0,
            fault_tolerance_pct=0.0,
            packet_delivery_ratio_pct=0.0,
            network_utilization_pct=0.0,
            dataset_block_count=len(dataset),
            sampled_block_count=len(sampled_blocks),
            sampled_data_gb=sampled_data_gb,
        )

    return ExperimentSummary(
        algorithm=algorithm_name,
        load_condition=scenario.load_condition,
        dataset_size_gb=scenario.dataset_size_gb,
        failure_rate=scenario.failure_rate,
        placements_completed=len(placement_results),
        wall_clock_time_ms=wall_clock_time_ms,
        optimizer_time_ms=sum(result.execution_time_ms for result in placement_results),
        average_optimizer_time_ms=mean(result.execution_time_ms for result in placement_results),
        average_replication_factor=mean(result.replication_factor for result in placement_results),
        average_read_latency_ms=mean(result.metrics.average_read_latency_ms for result in placement_results),
        average_write_latency_ms=mean(result.metrics.average_write_latency_ms for result in placement_results),
        data_availability_pct=mean(result.metrics.data_availability_pct for result in placement_results),
        fault_tolerance_pct=mean(result.metrics.fault_tolerance_pct for result in placement_results),
        packet_delivery_ratio_pct=mean(result.metrics.packet_delivery_ratio_pct for result in placement_results),
        network_utilization_pct=mean(result.metrics.network_utilization_pct for result in placement_results),
        dataset_block_count=len(dataset),
        sampled_block_count=len(sampled_blocks),
        sampled_data_gb=sampled_data_gb,
    )


def average_summaries(summaries: list[ExperimentSummary]) -> ExperimentSummary:
    if not summaries:
        raise ValueError("summaries must not be empty")

    return ExperimentSummary(
        algorithm=summaries[0].algorithm,
        load_condition=summaries[0].load_condition,
        dataset_size_gb=summaries[0].dataset_size_gb,
        failure_rate=summaries[0].failure_rate,
        placements_completed=int(round(mean(summary.placements_completed for summary in summaries))),
        wall_clock_time_ms=mean(summary.wall_clock_time_ms for summary in summaries),
        optimizer_time_ms=mean(summary.optimizer_time_ms for summary in summaries),
        average_optimizer_time_ms=mean(summary.average_optimizer_time_ms for summary in summaries),
        average_replication_factor=mean(summary.average_replication_factor for summary in summaries),
        average_read_latency_ms=mean(summary.average_read_latency_ms for summary in summaries),
        average_write_latency_ms=mean(summary.average_write_latency_ms for summary in summaries),
        data_availability_pct=mean(summary.data_availability_pct for summary in summaries),
        fault_tolerance_pct=mean(summary.fault_tolerance_pct for summary in summaries),
        packet_delivery_ratio_pct=mean(summary.packet_delivery_ratio_pct for summary in summaries),
        network_utilization_pct=mean(summary.network_utilization_pct for summary in summaries),
        dataset_block_count=int(round(mean(summary.dataset_block_count for summary in summaries))),
        sampled_block_count=int(round(mean(summary.sampled_block_count for summary in summaries))),
        sampled_data_gb=mean(summary.sampled_data_gb for summary in summaries),
    )


def run_execution_time_experiment() -> dict[str, list[ExperimentSummary]]:
    trial_count = 2
    execution_results = {"SI-CL-SDEO": [], "MBFOA": []}

    for load_name, scenario in SCENARIOS.items():
        si_trials: list[ExperimentSummary] = []
        mbfoa_trials: list[ExperimentSummary] = []

        for trial in range(trial_count):
            base_seed = (1000 * (trial + 1)) + (100 * (list(SCENARIOS.keys()).index(load_name) + 1))
            si_trials.append(
                run_algorithm_simulation(
                    si_cl_sdeo_optimize,
                    "SI-CL-SDEO",
                    scenario,
                    fixed_replication_factor=3,
                    cluster_seed=base_seed,
                    workload_seed=base_seed + 1,
                    optimizer_seed=base_seed + 10,
                )
            )
            mbfoa_trials.append(
                run_algorithm_simulation(
                    optimize_mbfoa,
                    "MBFOA",
                    scenario,
                    fixed_replication_factor=3,
                    cluster_seed=base_seed,
                    workload_seed=base_seed + 1,
                    optimizer_seed=base_seed + 10,
                )
            )

        execution_results["SI-CL-SDEO"].append(average_summaries(si_trials))
        execution_results["MBFOA"].append(average_summaries(mbfoa_trials))

    return execution_results


def run_rack_sweep_experiment() -> tuple[np.ndarray, dict[int, list[float]], dict[int, list[float]]]:
    racks_array = np.arange(2, 22, 2)
    da_results = {2: [], 3: [], 4: []}
    pdr_results = {2: [], 3: [], 4: []}
    scenario = SCENARIOS["Medium"]
    trial_count = 3

    for rack_count in racks_array:
        for replication_factor in (2, 3, 4):
            summaries: list[ExperimentSummary] = []
            for trial in range(trial_count):
                seed_offset = trial * 100
                summaries.append(
                    run_algorithm_simulation(
                        si_cl_sdeo_optimize,
                        "SI-CL-SDEO",
                        scenario,
                        num_nodes=max(20, rack_count * 6),
                        num_racks=int(rack_count),
                        fixed_replication_factor=replication_factor,
                        cluster_seed=2000 + (int(rack_count) * 10) + replication_factor + seed_offset,
                        workload_seed=2500 + (int(rack_count) * 10) + replication_factor + seed_offset,
                        optimizer_seed=3000 + (int(rack_count) * 10) + replication_factor + seed_offset,
                        placement_request_limit=8,
                    )
                )
            summary = average_summaries(summaries)
            da_results[replication_factor].append(summary.data_availability_pct)
            pdr_results[replication_factor].append(summary.packet_delivery_ratio_pct)

    return racks_array, da_results, pdr_results


def run_replication_factor_metric_summaries() -> dict[str, dict[int, ExperimentSummary]]:
    scenario = SCENARIOS["Medium"]
    summaries = {"fault_tolerance": {}, "latency": {}, "network": {}}
    trial_count = 2

    for replication_factor in (2, 3, 4):
        for category, num_racks, seed_base in (
            ("fault_tolerance", 6, 4000),
            ("latency", 8, 5000),
            ("network", 10, 6000),
        ):
            category_trials: list[ExperimentSummary] = []
            for trial in range(trial_count):
                seed_offset = trial * 100
                category_trials.append(
                    run_algorithm_simulation(
                        si_cl_sdeo_optimize,
                        "SI-CL-SDEO",
                        scenario,
                        num_nodes=48,
                        num_racks=num_racks,
                        fixed_replication_factor=replication_factor,
                        cluster_seed=seed_base + replication_factor + seed_offset,
                        workload_seed=(seed_base + 100) + replication_factor + seed_offset,
                        optimizer_seed=(seed_base + 200) + replication_factor + seed_offset,
                        placement_request_limit=10,
                    )
                )
            summaries[category][replication_factor] = average_summaries(category_trials)

    return summaries


def plot_execution_time(execution_results: dict[str, list[ExperimentSummary]]) -> None:
    load_labels = list(SCENARIOS.keys())
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"SI-CL-SDEO": "o", "MBFOA": "s"}
    colors = {"SI-CL-SDEO": "#004f6e", "MBFOA": "#cc5500"}

    for algorithm_name, summaries in execution_results.items():
        ax.plot(
            load_labels,
            [summary.optimizer_time_ms for summary in summaries],
            marker=markers[algorithm_name],
            color=colors[algorithm_name],
            linewidth=2.5,
            markersize=8,
            label=algorithm_name,
        )

    ax.set_title("Total Simulated Optimizer Time for Sampled Placements (RF=3)")
    ax.set_xlabel("Load Condition")
    ax.set_ylabel("Total Sampled Optimizer Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EXECUTION_FIGURE, dpi=300)
    plt.close(fig)


def plot_da_and_pdr(racks_array: np.ndarray, da_results: dict[int, list[float]], pdr_results: dict[int, list[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {2: "#4d4d4d", 3: "#c0392b", 4: "#1f5aa6"}
    markers = {2: "s", 3: "o", 4: "^"}

    for replication_factor in (2, 3, 4):
        label = rf"$\psi={replication_factor}$"
        axes[0].plot(
            racks_array,
            da_results[replication_factor],
            marker=markers[replication_factor],
            color=colors[replication_factor],
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            racks_array,
            pdr_results[replication_factor],
            marker=markers[replication_factor],
            color=colors[replication_factor],
            linewidth=2.0,
            label=label,
        )

    axes[0].set_title("Simulated Data Availability vs. Rack Count")
    axes[0].set_xlabel("Racks")
    axes[0].set_ylabel("Availability (%)")
    axes[0].set_xticks(racks_array)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Simulated Packet Delivery Ratio vs. Rack Count")
    axes[1].set_xlabel("Racks")
    axes[1].set_ylabel("PDR (%)")
    axes[1].set_xticks(racks_array)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(AVAILABILITY_FIGURE, dpi=300)
    plt.close(fig)


def plot_replication_factor_metrics(metric_summaries: dict[str, dict[int, ExperimentSummary]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    replication_factors = [2, 3, 4]
    bar_colors = ["#0f766e", "#d97706", "#b91c1c"]

    fault_values = [
        metric_summaries["fault_tolerance"][replication_factor].fault_tolerance_pct
        for replication_factor in replication_factors
    ]
    axes[0].bar(replication_factors, fault_values, color=bar_colors)
    axes[0].set_title("Simulated Fault Tolerance at 6 Racks")
    axes[0].set_xlabel("Replication Factor")
    axes[0].set_ylabel("Fault Tolerance (%)")

    read_values = [
        metric_summaries["latency"][replication_factor].average_read_latency_ms
        for replication_factor in replication_factors
    ]
    write_values = [
        metric_summaries["latency"][replication_factor].average_write_latency_ms
        for replication_factor in replication_factors
    ]
    width = 0.35
    x_positions = np.arange(len(replication_factors))
    axes[1].bar(x_positions - (width / 2), read_values, width=width, color="#2563eb", label="ARL")
    axes[1].bar(x_positions + (width / 2), write_values, width=width, color="#ea580c", label="AWL")
    axes[1].set_title("Simulated Read/Write Latency at 8 Racks")
    axes[1].set_xlabel("Replication Factor")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xticks(x_positions, replication_factors)
    axes[1].legend()

    network_values = [
        metric_summaries["network"][replication_factor].network_utilization_pct
        for replication_factor in replication_factors
    ]
    axes[2].bar(replication_factors, network_values, color=bar_colors)
    axes[2].set_title("Simulated Network Utilization at 10 Racks")
    axes[2].set_xlabel("Replication Factor")
    axes[2].set_ylabel("Utilization (%)")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35, axis="y")

    fig.tight_layout()
    fig.savefig(METRICS_FIGURE, dpi=300)
    plt.close(fig)


def save_results(
    execution_results: dict[str, list[ExperimentSummary]],
    rack_sweep,
    metric_summaries: dict[str, dict[int, ExperimentSummary]],
) -> None:
    racks_array, da_results, pdr_results = rack_sweep
    payload = {
        "execution_time_experiment": {
            algorithm: [asdict(summary) for summary in summaries]
            for algorithm, summaries in execution_results.items()
        },
        "rack_sweep": {
            "racks": racks_array.tolist(),
            "data_availability_pct": da_results,
            "packet_delivery_ratio_pct": pdr_results,
        },
        "replication_factor_metrics": {
            category: {replication_factor: asdict(summary) for replication_factor, summary in summaries.items()}
            for category, summaries in metric_summaries.items()
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(
    execution_results: dict[str, list[ExperimentSummary]],
    metric_summaries: dict[str, dict[int, ExperimentSummary]],
) -> None:
    print("Execution Time Comparison")
    for algorithm_name, summaries in execution_results.items():
        for summary in summaries:
            print(
                f"{algorithm_name:11s} | {summary.load_condition:6s} | "
                f"ET={summary.optimizer_time_ms:7.2f} ms | "
                f"Sample={summary.sampled_block_count:3d}/{summary.dataset_block_count:4d} blocks "
                f"({summary.sampled_data_gb:6.2f} GB) | "
                f"RF={summary.average_replication_factor:.2f} | "
                f"DA={summary.data_availability_pct:6.2f}% | "
                f"PDR={summary.packet_delivery_ratio_pct:6.2f}%"
            )

    print("\nReplication Factor Metric Summary")
    for replication_factor in (2, 3, 4):
        latency_summary = metric_summaries["latency"][replication_factor]
        fault_summary = metric_summaries["fault_tolerance"][replication_factor]
        network_summary = metric_summaries["network"][replication_factor]
        print(
            f"psi={replication_factor} | "
            f"ARL={latency_summary.average_read_latency_ms:6.2f} ms | "
            f"AWL={latency_summary.average_write_latency_ms:6.2f} ms | "
            f"FT={fault_summary.fault_tolerance_pct:6.2f}% | "
            f"NET={network_summary.network_utilization_pct:6.2f}%"
        )


def run_comprehensive_simulation() -> None:
    print("Running SI-CL-SDEO HDFS replication simulation...")
    execution_results = run_execution_time_experiment()
    rack_sweep = run_rack_sweep_experiment()
    metric_summaries = run_replication_factor_metric_summaries()

    plot_execution_time(execution_results)
    plot_da_and_pdr(*rack_sweep)
    plot_replication_factor_metrics(metric_summaries)
    save_results(execution_results, rack_sweep, metric_summaries)
    print_summary(execution_results, metric_summaries)

    print(f"\nSaved {EXECUTION_FIGURE}")
    print(f"Saved {AVAILABILITY_FIGURE}")
    print(f"Saved {METRICS_FIGURE}")
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    run_comprehensive_simulation()
