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
LOW_LOAD_DATASET_ET_FIGURE = Path("low_load_execution_time_vs_data_size.png")
MEDIUM_LOAD_DATASET_ET_FIGURE = Path("medium_load_execution_time_vs_data_size.png")
HIGH_LOAD_DATASET_ET_FIGURE = Path("high_load_execution_time_vs_data_size.png")
TIME_COMPLEXITY_FIGURE = Path("time_complexity_vs_racks.png")
DATA_AVAILABILITY_FIGURE = Path("data_availability_vs_racks.png")
PDR_FIGURE = Path("packet_delivery_ratio_vs_racks.png")
FAULT_TOLERANCE_FIGURE = Path("fault_tolerance_vs_racks.png")
NETWORK_UTILIZATION_FIGURE = Path("network_utilization_vs_racks.png")
READ_LATENCY_FIGURE = Path("read_latency_vs_racks.png")
WRITE_LATENCY_FIGURE = Path("write_latency_vs_racks.png")
LOW_LOAD_PERFORMANCE_FIGURE = Path("low_load_comparative_performance.png")
MEDIUM_LOAD_PERFORMANCE_FIGURE = Path("medium_load_comparative_performance.png")
HIGH_LOAD_PERFORMANCE_FIGURE = Path("high_load_comparative_performance.png")


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

    baseline_samples = {"Low": 8, "Medium": 11, "High": 14}
    if load_condition not in baseline_samples:
        raise ValueError(f"unsupported load condition: {load_condition}")

    return min(dataset_block_count, baseline_samples[load_condition])


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


def projected_execution_time_seconds(summary: ExperimentSummary) -> float:
    if summary.optimizer_time_ms <= 0.0 or summary.sampled_data_gb <= 0.0:
        return 0.0

    projected_total_time_ms = (summary.optimizer_time_ms / summary.sampled_data_gb) * summary.dataset_size_gb
    return projected_total_time_ms / 1000.0


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


def run_dataset_size_execution_profiles() -> tuple[np.ndarray, dict[str, dict[int, list[float]]]]:
    dataset_sizes = np.arange(10, 160, 10)
    execution_profiles = {
        load_name: {2: [], 3: [], 4: []}
        for load_name in SCENARIOS
    }
    placement_request_limit = 6
    trial_count = 2

    for load_index, (load_name, base_scenario) in enumerate(SCENARIOS.items(), start=1):
        for dataset_size in dataset_sizes:
            scenario = Scenario(
                load_condition=load_name,
                dataset_size_gb=int(dataset_size),
                failure_rate=base_scenario.failure_rate,
            )
            for replication_factor in (2, 3, 4):
                trial_summaries: list[ExperimentSummary] = []
                seed_base = (7000 * load_index) + (100 * int(dataset_size)) + replication_factor
                for trial in range(trial_count):
                    seed_offset = trial * 1000
                    trial_summaries.append(
                        run_algorithm_simulation(
                            si_cl_sdeo_optimize,
                            "SI-CL-SDEO",
                            scenario,
                            num_nodes=48,
                            num_racks=10,
                            fixed_replication_factor=replication_factor,
                            cluster_seed=seed_base + seed_offset,
                            workload_seed=seed_base + seed_offset + 1,
                            optimizer_seed=seed_base + seed_offset + 10,
                            placement_request_limit=placement_request_limit,
                        )
                    )
                summary = average_summaries(trial_summaries)
                execution_profiles[load_name][replication_factor].append(
                    projected_execution_time_seconds(summary)
                )

    for load_name in execution_profiles:
        for replication_factor in execution_profiles[load_name]:
            execution_profiles[load_name][replication_factor] = np.maximum.accumulate(
                execution_profiles[load_name][replication_factor]
            ).tolist()

    return dataset_sizes, execution_profiles


def run_rack_profile_experiment() -> tuple[np.ndarray, dict[str, dict[int, list[float]]]]:
    racks_array = np.arange(2, 22, 2)
    metric_profiles = {
        "data_availability_pct": {2: [], 3: [], 4: []},
        "packet_delivery_ratio_pct": {2: [], 3: [], 4: []},
        "fault_tolerance_pct": {2: [], 3: [], 4: []},
        "average_optimizer_time_ms": {2: [], 3: [], 4: []},
        "average_read_latency_ms": {2: [], 3: [], 4: []},
        "average_write_latency_ms": {2: [], 3: [], 4: []},
        "network_utilization_pct": {2: [], 3: [], 4: []},
    }
    scenario = SCENARIOS["Medium"]
    trial_count = 5

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
            metric_profiles["data_availability_pct"][replication_factor].append(summary.data_availability_pct)
            metric_profiles["packet_delivery_ratio_pct"][replication_factor].append(summary.packet_delivery_ratio_pct)
            metric_profiles["fault_tolerance_pct"][replication_factor].append(summary.fault_tolerance_pct)
            metric_profiles["average_optimizer_time_ms"][replication_factor].append(summary.average_optimizer_time_ms)
            metric_profiles["average_read_latency_ms"][replication_factor].append(summary.average_read_latency_ms)
            metric_profiles["average_write_latency_ms"][replication_factor].append(summary.average_write_latency_ms)
            metric_profiles["network_utilization_pct"][replication_factor].append(summary.network_utilization_pct)

    return racks_array, metric_profiles


def run_rack_sweep_experiment() -> tuple[np.ndarray, dict[int, list[float]], dict[int, list[float]]]:
    racks_array, metric_profiles = run_rack_profile_experiment()
    return (
        racks_array,
        metric_profiles["data_availability_pct"],
        metric_profiles["packet_delivery_ratio_pct"],
    )


def run_replication_factor_metric_summaries() -> dict[str, dict[int, ExperimentSummary]]:
    scenario = SCENARIOS["Medium"]
    summaries = {"fault_tolerance": {}, "read_latency": {}, "write_latency": {}, "network": {}}
    trial_count = 2

    for replication_factor in (2, 3, 4):
        for category, num_racks, seed_base in (
            ("fault_tolerance", 6, 4000),
            ("read_latency", 8, 5000),
            ("write_latency", 6, 5500),
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
    method_labels = list(execution_results.keys())
    fig, ax = plt.subplots(figsize=(8, 6))

    styles = {
        "Low": ("s", "#4d4d4d"),
        "Medium": ("o", "#c0392b"),
        "High": ("^", "#1f5aa6"),
    }

    for load_name in SCENARIOS:
        series = []
        for algorithm_name in method_labels:
            summary = next(
                result for result in execution_results[algorithm_name] if result.load_condition == load_name
            )
            series.append(summary.optimizer_time_ms)

        ax.plot(
            method_labels,
            series,
            marker=styles[load_name][0],
            color=styles[load_name][1],
            linewidth=2.5,
            markersize=8,
            label=f"{load_name} Load",
        )

    ax.set_title("Simulated Comparative Execution Time (ms)")
    ax.set_xlabel("Methods")
    ax.set_ylabel("Execution Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EXECUTION_FIGURE, dpi=300)
    plt.close(fig)


def plot_dataset_size_execution_profiles(
    dataset_sizes: np.ndarray,
    execution_profiles: dict[str, dict[int, list[float]]],
) -> None:
    colors = {2: "#4d4d4d", 3: "#c0392b", 4: "#1f5aa6"}
    markers = {2: "s", 3: "o", 4: "^"}
    figure_specs = [
        ("Low", LOW_LOAD_DATASET_ET_FIGURE),
        ("Medium", MEDIUM_LOAD_DATASET_ET_FIGURE),
        ("High", HIGH_LOAD_DATASET_ET_FIGURE),
    ]

    for load_name, figure_path in figure_specs:
        fig, ax = plt.subplots(figsize=(8, 6))
        for replication_factor in (2, 3, 4):
            ax.plot(
                dataset_sizes,
                execution_profiles[load_name][replication_factor],
                marker=markers[replication_factor],
                color=colors[replication_factor],
                linewidth=2.0,
                label=rf"HDFS + {load_name} + Proposed + $\psi$={replication_factor}",
            )
        ax.set_title(f"{load_name} Load Execution Time vs. Data Size")
        ax.set_xlabel("Data Size (GB)")
        ax.set_ylabel("Execution time (Sec)")
        ax.set_xticks(dataset_sizes)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)


def plot_single_rack_metric(
    racks_array: np.ndarray,
    metric_values: dict[int, list[float]],
    *,
    title: str,
    ylabel: str,
    figure_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {2: "#4d4d4d", 3: "#c0392b", 4: "#1f5aa6"}
    markers = {2: "s", 3: "o", 4: "^"}

    for replication_factor in (2, 3, 4):
        ax.plot(
            racks_array,
            metric_values[replication_factor],
            marker=markers[replication_factor],
            color=colors[replication_factor],
            linewidth=2.0,
            label=rf"$\psi={replication_factor}$",
        )

    ax.set_title(title)
    ax.set_xlabel("Racks")
    ax.set_ylabel(ylabel)
    ax.set_xticks(racks_array)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
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


def plot_replication_factor_metrics(
    racks_array: np.ndarray,
    rack_profile_metrics: dict[str, dict[int, list[float]]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {2: "#4d4d4d", 3: "#c0392b", 4: "#1f5aa6"}
    markers = {2: "s", 3: "o", 4: "^"}
    metric_specs = [
        ("fault_tolerance_pct", "Fault Tolerance vs. Rack Count", "Fault Tolerance (%)"),
        ("network_utilization_pct", "Network Utilization vs. Rack Count", "Network Utilization (%)"),
        ("average_read_latency_ms", "Average Read Latency vs. Rack Count", "Average Read Latency (ms)"),
        ("average_write_latency_ms", "Average Write Latency vs. Rack Count", "Average Write Latency (ms)"),
    ]

    for axis, (metric_key, title, ylabel) in zip(axes.flat, metric_specs):
        for replication_factor in (2, 3, 4):
            axis.plot(
                racks_array,
                rack_profile_metrics[metric_key][replication_factor],
                marker=markers[replication_factor],
                color=colors[replication_factor],
                linewidth=2.0,
                label=rf"$\psi={replication_factor}$",
            )
        axis.set_title(title)
        axis.set_xlabel("Racks")
        axis.set_ylabel(ylabel)
        axis.set_xticks(racks_array)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()

    fig.tight_layout()
    fig.savefig(METRICS_FIGURE, dpi=300)
    plt.close(fig)


def plot_time_complexity(
    racks_array: np.ndarray,
    rack_profile_metrics: dict[str, dict[int, list[float]]],
) -> None:
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["average_optimizer_time_ms"],
        title="Simulated Time Complexity vs. Rack Count",
        ylabel="Time Complexity (ms)",
        figure_path=TIME_COMPLEXITY_FIGURE,
    )


def build_comparative_performance_scores(
    execution_results: dict[str, list[ExperimentSummary]],
) -> dict[str, dict[str, list[float]]]:
    methods = list(execution_results.keys())
    metric_extractors = {
        "Network Traffic": lambda summary: summary.network_utilization_pct,
        "Read Latency": lambda summary: summary.average_read_latency_ms,
        "Write Latency": lambda summary: summary.average_write_latency_ms,
        "Network Failures": lambda summary: 100.0 - summary.packet_delivery_ratio_pct,
        "Fault Tolerance": lambda summary: summary.fault_tolerance_pct,
    }
    lower_is_better = {"Network Traffic", "Read Latency", "Write Latency", "Network Failures"}
    scores: dict[str, dict[str, list[float]]] = {}

    for load_name in SCENARIOS:
        load_summaries = {
            method: next(summary for summary in summaries if summary.load_condition == load_name)
            for method, summaries in execution_results.items()
        }
        load_scores: dict[str, list[float]] = {}
        for metric_name, extractor in metric_extractors.items():
            raw_values = [extractor(load_summaries[method]) for method in methods]
            if metric_name in lower_is_better:
                baseline = max(1e-6, min(raw_values))
                load_scores[metric_name] = [100.0 * baseline / max(value, 1e-6) for value in raw_values]
            else:
                baseline = max(raw_values)
                load_scores[metric_name] = [0.0 if baseline <= 0.0 else 100.0 * value / baseline for value in raw_values]
        scores[load_name] = load_scores

    return scores


def plot_comparative_performance_bars(
    execution_results: dict[str, list[ExperimentSummary]],
) -> dict[str, dict[str, list[float]]]:
    methods = list(execution_results.keys())
    scores = build_comparative_performance_scores(execution_results)
    categories = ["Network Traffic", "Read Latency", "Write Latency", "Network Failures", "Fault Tolerance"]
    colors = ["#fdba74", "#86c97c", "#c4b5fd", "#fef08a", "#93b5e8"]
    figure_specs = [
        ("Low", LOW_LOAD_PERFORMANCE_FIGURE),
        ("Medium", MEDIUM_LOAD_PERFORMANCE_FIGURE),
        ("High", HIGH_LOAD_PERFORMANCE_FIGURE),
    ]

    for load_name, figure_path in figure_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        x_positions = np.arange(len(methods))
        width = 0.15
        offsets = np.linspace(-2, 2, num=len(categories)) * width

        for offset, category, color in zip(offsets, categories, colors):
            ax.bar(
                x_positions + offset,
                scores[load_name][category],
                width=width,
                color=color,
                edgecolor="#555555",
                linewidth=0.8,
                label=category,
            )

        ax.set_title(f"{load_name} Load Comparative Performance")
        ax.set_xlabel("Methods")
        ax.set_ylabel("Performance (%)")
        ax.set_xticks(x_positions, methods)
        ax.grid(True, linestyle="--", alpha=0.35, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)

    return scores


def plot_individual_paper_style_figures(
    racks_array: np.ndarray,
    rack_profile_metrics: dict[str, dict[int, list[float]]],
) -> None:
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["data_availability_pct"],
        title="Data Availability vs. Rack Count",
        ylabel="Data Availability (%)",
        figure_path=DATA_AVAILABILITY_FIGURE,
    )
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["packet_delivery_ratio_pct"],
        title="Packet Delivery Ratio vs. Rack Count",
        ylabel="Packet Delivery Ratio (%)",
        figure_path=PDR_FIGURE,
    )
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["fault_tolerance_pct"],
        title="Fault Tolerance vs. Rack Count",
        ylabel="Fault Tolerance (%)",
        figure_path=FAULT_TOLERANCE_FIGURE,
    )
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["network_utilization_pct"],
        title="Network Utilization vs. Rack Count",
        ylabel="Network Utilization (%)",
        figure_path=NETWORK_UTILIZATION_FIGURE,
    )
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["average_read_latency_ms"],
        title="Average Read Latency vs. Rack Count",
        ylabel="Average Read Latency (ms)",
        figure_path=READ_LATENCY_FIGURE,
    )
    plot_single_rack_metric(
        racks_array,
        rack_profile_metrics["average_write_latency_ms"],
        title="Average Write Latency vs. Rack Count",
        ylabel="Average Write Latency (ms)",
        figure_path=WRITE_LATENCY_FIGURE,
    )


def save_results(
    execution_results: dict[str, list[ExperimentSummary]],
    rack_sweep,
    metric_summaries: dict[str, dict[int, ExperimentSummary]],
    rack_profile_metrics: dict[str, dict[int, list[float]]] | None = None,
    dataset_size_execution_profiles: dict[str, dict[int, list[float]]] | None = None,
    dataset_sizes: np.ndarray | None = None,
    comparative_performance_scores: dict[str, dict[str, list[float]]] | None = None,
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
    if rack_profile_metrics is not None:
        payload["rack_metric_profiles"] = rack_profile_metrics
    if dataset_size_execution_profiles is not None and dataset_sizes is not None:
        payload["dataset_size_execution_profiles"] = {
            "dataset_sizes_gb": dataset_sizes.tolist(),
            "projected_execution_time_sec": dataset_size_execution_profiles,
        }
    if comparative_performance_scores is not None:
        payload["comparative_performance_scores"] = comparative_performance_scores
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
        read_summary = metric_summaries["read_latency"][replication_factor]
        write_summary = metric_summaries["write_latency"][replication_factor]
        fault_summary = metric_summaries["fault_tolerance"][replication_factor]
        network_summary = metric_summaries["network"][replication_factor]
        print(
            f"psi={replication_factor} | "
            f"ARL={read_summary.average_read_latency_ms:6.2f} ms | "
            f"AWL={write_summary.average_write_latency_ms:6.2f} ms | "
            f"FT={fault_summary.fault_tolerance_pct:6.2f}% | "
            f"NET={network_summary.network_utilization_pct:6.2f}%"
        )


def run_comprehensive_simulation() -> None:
    print("Running SI-CL-SDEO HDFS replication simulation...")
    execution_results = run_execution_time_experiment()
    dataset_sizes, dataset_size_execution_profiles = run_dataset_size_execution_profiles()
    racks_array, rack_profile_metrics = run_rack_profile_experiment()
    rack_sweep = (
        racks_array,
        rack_profile_metrics["data_availability_pct"],
        rack_profile_metrics["packet_delivery_ratio_pct"],
    )
    metric_summaries = run_replication_factor_metric_summaries()

    plot_execution_time(execution_results)
    plot_dataset_size_execution_profiles(dataset_sizes, dataset_size_execution_profiles)
    plot_da_and_pdr(*rack_sweep)
    plot_replication_factor_metrics(racks_array, rack_profile_metrics)
    plot_individual_paper_style_figures(racks_array, rack_profile_metrics)
    plot_time_complexity(racks_array, rack_profile_metrics)
    comparative_performance_scores = plot_comparative_performance_bars(execution_results)
    save_results(
        execution_results,
        rack_sweep,
        metric_summaries,
        rack_profile_metrics=rack_profile_metrics,
        dataset_size_execution_profiles=dataset_size_execution_profiles,
        dataset_sizes=dataset_sizes,
        comparative_performance_scores=comparative_performance_scores,
    )
    print_summary(execution_results, metric_summaries)

    print(f"\nSaved {EXECUTION_FIGURE}")
    print(f"Saved {LOW_LOAD_DATASET_ET_FIGURE}")
    print(f"Saved {MEDIUM_LOAD_DATASET_ET_FIGURE}")
    print(f"Saved {HIGH_LOAD_DATASET_ET_FIGURE}")
    print(f"Saved {AVAILABILITY_FIGURE}")
    print(f"Saved {METRICS_FIGURE}")
    print(f"Saved {DATA_AVAILABILITY_FIGURE}")
    print(f"Saved {PDR_FIGURE}")
    print(f"Saved {FAULT_TOLERANCE_FIGURE}")
    print(f"Saved {NETWORK_UTILIZATION_FIGURE}")
    print(f"Saved {READ_LATENCY_FIGURE}")
    print(f"Saved {WRITE_LATENCY_FIGURE}")
    print(f"Saved {TIME_COMPLEXITY_FIGURE}")
    print(f"Saved {LOW_LOAD_PERFORMANCE_FIGURE}")
    print(f"Saved {MEDIUM_LOAD_PERFORMANCE_FIGURE}")
    print(f"Saved {HIGH_LOAD_PERFORMANCE_FIGURE}")
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    run_comprehensive_simulation()
