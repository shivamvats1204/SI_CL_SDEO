from __future__ import annotations

from typing import Sequence

import numpy as np

from src.hdfs_env.datanode import DataNode


def _load_pressure(active_nodes: Sequence[DataNode], failure_rate: float) -> float:
    average_load = float(np.mean([node.current_load for node in active_nodes]))
    return 1.0 + (0.30 * average_load) + (0.75 * failure_rate)


def _average_latency(active_nodes: Sequence[DataNode]) -> float:
    return float(np.mean([node.get_current_latency() for node in active_nodes]))


def estimate_si_execution_time_ms(
    active_nodes: Sequence[DataNode],
    replication_factor: int,
    block_size_mb: int,
    population_size: int,
    iterations_completed: int,
    failure_rate: float,
) -> float:
    active_count = len(active_nodes)
    active_rack_count = len({node.rack_id for node in active_nodes})
    block_factor = block_size_mb / 128.0
    pressure = _load_pressure(active_nodes, failure_rate)
    average_latency = _average_latency(active_nodes)
    search_cost = active_count * replication_factor * population_size * max(1, iterations_completed)
    orchestration_cost = 2.0 + (0.06 * average_latency) + (0.5 * replication_factor)
    topology_multiplier = 1.0 + (0.025 * max(0, active_rack_count - 10))
    sparse_cluster_penalty = 1.0 + (0.035 * max(0, 6 - active_rack_count))
    simulated_time_ms = orchestration_cost + (
        pressure * (((0.00030 * search_cost) + (1.35 * block_factor)) * topology_multiplier * sparse_cluster_penalty)
    )
    return float(simulated_time_ms)


def estimate_mbfoa_execution_time_ms(
    active_nodes: Sequence[DataNode],
    replication_factor: int,
    block_size_mb: int,
    population_size: int,
    iterations_completed: int,
    failure_rate: float,
) -> float:
    active_count = len(active_nodes)
    active_rack_count = len({node.rack_id for node in active_nodes})
    block_factor = block_size_mb / 128.0
    pressure = _load_pressure(active_nodes, failure_rate)
    average_latency = _average_latency(active_nodes)
    pairwise_search_cost = (active_count**2) * max(1, iterations_completed) * (population_size / 16.0)
    orchestration_cost = 3.2 + (0.12 * average_latency) + (0.8 * replication_factor)
    topology_multiplier = 1.0 + (0.025 * max(0, active_rack_count - 10))
    sparse_cluster_penalty = 1.0 + (0.035 * max(0, 6 - active_rack_count))
    simulated_time_ms = orchestration_cost + (
        pressure
        * (((0.00012 * pairwise_search_cost) + (1.75 * block_factor)) * topology_multiplier * sparse_cluster_penalty)
    )
    return float(simulated_time_ms)
