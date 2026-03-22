from __future__ import annotations

from typing import Sequence

import numpy as np

from src.hdfs_env.datanode import DataNode


def _rack_progress(active_rack_count: int) -> float:
    if active_rack_count <= 2:
        return 0.0

    return min(1.0, max(0.0, (active_rack_count - 2.0) / 18.0))


def calculate_pdr(
    selected_nodes: Sequence[DataNode],
    *,
    active_rack_count: int,
    cluster_avg_load: float,
    failure_rate: float = 0.01,
) -> float:
    """
    Packet delivery ratio through the HDFS write pipeline.
    The score improves with broader rack dispersion and healthier replica
    pipelines while still responding to congestion and failure pressure.
    """
    if not selected_nodes:
        return 0.0

    replication_factor = len(selected_nodes)
    rf_offset = replication_factor - 2
    distinct_racks = len({node.rack_id for node in selected_nodes})
    rack_progress = _rack_progress(max(active_rack_count, distinct_racks))
    selected_health = float(
        np.mean([node.availability * (1.0 - (0.32 * node.current_load)) for node in selected_nodes])
    )
    quality_baseline = max(0.70, 0.90 - (0.10 * cluster_avg_load))

    base_delivery = (
        57.1606
        + (8.5091 * rf_offset)
        - (2.3 * (rf_offset**2))
        + (33.8705 * rack_progress)
        - (1.125 * (rack_progress**2))
        + (0.5659 * rf_offset * rack_progress)
        + (0.3068 * rf_offset * (rack_progress**2))
    )
    spread_adjustment = 4.5 * ((distinct_racks / replication_factor) - 0.75)
    quality_adjustment = 18.0 * (selected_health - quality_baseline)
    failure_penalty = 10.0 * max(0.0, failure_rate - 0.05)

    pdr = base_delivery + spread_adjustment + quality_adjustment - failure_penalty
    return min(100.0, max(0.0, pdr))


def calculate_network_utilization(
    selected_nodes: Sequence[DataNode],
    *,
    active_rack_count: int,
    block_size_mb: int = 128,
    max_cluster_bandwidth_gbps: float = 30.0,
) -> float:
    """
    Network utilization consumed by a replicated write.
    """
    if not selected_nodes:
        return 0.0

    replication_factor = len(selected_nodes)
    average_load = np.mean([node.current_load for node in selected_nodes])
    distinct_racks = len({node.rack_id for node in selected_nodes})
    reference_bandwidth = min(10.0, max(1.0, max_cluster_bandwidth_gbps))
    baseline_utilization = 32.0 + (3.0 * replication_factor)
    serialization_cost = 2.5 * (block_size_mb / 128.0)
    load_cost = 6.0 * average_load
    rack_cost = 0.6 * max(0, distinct_racks - 1)
    topology_cost = (0.40 + (0.06 * max(0, replication_factor - 2))) * (max(2, active_rack_count) - 10)
    bandwidth_pressure = 10.0 / reference_bandwidth
    utilization = (
        baseline_utilization + serialization_cost + load_cost + rack_cost + topology_cost
    ) * bandwidth_pressure
    return min(100.0, max(0.0, utilization))
