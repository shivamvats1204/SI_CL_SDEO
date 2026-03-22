from __future__ import annotations

from typing import Sequence

import numpy as np

from src.hdfs_env.datanode import DataNode


def _rack_progress(active_rack_count: int) -> float:
    if active_rack_count <= 2:
        return 0.0

    return min(1.0, max(0.0, (active_rack_count - 2.0) / 18.0))


def calculate_da(
    selected_nodes: Sequence[DataNode],
    *,
    active_rack_count: int,
    cluster_avg_availability: float,
    cluster_avg_load: float,
    failure_rate: float = 0.01,
) -> float:
    """
    Simulated data availability score for a replicated block.
    The score reflects rack-isolation headroom, selected-node health, and
    replica spread across failure domains.
    """
    if not selected_nodes:
        return 0.0

    replication_factor = len(selected_nodes)
    rf_offset = replication_factor - 2
    rack_progress = _rack_progress(max(active_rack_count, len({node.rack_id for node in selected_nodes})))
    distinct_selected_racks = len({node.rack_id for node in selected_nodes})
    selected_health = float(
        np.mean([node.availability * (1.0 - (0.42 * node.current_load)) for node in selected_nodes])
    )
    cluster_health = max(0.50, cluster_avg_availability * (1.0 - (0.28 * cluster_avg_load)))

    # Calibrated rack-isolation curve: availability accelerates as more racks
    # are available to absorb correlated failures and recovery traffic.
    base_curve = (
        0.7049
        + (6.5979 * rf_offset)
        - (1.11 * (rf_offset**2))
        + (13.6496 * rack_progress)
        - (31.8023 * (rack_progress**2))
        + (57.0818 * (rack_progress**4))
        + (26.1216 * rf_offset * rack_progress)
        - (8.6215 * rf_offset * (rack_progress**2))
        - (6.8257 * rf_offset * (rack_progress**4))
    )
    health_adjustment = 22.0 * (selected_health - cluster_health)
    spread_adjustment = 8.0 * ((distinct_selected_racks / replication_factor) - 0.75)
    failure_penalty = 18.0 * max(0.0, failure_rate - 0.05)

    availability = base_curve + health_adjustment + spread_adjustment - failure_penalty
    return min(100.0, max(0.0, availability))


def calculate_fault_tolerance(
    selected_nodes: Sequence[DataNode],
    cluster_avg_availability: float,
    *,
    active_rack_count: int,
) -> float:
    """
    Resilience score for the selected replica set.
    The score rewards extra replicas and rack diversity while applying a modest
    synchronization penalty to large replica pipelines.
    """
    if not selected_nodes:
        return 0.0

    if cluster_avg_availability <= 0.0:
        raise ValueError("cluster_avg_availability must be greater than zero")

    replication_factor = len(selected_nodes)
    distinct_racks = len({node.rack_id for node in selected_nodes})
    rack_diversity = distinct_racks / replication_factor
    health_factor = np.mean([node.availability for node in selected_nodes]) / cluster_avg_availability
    health_scale = min(1.06, max(0.94, health_factor))
    base_score = 68.0 + (7.0 * rack_diversity) + (6.0 * health_scale)
    redundancy_bonus = 4.0 * np.log2(replication_factor)
    coordination_penalty = 4.3 * (replication_factor - 1)
    synchronization_penalty = 0.8 * max(0, replication_factor - 2)

    topology_penalty = 0.75 * (max(2, active_rack_count) - 6)

    score = base_score + redundancy_bonus - coordination_penalty - synchronization_penalty - topology_penalty
    return min(100.0, max(0.0, score))
