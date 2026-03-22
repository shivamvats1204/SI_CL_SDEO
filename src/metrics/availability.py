from __future__ import annotations

from typing import Sequence

import numpy as np

from src.hdfs_env.datanode import DataNode


def calculate_da(selected_nodes: Sequence[DataNode]) -> float:
    """
    Data availability for a replicated block.
    The block remains available if at least one replica remains reachable.
    """
    if not selected_nodes:
        return 0.0

    probability_all_replicas_unavailable = 1.0
    for node in selected_nodes:
        effective_uptime = node.availability * (1.0 - (node.current_load * 0.08))
        effective_uptime = min(0.999, max(0.01, effective_uptime))
        probability_all_replicas_unavailable *= (1.0 - effective_uptime)

    return min(100.0, max(0.0, (1.0 - probability_all_replicas_unavailable) * 100.0))


def calculate_fault_tolerance(selected_nodes: Sequence[DataNode], cluster_avg_availability: float) -> float:
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
    base_score = 38.0 + (11.0 * replication_factor)
    redundancy_bonus = 5.5 * np.log2(replication_factor)
    rack_bonus = 8.0 * rack_diversity
    health_scale = min(1.08, max(0.92, health_factor))
    synchronization_penalty = 1.5 * max(0, replication_factor - 2)

    score = ((base_score + redundancy_bonus + rack_bonus) * health_scale) - synchronization_penalty
    return min(100.0, max(0.0, score))
