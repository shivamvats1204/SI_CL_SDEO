from __future__ import annotations

from typing import Sequence

import numpy as np

from src.hdfs_env.datanode import DataNode


def calculate_pdr(selected_nodes: Sequence[DataNode]) -> float:
    """
    Packet delivery ratio through the HDFS write pipeline.
    The score rewards redundancy while still accounting for congestion and
    cross-rack forwarding overhead.
    """
    if not selected_nodes:
        return 0.0

    node_success_rates: list[float] = []
    for node in selected_nodes:
        node_success = node.availability * (1.0 - (node.current_load * 0.06))
        node_success_rates.append(min(0.999, max(0.01, node_success)))

    replication_factor = len(selected_nodes)
    distinct_racks = len({node.rack_id for node in selected_nodes})
    average_success = float(np.mean(node_success_rates))
    average_load = float(np.mean([node.current_load for node in selected_nodes]))
    cross_rack_hops = sum(1 for left, right in zip(selected_nodes, selected_nodes[1:]) if left.rack_id != right.rack_id)

    base_delivery = 55.0 + (18.0 * average_success)
    replication_bonus = {1: 0.0, 2: 0.0, 3: 5.5, 4: 8.0}.get(
        replication_factor,
        8.0 + (1.5 * max(0, replication_factor - 4)),
    )
    rack_bonus = 0.9 * max(0, distinct_racks - 1)
    load_penalty = 7.5 * average_load
    cross_rack_penalty = 0.8 * cross_rack_hops

    pdr = base_delivery + replication_bonus + rack_bonus - load_penalty - cross_rack_penalty
    return min(100.0, max(0.0, pdr))


def calculate_network_utilization(
    selected_nodes: Sequence[DataNode],
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
    bandwidth_pressure = 10.0 / reference_bandwidth
    utilization = (baseline_utilization + serialization_cost + load_cost + rack_cost) * bandwidth_pressure
    return min(100.0, max(0.0, utilization))
