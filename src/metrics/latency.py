from __future__ import annotations

from typing import Sequence

from src.hdfs_env.datanode import DataNode


def calculate_arl(selected_nodes: Sequence[DataNode]) -> float:
    """
    Average Read Latency.
    HDFS reads the closest or fastest available replica, but more replicas still
    add selection and coordination overhead at the NameNode and client side.
    """
    if not selected_nodes:
        return float("inf")

    latencies = sorted(node.get_current_latency() for node in selected_nodes)
    replication_factor = len(selected_nodes)
    distinct_racks = len({node.rack_id for node in selected_nodes})
    selection_overhead = 1.8 * (replication_factor - 1)
    coordination_overhead = 0.9 * max(0, distinct_racks - 1)
    probe_overhead = 0.12 * sum(latencies[: min(2, len(latencies))])
    return latencies[0] + selection_overhead + coordination_overhead + probe_overhead


def calculate_awl(selected_nodes: Sequence[DataNode], block_size_mb: int = 128) -> float:
    """
    Average Write Latency.
    Writes traverse the replica pipeline, paying per-node latency, per-hop cost,
    and a serialization penalty proportional to block size.
    """
    if not selected_nodes:
        return float("inf")

    pipeline_latency = sum(node.get_current_latency() for node in selected_nodes)
    hop_penalty = 0.0
    for left, right in zip(selected_nodes, selected_nodes[1:]):
        hop_penalty += 4.0 if left.rack_id == right.rack_id else 9.0

    serialization_penalty = 3.0 * (block_size_mb / 64.0)
    return pipeline_latency + hop_penalty + serialization_penalty
