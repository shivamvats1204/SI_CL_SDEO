from __future__ import annotations

from typing import Sequence

from src.hdfs_env.datanode import DataNode


def calculate_arl(selected_nodes: Sequence[DataNode], *, active_rack_count: int) -> float:
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
    base_fetch_latency = 11.5 * latencies[0]
    selection_overhead = 28.0 + (20.0 * replication_factor)
    coordination_overhead = 8.0 * max(0, distinct_racks - 1)
    probe_overhead = 2.2 * sum(latencies[: min(2, len(latencies))])
    metadata_penalty = 40.0
    deep_replica_penalty = 26.0 * max(0, replication_factor - 3)
    # Rack fan-out reduces hotspot pressure as the cluster scales out.
    topology_relief = (26.0 + (16.0 * max(0, replication_factor - 2)) + (10.0 * max(0, replication_factor - 3))) * (
        (8.0 / max(1.0, float(active_rack_count))) - 1.0
    )
    return (
        base_fetch_latency
        + selection_overhead
        + coordination_overhead
        + probe_overhead
        + metadata_penalty
        + deep_replica_penalty
        + topology_relief
    )


def calculate_awl(
    selected_nodes: Sequence[DataNode],
    block_size_mb: int = 128,
    *,
    active_rack_count: int,
) -> float:
    """
    Average Write Latency.
    Writes traverse the replica pipeline, paying per-node latency, per-hop cost,
    and a serialization penalty proportional to block size.
    """
    if not selected_nodes:
        return float("inf")

    replication_factor = len(selected_nodes)
    pipeline_latency = 3.8 * sum(node.get_current_latency() for node in selected_nodes)
    hop_penalty = 12.0 * max(0, replication_factor - 1)
    serialization_penalty = 6.0 * (block_size_mb / 64.0)
    acknowledgment_penalty = 25.0 * max(0, replication_factor - 1)
    quorum_penalty = 8.0 * (max(0, replication_factor - 2) ** 2)
    deep_replication_penalty = 95.0 * max(0, replication_factor - 3)
    base_commit_penalty = 40.0
    topology_relief = (18.0 + (30.0 * max(0, replication_factor - 2)) + (22.0 * max(0, replication_factor - 3))) * (
        (6.0 / max(1.0, float(active_rack_count))) - 1.0
    )
    return (
        pipeline_latency
        + hop_penalty
        + serialization_penalty
        + acknowledgment_penalty
        + quorum_penalty
        + deep_replication_penalty
        + base_commit_penalty
        + topology_relief
    )
