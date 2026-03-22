from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.hdfs_env.datanode import DataNode
from src.metrics.availability import calculate_da, calculate_fault_tolerance
from src.metrics.latency import calculate_arl, calculate_awl
from src.metrics.network import calculate_network_utilization, calculate_pdr


@dataclass(frozen=True)
class PlacementMetrics:
    average_read_latency_ms: float
    average_write_latency_ms: float
    composite_latency_ms: float
    data_availability_pct: float
    fault_tolerance_pct: float
    packet_delivery_ratio_pct: float
    network_utilization_pct: float


@dataclass(frozen=True)
class FitnessContext:
    active_nodes: tuple[DataNode, ...]
    block_size_mb: int
    cluster_avg_availability: float
    cluster_total_bandwidth_gbps: float
    latency_reference_ms: float
    weights: tuple[float, float] = (0.7, 0.3)


@dataclass(frozen=True)
class FitnessEvaluation:
    score: float
    selected_nodes: tuple[DataNode, ...]
    metrics: PlacementMetrics


@dataclass(frozen=True)
class OptimizationResult:
    algorithm: str
    replication_factor: int
    selected_node_ids: tuple[int, ...]
    fitness: float
    metrics: PlacementMetrics
    convergence_curve: tuple[float, ...]
    execution_time_ms: float


def build_fitness_context(
    active_nodes: Sequence[DataNode],
    block_size_mb: int,
    weights: tuple[float, float] = (0.7, 0.3),
) -> FitnessContext:
    nodes = tuple(active_nodes)
    if not nodes:
        raise ValueError("active_nodes must not be empty")

    cluster_avg_availability = float(np.mean([node.availability for node in nodes]))
    cluster_total_bandwidth = float(sum(node.bandwidth_gbps for node in nodes))
    average_latency = float(np.mean([node.get_current_latency() for node in nodes]))
    latency_reference = average_latency + calculate_awl(nodes[: min(3, len(nodes))], block_size_mb)

    return FitnessContext(
        active_nodes=nodes,
        block_size_mb=block_size_mb,
        cluster_avg_availability=cluster_avg_availability,
        cluster_total_bandwidth_gbps=max(10.0, cluster_total_bandwidth),
        latency_reference_ms=max(1.0, latency_reference),
        weights=weights,
    )


def decode_solution(solution_vector: Sequence[float], context: FitnessContext) -> tuple[tuple[DataNode, ...], int]:
    replication_factor = len(solution_vector)
    active_nodes = context.active_nodes
    if replication_factor == 0 or len(active_nodes) < replication_factor:
        return tuple(), replication_factor

    ordered_indices: list[int] = []
    duplicate_count = 0
    seen: set[int] = set()

    for value in solution_vector:
        index = int(abs(value)) % len(active_nodes)
        if index in seen:
            duplicate_count += 1
            continue
        ordered_indices.append(index)
        seen.add(index)

    if len(ordered_indices) < replication_factor:
        for index in np.argsort([node.get_current_latency() for node in active_nodes]):
            normalized_index = int(index)
            if normalized_index in seen:
                continue
            ordered_indices.append(normalized_index)
            seen.add(normalized_index)
            if len(ordered_indices) == replication_factor:
                break

    selected_nodes = tuple(active_nodes[index] for index in ordered_indices[:replication_factor])
    return selected_nodes, duplicate_count


def compute_placement_metrics(selected_nodes: Sequence[DataNode], context: FitnessContext) -> PlacementMetrics:
    average_read_latency = calculate_arl(selected_nodes)
    average_write_latency = calculate_awl(selected_nodes, context.block_size_mb)
    composite_latency = (0.4 * average_read_latency) + (0.6 * average_write_latency)
    data_availability = calculate_da(selected_nodes)
    fault_tolerance = calculate_fault_tolerance(selected_nodes, context.cluster_avg_availability)
    packet_delivery_ratio = calculate_pdr(selected_nodes)
    network_utilization = calculate_network_utilization(
        selected_nodes,
        block_size_mb=context.block_size_mb,
        max_cluster_bandwidth_gbps=context.cluster_total_bandwidth_gbps,
    )

    return PlacementMetrics(
        average_read_latency_ms=average_read_latency,
        average_write_latency_ms=average_write_latency,
        composite_latency_ms=composite_latency,
        data_availability_pct=data_availability,
        fault_tolerance_pct=fault_tolerance,
        packet_delivery_ratio_pct=packet_delivery_ratio,
        network_utilization_pct=network_utilization,
    )


def evaluate_fitness(solution_vector: Sequence[float], context: FitnessContext) -> FitnessEvaluation:
    selected_nodes, duplicate_count = decode_solution(solution_vector, context)
    if not selected_nodes or len(selected_nodes) != len(solution_vector):
        empty_metrics = PlacementMetrics(
            average_read_latency_ms=float("inf"),
            average_write_latency_ms=float("inf"),
            composite_latency_ms=float("inf"),
            data_availability_pct=0.0,
            fault_tolerance_pct=0.0,
            packet_delivery_ratio_pct=0.0,
            network_utilization_pct=100.0,
        )
        return FitnessEvaluation(score=float("inf"), selected_nodes=tuple(), metrics=empty_metrics)

    metrics = compute_placement_metrics(selected_nodes, context)
    latency_penalty = metrics.composite_latency_ms / context.latency_reference_ms
    availability_penalty = 1.0 - (metrics.data_availability_pct / 100.0)

    desired_distinct_racks = min(len(selected_nodes), 3)
    distinct_racks = len({node.rack_id for node in selected_nodes})
    rack_penalty = 0.14 * max(0, desired_distinct_racks - distinct_racks)

    storage_penalty = 0.0
    for node in selected_nodes:
        if not node.can_store_block(context.block_size_mb):
            storage_penalty += 0.5

    duplicate_penalty = 0.35 * duplicate_count
    utilization_penalty = 0.08 * (metrics.network_utilization_pct / 100.0)

    score = (context.weights[0] * latency_penalty) + (context.weights[1] * availability_penalty)
    score += rack_penalty + storage_penalty + duplicate_penalty + utilization_penalty

    return FitnessEvaluation(score=float(score), selected_nodes=selected_nodes, metrics=metrics)
