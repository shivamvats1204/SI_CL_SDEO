import math

from src.algorithms.fitness import build_fitness_context, evaluate_fitness
from src.hdfs_env.cluster import HDFSCluster
from src.hdfs_env.datanode import DataNode
from src.metrics.availability import calculate_fault_tolerance


def test_evaluate_fitness_returns_finite_score_and_unique_nodes():
    cluster = HDFSCluster(num_nodes=12, num_racks=3, seed=123)
    active_nodes = cluster.simulate_cluster_state(failure_rate=0.01)
    context = build_fitness_context(active_nodes, block_size_mb=128)

    evaluation = evaluate_fitness([0.1, 3.9, 7.2], context)

    assert math.isfinite(evaluation.score)
    assert len(evaluation.selected_nodes) == 3
    assert len({node.node_id for node in evaluation.selected_nodes}) == 3
    assert 0.0 <= evaluation.metrics.data_availability_pct <= 100.0
    assert 0.0 <= evaluation.metrics.packet_delivery_ratio_pct <= 100.0


def test_fault_tolerance_increases_with_replication_factor_for_healthy_diverse_nodes():
    def make_node(node_id: int, rack_id: int) -> DataNode:
        return DataNode(
            node_id=node_id,
            rack_id=rack_id,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=6.0,
            bandwidth_gbps=10.0,
            availability=0.98,
            current_load=0.12,
        )

    cluster_avg_availability = 0.96
    rf2 = calculate_fault_tolerance((make_node(0, 0), make_node(1, 1)), cluster_avg_availability)
    rf3 = calculate_fault_tolerance((make_node(0, 0), make_node(1, 1), make_node(2, 2)), cluster_avg_availability)
    rf4 = calculate_fault_tolerance(
        (make_node(0, 0), make_node(1, 1), make_node(2, 2), make_node(3, 3)),
        cluster_avg_availability,
    )

    assert rf2 < rf3 < rf4
