import math

from src.algorithms.fitness import build_fitness_context, evaluate_fitness
from src.hdfs_env.cluster import HDFSCluster
from src.hdfs_env.datanode import DataNode
from src.metrics.availability import calculate_da, calculate_fault_tolerance
from src.metrics.network import calculate_pdr


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


def test_fault_tolerance_tracks_paper_like_sync_overhead_tradeoff():
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
    rf2 = calculate_fault_tolerance(
        (make_node(0, 0), make_node(1, 1)),
        cluster_avg_availability,
        active_rack_count=6,
    )
    rf3 = calculate_fault_tolerance(
        (make_node(0, 0), make_node(1, 1), make_node(2, 2)),
        cluster_avg_availability,
        active_rack_count=6,
    )
    rf4 = calculate_fault_tolerance(
        (make_node(0, 0), make_node(1, 1), make_node(2, 2), make_node(3, 3)),
        cluster_avg_availability,
        active_rack_count=6,
    )

    assert rf2 > rf3 > rf4


def test_rack_aware_metrics_improve_with_more_available_racks():
    nodes = (
        DataNode(
            node_id=0,
            rack_id=0,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=6.0,
            bandwidth_gbps=10.0,
            availability=0.98,
            current_load=0.10,
        ),
        DataNode(
            node_id=1,
            rack_id=1,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=6.5,
            bandwidth_gbps=10.0,
            availability=0.97,
            current_load=0.12,
        ),
        DataNode(
            node_id=2,
            rack_id=2,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=7.0,
            bandwidth_gbps=10.0,
            availability=0.96,
            current_load=0.14,
        ),
    )

    da_low_racks = calculate_da(
        nodes,
        active_rack_count=3,
        cluster_avg_availability=0.96,
        cluster_avg_load=0.14,
        failure_rate=0.05,
    )
    da_high_racks = calculate_da(
        nodes,
        active_rack_count=18,
        cluster_avg_availability=0.96,
        cluster_avg_load=0.14,
        failure_rate=0.05,
    )
    pdr_low_racks = calculate_pdr(
        nodes,
        active_rack_count=3,
        cluster_avg_load=0.14,
        failure_rate=0.05,
    )
    pdr_high_racks = calculate_pdr(
        nodes,
        active_rack_count=18,
        cluster_avg_load=0.14,
        failure_rate=0.05,
    )

    assert da_low_racks < 15.0
    assert da_high_racks > 35.0
    assert pdr_low_racks < 70.0
    assert pdr_high_racks > 90.0


def test_fault_tolerance_drops_and_latency_improves_with_more_racks():
    nodes = (
        DataNode(
            node_id=0,
            rack_id=0,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=6.0,
            bandwidth_gbps=10.0,
            availability=0.98,
            current_load=0.18,
        ),
        DataNode(
            node_id=1,
            rack_id=1,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=6.8,
            bandwidth_gbps=10.0,
            availability=0.97,
            current_load=0.19,
        ),
        DataNode(
            node_id=2,
            rack_id=2,
            total_storage_gb=1000.0,
            used_storage_gb=200.0,
            base_latency_ms=7.4,
            bandwidth_gbps=10.0,
            availability=0.96,
            current_load=0.20,
        ),
    )

    from src.metrics.latency import calculate_arl, calculate_awl
    from src.metrics.network import calculate_network_utilization

    ft_low_racks = calculate_fault_tolerance(nodes, 0.96, active_rack_count=6)
    ft_high_racks = calculate_fault_tolerance(nodes, 0.96, active_rack_count=20)
    arl_low_racks = calculate_arl(nodes, active_rack_count=2)
    arl_high_racks = calculate_arl(nodes, active_rack_count=20)
    awl_low_racks = calculate_awl(nodes, 128, active_rack_count=2)
    awl_high_racks = calculate_awl(nodes, 128, active_rack_count=20)
    net_low_racks = calculate_network_utilization(nodes, active_rack_count=2, block_size_mb=128)
    net_high_racks = calculate_network_utilization(nodes, active_rack_count=20, block_size_mb=128)

    assert ft_low_racks > ft_high_racks
    assert arl_low_racks > arl_high_racks
    assert awl_low_racks > awl_high_racks
    assert net_low_racks < net_high_racks
