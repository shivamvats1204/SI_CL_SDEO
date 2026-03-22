import math

from src.algorithms.baselines import optimize_mbfoa
from src.algorithms.si_cl_sdeo import si_cl_sdeo_optimize
from src.hdfs_env.cluster import HDFSCluster


def test_si_cl_sdeo_selects_valid_replication_factor_and_nodes():
    cluster = HDFSCluster(num_nodes=16, num_racks=4, seed=77)
    active_nodes = cluster.simulate_cluster_state(failure_rate=0.02)

    result = si_cl_sdeo_optimize(active_nodes, block_size_mb=128, seed=11)

    assert result.replication_factor in {2, 3, 4}
    assert len(result.selected_node_ids) == result.replication_factor
    assert len(set(result.selected_node_ids)) == result.replication_factor
    assert math.isfinite(result.fitness)


def test_mbfoa_supports_fixed_replication_factor():
    cluster = HDFSCluster(num_nodes=16, num_racks=4, seed=91)
    active_nodes = cluster.simulate_cluster_state(failure_rate=0.02)

    result = optimize_mbfoa(
        active_nodes,
        block_size_mb=256,
        fixed_replication_factor=3,
        seed=5,
    )

    assert result.replication_factor == 3
    assert len(result.selected_node_ids) == 3
    assert math.isfinite(result.fitness)


def test_optimizers_handle_empty_active_node_lists():
    si_result = si_cl_sdeo_optimize([], block_size_mb=128, fixed_replication_factor=2)
    mbfoa_result = optimize_mbfoa([], block_size_mb=128, fixed_replication_factor=2)

    assert si_result.selected_node_ids == tuple()
    assert math.isinf(si_result.fitness)
    assert mbfoa_result.selected_node_ids == tuple()
    assert math.isinf(mbfoa_result.fitness)
