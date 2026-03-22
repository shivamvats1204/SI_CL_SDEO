import pytest

from src.hdfs_env.cluster import HDFSCluster


def test_cluster_validates_constructor_arguments():
    with pytest.raises(ValueError):
        HDFSCluster(num_nodes=0, num_racks=2)

    with pytest.raises(ValueError):
        HDFSCluster(num_nodes=4, num_racks=0)


def test_place_block_updates_storage_and_load():
    cluster = HDFSCluster(num_nodes=6, num_racks=2, seed=8)
    node = cluster.get_node_by_id(0)
    assert node is not None

    initial_storage = node.used_storage_gb
    initial_load = node.current_load

    cluster.place_block([0], block_size_mb=256)

    assert node.used_storage_gb > initial_storage
    assert node.current_load > initial_load
