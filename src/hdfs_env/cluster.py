from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

import numpy as np

from .datanode import DataNode


@dataclass
class HDFSCluster:
    num_nodes: int
    num_racks: int
    seed: int | None = None
    datanodes: list[DataNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be greater than zero")
        if self.num_racks <= 0:
            raise ValueError("num_racks must be greater than zero")

        self.rng = np.random.default_rng(self.seed)
        self._initialize_cluster()

    def _initialize_cluster(self) -> None:
        self.datanodes.clear()
        for node_id in range(self.num_nodes):
            rack_id = node_id % self.num_racks
            total_storage_gb = 1000.0
            used_storage_gb = float(self.rng.uniform(80.0, 420.0))
            base_latency_ms = float(self.rng.uniform(4.0, 14.0) + (rack_id * 0.08))
            bandwidth_gbps = float(self.rng.choice([1.0, 10.0], p=[0.35, 0.65]))
            availability = float(self.rng.uniform(0.92, 0.995))
            current_load = float(self.rng.uniform(0.05, 0.22))
            self.datanodes.append(
                DataNode(
                    node_id=node_id,
                    rack_id=rack_id,
                    total_storage_gb=total_storage_gb,
                    used_storage_gb=used_storage_gb,
                    base_latency_ms=base_latency_ms,
                    bandwidth_gbps=bandwidth_gbps,
                    availability=availability,
                    current_load=current_load,
                )
            )

    def reset_loads(self) -> None:
        for node in self.datanodes:
            node.current_load = float(self.rng.uniform(0.05, 0.22))

    def cool_down(self, amount: float = 0.015) -> None:
        for node in self.datanodes:
            node.cool_down(amount)

    def get_all_nodes(self) -> list[DataNode]:
        return self.datanodes

    def get_node_by_id(self, node_id: int) -> DataNode | None:
        if 0 <= node_id < self.num_nodes:
            return self.datanodes[node_id]
        return None

    def total_bandwidth_gbps(self) -> float:
        return sum(node.bandwidth_gbps for node in self.datanodes)

    def average_availability(self, nodes: Iterable[DataNode] | None = None) -> float:
        candidates = list(nodes) if nodes is not None else self.datanodes
        return mean(node.availability for node in candidates) if candidates else 0.0

    def average_latency(self, nodes: Iterable[DataNode] | None = None) -> float:
        candidates = list(nodes) if nodes is not None else self.datanodes
        return mean(node.get_current_latency() for node in candidates) if candidates else float("inf")

    def simulate_cluster_state(self, failure_rate: float = 0.01) -> list[DataNode]:
        active_nodes: list[DataNode] = []
        for node in self.datanodes:
            random_value = float(self.rng.random())
            if node.is_alive(failure_rate, random_value):
                active_nodes.append(node)
        return active_nodes

    def place_block(self, node_ids: Iterable[int], block_size_mb: int) -> None:
        for node_id in node_ids:
            node = self.get_node_by_id(node_id)
            if node is None:
                continue
            node.apply_block_write(block_size_mb)

    def cluster_summary(self) -> dict[str, float]:
        return {
            "total_nodes": float(self.num_nodes),
            "total_racks": float(self.num_racks),
            "average_load": float(mean(node.current_load for node in self.datanodes)),
            "average_latency_ms": float(self.average_latency()),
            "average_availability": float(self.average_availability()),
            "total_bandwidth_gbps": float(self.total_bandwidth_gbps()),
        }
