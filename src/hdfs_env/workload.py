from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataBlock:
    block_id: int
    size_mb: int
    hotness: float

    def __repr__(self) -> str:
        return f"Block_{self.block_id}({self.size_mb}MB, hotness={self.hotness:.2f})"


class WorkloadGenerator:
    LOAD_FACTORS = {"Low": 0.85, "Medium": 1.15, "High": 1.55}

    def __init__(self, load_condition: str = "Medium", request_count: int = 6000, seed: int | None = None):
        if load_condition not in self.LOAD_FACTORS:
            raise ValueError("load_condition must be 'Low', 'Medium', or 'High'")

        self.load_condition = load_condition
        self.request_count = request_count
        self.valid_block_sizes = [64, 128, 256]
        self.network_congestion_factor = self.LOAD_FACTORS[load_condition]
        self.rng = np.random.default_rng(seed)

    def generate_dataset(self, total_size_gb: int) -> list[DataBlock]:
        total_size_mb = int(total_size_gb * 1024)
        current_size_mb = 0
        blocks: list[DataBlock] = []
        block_id_counter = 0

        while current_size_mb < total_size_mb:
            block_size = int(self.rng.choice(self.valid_block_sizes, p=[0.25, 0.5, 0.25]))
            remaining_mb = total_size_mb - current_size_mb
            if remaining_mb < 64:
                block_size = remaining_mb
            else:
                block_size = min(block_size, remaining_mb)

            hotness = float(self.rng.uniform(0.6, 1.1) * self.network_congestion_factor)
            blocks.append(DataBlock(block_id=block_id_counter, size_mb=block_size, hotness=hotness))
            current_size_mb += block_size
            block_id_counter += 1

        return blocks

    def placement_request_count(self) -> int:
        if self.load_condition == "Low":
            return 8
        if self.load_condition == "Medium":
            return 12
        return 16

    def apply_load_to_cluster(self, cluster) -> None:
        base_pressure = self.request_count / max(1, cluster.num_nodes * 1000)
        for node in cluster.get_all_nodes():
            random_spike = float(self.rng.uniform(0.02, 0.12))
            pressure = (base_pressure + random_spike) * self.network_congestion_factor
            node.current_load = min(1.0, node.current_load + pressure)
