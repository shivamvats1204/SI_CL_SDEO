from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataNode:
    """Simplified HDFS DataNode model used by the placement simulation."""

    node_id: int
    rack_id: int
    total_storage_gb: float
    used_storage_gb: float
    base_latency_ms: float
    bandwidth_gbps: float
    availability: float
    current_load: float

    @property
    def free_storage_gb(self) -> float:
        return max(0.0, self.total_storage_gb - self.used_storage_gb)

    def get_current_latency(self) -> float:
        """Approximate latency under queueing and storage pressure."""
        queue_penalty = 1.0 + (1.4 * self.current_load)
        storage_penalty = 1.0 + (0.15 * (self.used_storage_gb / self.total_storage_gb))
        bandwidth_bonus = 1.0 if self.bandwidth_gbps <= 1.0 else 0.92
        return self.base_latency_ms * queue_penalty * storage_penalty * bandwidth_bonus

    def can_store_block(self, block_size_mb: int) -> bool:
        return self.free_storage_gb >= (block_size_mb / 1024.0)

    def is_alive(self, failure_rate: float, random_value: float) -> bool:
        """Failure probability rises under load and with weaker baseline uptime."""
        effective_failure_rate = failure_rate
        effective_failure_rate += (1.0 - self.availability) * 0.35
        effective_failure_rate += self.current_load * 0.08
        return random_value > min(0.99, max(0.0, effective_failure_rate))

    def apply_block_write(self, block_size_mb: int) -> None:
        block_size_gb = block_size_mb / 1024.0
        self.used_storage_gb = min(self.total_storage_gb, self.used_storage_gb + block_size_gb)
        write_pressure = 0.02 * (block_size_mb / 128.0)
        self.current_load = min(1.0, self.current_load + write_pressure)

    def cool_down(self, amount: float) -> None:
        self.current_load = max(0.02, self.current_load - amount)

    def __repr__(self) -> str:
        return (
            "DataNode("
            f"id={self.node_id}, rack={self.rack_id}, "
            f"load={self.current_load:.2f}, avail={self.availability:.3f})"
        )
