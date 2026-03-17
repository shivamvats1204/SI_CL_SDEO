import random

class DataNode:
    def __init__(self, node_id, rack_id):
        self.node_id = node_id
        self.rack_id = rack_id
        
        # Simulated Hardware & Network Metrics
        self.total_storage_gb = 1000.0  # 1 TB SSD as per the paper 
        self.used_storage_gb = random.uniform(50.0, 500.0)
        
        # Base latency (ms) with some randomness to simulate network jitter
        self.base_latency = random.uniform(5.0, 20.0) 
        
        # Availability (Uptime percentage)
        self.availability = random.uniform(0.85, 0.99)
        
        # Current load (0.0 to 1.0)
        self.current_load = random.uniform(0.1, 0.8)

    def get_current_latency(self):
        """Calculates latency penalized by current load"""
        return self.base_latency * (1.0 + self.current_load)

    def is_alive(self, failure_rate):
        """Simulates node failure based on the given failure rate (e.g., 0.01 for 1%)"""
        return random.random() > failure_rate

    def __repr__(self):
        return f"DataNode(id={self.node_id}, rack={self.rack_id}, load={self.current_load:.2f})"