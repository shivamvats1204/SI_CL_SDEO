import random

class DataBlock:
    def __init__(self, block_id, size_mb):
        self.block_id = block_id
        self.size_mb = size_mb
        
    def __repr__(self):
        return f"Block_{self.block_id}({self.size_mb}MB)"

class WorkloadGenerator:
    def __init__(self, load_condition="Medium"):
        """
        Initializes the generator based on the load condition.
        load_condition: 'Low', 'Medium', or 'High'
        """
        self.load_condition = load_condition
        self.valid_block_sizes = [64, 128, 256] # Standard HDFS block sizes from the paper
        
        # Load conditions dictate the intensity/concurrency of the environment
        if self.load_condition == "Low":
            self.network_congestion_factor = 1.0
        elif self.load_condition == "Medium":
            self.network_congestion_factor = 1.5
        elif self.load_condition == "High":
            self.network_congestion_factor = 2.5
        else:
            raise ValueError("load_condition must be 'Low', 'Medium', or 'High'")

    def generate_dataset(self, total_size_gb):
        """
        Generates a list of DataBlocks that sum up to the requested total dataset size.
        """
        total_size_mb = total_size_gb * 1024
        current_size_mb = 0
        blocks = []
        block_id_counter = 0
        
        while current_size_mb < total_size_mb:
            # Randomly pick a block size: 64MB, 128MB, or 256MB
            block_size = random.choice(self.valid_block_sizes)
            
            # Ensure we don't wildly overshoot the target size
            if current_size_mb + block_size > total_size_mb and (total_size_mb - current_size_mb) < 64:
                block_size = total_size_mb - current_size_mb
                
            blocks.append(DataBlock(block_id_counter, block_size))
            current_size_mb += block_size
            block_id_counter += 1
            
        return blocks

    def apply_load_to_cluster(self, cluster):
        """
        Artificially spikes the current load and latency on the cluster's DataNodes
        based on whether the workload is Low, Medium, or High.
        """
        for node in cluster.get_all_nodes():
            # Add artificial stress to the nodes to simulate high traffic
            stress = random.uniform(0.0, 0.2) * self.network_congestion_factor
            node.current_load = min(1.0, node.current_load + stress)