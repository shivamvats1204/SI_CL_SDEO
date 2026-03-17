from .datanode import DataNode

class HDFSCluster:
    def __init__(self, num_nodes, num_racks):
        self.num_nodes = num_nodes
        self.num_racks = num_racks
        self.datanodes = []
        self._initialize_cluster()

    def _initialize_cluster(self):
        """Distributes DataNodes evenly across the specified number of racks."""
        for i in range(self.num_nodes):
            rack_id = i % self.num_racks
            node = DataNode(node_id=i, rack_id=rack_id)
            self.datanodes.append(node)

    def get_all_nodes(self):
        return self.datanodes

    def get_node_by_id(self, node_id):
        if 0 <= node_id < self.num_nodes:
            return self.datanodes[node_id]
        return None

    def simulate_cluster_state(self, failure_rate=0.01):
        """Returns only the nodes that are currently 'alive' based on failure rates."""
        active_nodes = [node for node in self.datanodes if node.is_alive(failure_rate)]
        return active_nodes

    def cluster_summary(self):
        print(f"--- Cluster Topology ---")
        print(f"Total Nodes: {self.num_nodes}")
        print(f"Total Racks: {self.num_racks}")
        print(f"Average Load: {sum(n.current_load for n in self.datanodes)/self.num_nodes:.2f}")