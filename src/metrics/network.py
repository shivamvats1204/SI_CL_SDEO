# src/metrics/network.py

import numpy as np

def calculate_pdr(selected_nodes):
    """
    Packet Delivery Ratio (PDR).
    Measures the success rate of packets navigating the HDFS write pipeline.
    Calculated as the product of the successful transmission probabilities.
    """
    if not selected_nodes:
        return 0.0
        
    pdr = 1.0
    for node in selected_nodes:
        # High node load slightly reduces packet delivery success in the simulation
        success_rate = node.availability * (1.0 - (node.current_load * 0.05))
        pdr *= success_rate
        
    return pdr * 100.0

def calculate_network_utilization(selected_nodes, max_cluster_bandwidth=10.0):
    """
    Network Utilization (%).
    Writing more replicas consumes proportionally more network bandwidth.
    """
    if not selected_nodes:
        return 0.0
        
    rf = len(selected_nodes)
    base_cost_per_replica = 1.5 
    
    # Congestion multiplier based on how heavily loaded the chosen nodes are
    avg_load = np.mean([node.current_load for node in selected_nodes])
    consumed_bandwidth = base_cost_per_replica * rf * (1.0 + avg_load)
    
    utilization = (consumed_bandwidth / max_cluster_bandwidth) * 100.0
    return min(100.0, utilization)