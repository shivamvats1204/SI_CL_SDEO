# src/metrics/availability.py

import numpy as np

def calculate_da(selected_nodes):
    """
    Data Availability (DA) Percentage.
    Probabilistically, data is available if AT LEAST ONE replica is online.
    Formula: P(available) = 1 - P(all replicas fail)
    """
    if not selected_nodes:
        return 0.0
        
    p_all_fail = 1.0
    for node in selected_nodes:
        # Probability that this specific node fails
        p_fail = 1.0 - node.availability
        p_all_fail *= p_fail
        
    da_percentage = (1.0 - p_all_fail) * 100.0
    return min(100.0, da_percentage)

def calculate_fault_tolerance(selected_nodes, cluster_avg_availability):
    """
    Fault Tolerance.
    Measures resilience against node failures. Higher replication provides 
    higher baseline tolerance, scaled by the actual reliability of the chosen nodes.
    """
    if not selected_nodes:
        return 0.0
        
    rf = len(selected_nodes)
    
    # Baseline theoretical fault tolerance based on RF
    # RF=2 -> ~50%, RF=3 -> ~75%, RF=4 -> ~87.5%
    theoretical_ft = (1.0 - (1.0 / (2 ** (rf - 1)))) * 100.0
    
    # Scale it by how healthy the selected nodes are compared to the cluster average
    node_health_factor = np.mean([node.availability for node in selected_nodes]) / cluster_avg_availability
    
    actual_ft = theoretical_ft * node_health_factor
    return min(100.0, actual_ft)