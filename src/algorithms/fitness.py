# src/algorithms/fitness.py

import numpy as np

def evaluate_fitness(solution_vector, active_nodes):
    """
    Evaluates a proposed block placement strategy.
    solution_vector: Array of continuous values representing chosen DataNodes.
    active_nodes: List of currently alive DataNode objects in the cluster.
    """
    if not active_nodes:
        return float('inf') # Infinite penalty if cluster is dead

    W1 = 0.7  # Latency priority (alpha=7)
    W2 = 0.3  # Availability priority (beta=3)

    num_active = len(active_nodes)
    
    # Map the continuous vector values to valid discrete node indices
    selected_indices = [int(abs(x)) % num_active for x in solution_vector]
    selected_nodes = [active_nodes[i] for i in selected_indices]

    # Calculate L(i) and A(i)
    avg_latency = np.mean([node.get_current_latency() for node in selected_nodes])
    avg_availability = np.mean([node.availability for node in selected_nodes])

    # We want to minimize latency and maximize availability. 
    # Therefore, we penalize low availability by doing (1 - availability).
    fitness_score = (W1 * avg_latency) + (W2 * (1.0 - avg_availability) * 100)
    
    # Penalize if the algorithm picks the exact same node multiple times (violates HDFS rules)
    unique_nodes = len(set([node.node_id for node in selected_nodes]))
    if unique_nodes < len(solution_vector):
        fitness_score += 500.0  # Heavy penalty for rack/node redundancy violations

    return fitness_score