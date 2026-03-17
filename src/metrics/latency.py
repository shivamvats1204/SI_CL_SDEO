# src/metrics/latency.py

def calculate_arl(selected_nodes):
    """
    Average Read Latency (ARL).
    In HDFS, a client reads from the closest/fastest single replica.
    Therefore, read time is determined by the minimum latency among the chosen replicas.
    """
    if not selected_nodes:
        return float('inf')
    
    # Client reads from the fastest available replica
    return min([node.get_current_latency() for node in selected_nodes])

def calculate_awl(selected_nodes):
    """
    Average Write Latency (AWL).
    HDFS writes in a pipeline (e.g., Client -> DN1 -> DN2 -> DN3).
    Write latency is the sum of the pipeline node latencies plus network hop overhead.
    """
    if not selected_nodes:
        return float('inf')
    
    # Pipeline write time scales with the sum of latencies
    pipeline_latency = sum([node.get_current_latency() for node in selected_nodes])
    
    # Add a simulated network hop penalty (e.g., 5ms per hop)
    network_hop_penalty = 5.0 * (len(selected_nodes) - 1)
    
    return pipeline_latency + network_hop_penalty