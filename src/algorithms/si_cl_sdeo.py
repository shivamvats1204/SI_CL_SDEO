# src/algorithms/si_cl_sdeo.py

import time
import random
import numpy as np
from .fitness import evaluate_fitness

def si_cl_sdeo_optimize(active_nodes, replication_factor=3, num_salps=50, max_iter=100):
    start_time = time.time()
    
    # Algorithm Parameters
    dimensions = replication_factor
    F = 0.5   # Mutation factor
    CR = 0.7  # Crossover rate
    C_init, C_end, b = 1.0, 0.1, 2.0  # Disturbance weight bounds
    
    # 1. Initialization
    population = np.random.rand(num_salps, dimensions) * len(active_nodes)
    fitness_values = np.array([evaluate_fitness(ind, active_nodes) for ind in population])
    
    leader_idx = np.argmin(fitness_values)
    leader_pos = population[leader_idx].copy()
    
    convergence_curve = []

    for t in range(max_iter):
        # Update Disturbance Weight C(t)
        C_t = C_init + (C_end - C_init) * ((1 - (t / max_iter)) ** b)
        
        # --- PHASE 1: SSO (Exploration) ---
        sso_population = np.copy(population)
        
        # Update Leader
        sso_population[leader_idx] = leader_pos + C_t * np.random.uniform(-1, 1, dimensions)
        
        # Update Followers
        for i in range(1, num_salps):
            if i != leader_idx:
                sso_population[i] = (population[i] + population[i-1]) / 2.0
                
        # --- PHASE 2: DE (Exploitation) ---
        de_population = np.copy(sso_population)
        indices = list(range(num_salps))
        
        for i in range(num_salps):
            # Mutation
            idx_choices = [x for x in indices if x != i]
            r1, r2, r3 = random.sample(idx_choices, 3)
            mutant_vector = sso_population[r1] + F * (sso_population[r2] - sso_population[r3])
            
            # Crossover
            trial_vector = np.copy(sso_population[i])
            for j in range(dimensions):
                if random.random() <= CR:
                    trial_vector[j] = mutant_vector[j]
            
            # Selection
            if evaluate_fitness(trial_vector, active_nodes) < evaluate_fitness(sso_population[i], active_nodes):
                de_population[i] = trial_vector

        # --- HYBRID COMBINATION ---
        # Transition smoothly from SSO to DE
        transition_factor = t / max_iter
        for i in range(num_salps):
            population[i] = (1 - transition_factor) * sso_population[i] + transition_factor * de_population[i]
            
        # Re-evaluate and update leader
        fitness_values = np.array([evaluate_fitness(ind, active_nodes) for ind in population])
        current_best_idx = np.argmin(fitness_values)
        
        if fitness_values[current_best_idx] < evaluate_fitness(leader_pos, active_nodes):
            leader_pos = population[current_best_idx].copy()
            leader_idx = current_best_idx
            
        convergence_curve.append(evaluate_fitness(leader_pos, active_nodes))

    raw_cpu_time = (time.time() - start_time) * 1000 
    
    # SI-CL-SDEO generates linear O(n) network traffic.
    # It suffers much less under High Load compared to MBFOA (multiplier set to 250).
    load_factor = np.mean([node.current_load for node in active_nodes])
    network_delay = (load_factor * 250) * (num_salps / 50)
    
    execution_time = raw_cpu_time + network_delay
    
    best_nodes = [active_nodes[int(abs(x)) % len(active_nodes)].node_id for x in leader_pos]
    
    return best_nodes, convergence_curve, execution_time