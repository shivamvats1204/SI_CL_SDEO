from __future__ import annotations

from time import perf_counter

import numpy as np

from .fitness import OptimizationResult, PlacementMetrics, build_fitness_context, evaluate_fitness


def _empty_metrics() -> PlacementMetrics:
    return PlacementMetrics(
        average_read_latency_ms=float("inf"),
        average_write_latency_ms=float("inf"),
        composite_latency_ms=float("inf"),
        data_availability_pct=0.0,
        fault_tolerance_pct=0.0,
        packet_delivery_ratio_pct=0.0,
        network_utilization_pct=100.0,
    )


def _invalid_result(replication_factor: int) -> OptimizationResult:
    return OptimizationResult(
        algorithm="SI-CL-SDEO",
        replication_factor=replication_factor,
        selected_node_ids=tuple(),
        fitness=float("inf"),
        metrics=_empty_metrics(),
        convergence_curve=tuple(),
        execution_time_ms=0.0,
    )


def _rf_selection_penalty(
    replication_factor: int,
    candidate_replication_factors: tuple[int, ...],
    average_load: float,
    failure_rate: float,
) -> float:
    if len(candidate_replication_factors) == 1:
        return 0.0

    risk_score = (0.55 * average_load) + (6.5 * failure_rate)
    if risk_score < 0.35:
        target_replication_factor = 2
    elif risk_score < 0.85:
        target_replication_factor = 3
    else:
        target_replication_factor = 4

    storage_penalty = 0.06 * max(0, replication_factor - target_replication_factor)
    resilience_penalty = 0.18 * max(0, target_replication_factor - replication_factor)
    distance_penalty = 0.05 * abs(replication_factor - target_replication_factor)
    return storage_penalty + resilience_penalty + distance_penalty


def _heuristic_seed_candidate(active_nodes, replication_factor: int) -> np.ndarray:
    if replication_factor <= 0:
        return np.array([], dtype=float)

    node_order = sorted(
        range(len(active_nodes)),
        key=lambda index: (
            active_nodes[index].current_load,
            active_nodes[index].get_current_latency(),
            -active_nodes[index].availability,
            -active_nodes[index].bandwidth_gbps,
        ),
    )

    selected_indices: list[int] = []
    used_racks: set[int] = set()

    for index in node_order:
        rack_id = active_nodes[index].rack_id
        if rack_id in used_racks:
            continue
        selected_indices.append(index)
        used_racks.add(rack_id)
        if len(selected_indices) == replication_factor:
            break

    if len(selected_indices) < replication_factor:
        for index in node_order:
            if index in selected_indices:
                continue
            selected_indices.append(index)
            if len(selected_indices) == replication_factor:
                break

    return np.array(selected_indices, dtype=float)


def _run_single_rf(
    active_nodes,
    replication_factor: int,
    block_size_mb: int,
    num_salps: int,
    max_iter: int,
    seed: int | None,
) -> OptimizationResult:
    if not active_nodes or len(active_nodes) < replication_factor:
        return _invalid_result(replication_factor)

    population_size = max(4, num_salps)
    rng = np.random.default_rng(seed)
    context = build_fitness_context(active_nodes, block_size_mb)

    start_time = perf_counter()
    dimensions = replication_factor
    F = 0.5
    CR = 0.7
    C_init, C_end, b = 1.0, 0.1, 2.0
    search_upper_bound = max(1.0, len(active_nodes) - 1)

    population = rng.uniform(0.0, search_upper_bound, size=(population_size, dimensions))
    heuristic_candidate = _heuristic_seed_candidate(active_nodes, replication_factor)
    if len(heuristic_candidate) == dimensions:
        heuristic_jitter = rng.uniform(-0.18, 0.18, dimensions)
        population[0] = np.clip(heuristic_candidate + heuristic_jitter, 0.0, search_upper_bound)

    fitness_values = np.array([evaluate_fitness(candidate, context).score for candidate in population])
    leader_index = int(np.argmin(fitness_values))
    if leader_index != 0:
        population[[0, leader_index]] = population[[leader_index, 0]]
        fitness_values[[0, leader_index]] = fitness_values[[leader_index, 0]]

    leader_position = population[0].copy()
    best_score = float(fitness_values[0])
    convergence_curve: list[float] = [best_score]
    stagnant_iterations = 0
    early_stop_window = max(5, max_iter // 4)

    for iteration in range(max_iter):
        transition_factor = iteration / max(1, max_iter)
        disturbance_weight = C_init + (C_end - C_init) * ((1.0 - (iteration / max(1, max_iter))) ** b)

        sso_population = np.copy(population)
        leader_noise = rng.uniform(-1.0, 1.0, dimensions)
        sso_population[0] = np.clip(
            leader_position + (disturbance_weight * leader_noise * search_upper_bound * 0.2),
            0.0,
            search_upper_bound,
        )

        for follower_index in range(1, population_size):
            sso_population[follower_index] = 0.5 * (
                population[follower_index] + population[follower_index - 1]
            )

        de_population = np.copy(sso_population)
        for candidate_index in range(population_size):
            pool = [index for index in range(population_size) if index != candidate_index]
            r1, r2, r3 = rng.choice(pool, size=3, replace=False)
            mutant_vector = sso_population[r1] + F * (sso_population[r2] - sso_population[r3])
            mutant_vector = np.clip(mutant_vector, 0.0, search_upper_bound)

            crossover_mask = rng.random(dimensions) <= CR
            if not np.any(crossover_mask):
                crossover_mask[rng.integers(0, dimensions)] = True

            trial_vector = np.copy(sso_population[candidate_index])
            trial_vector[crossover_mask] = mutant_vector[crossover_mask]

            trial_fitness = evaluate_fitness(trial_vector, context).score
            incumbent_fitness = evaluate_fitness(sso_population[candidate_index], context).score
            if trial_fitness <= incumbent_fitness:
                de_population[candidate_index] = trial_vector

        population = ((1.0 - transition_factor) * sso_population) + (transition_factor * de_population)
        population = np.clip(population, 0.0, search_upper_bound)

        fitness_values = np.array([evaluate_fitness(candidate, context).score for candidate in population])
        best_index = int(np.argmin(fitness_values))
        if best_index != 0:
            population[[0, best_index]] = population[[best_index, 0]]
            fitness_values[[0, best_index]] = fitness_values[[best_index, 0]]

        if float(fitness_values[0]) <= best_score:
            best_score = float(fitness_values[0])
            leader_position = population[0].copy()
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1

        convergence_curve.append(best_score)
        if iteration >= max(4, max_iter // 3) and stagnant_iterations >= early_stop_window:
            break

    final_evaluation = evaluate_fitness(leader_position, context)
    execution_time_ms = (perf_counter() - start_time) * 1000.0

    return OptimizationResult(
        algorithm="SI-CL-SDEO",
        replication_factor=replication_factor,
        selected_node_ids=tuple(node.node_id for node in final_evaluation.selected_nodes),
        fitness=final_evaluation.score,
        metrics=final_evaluation.metrics,
        convergence_curve=tuple(convergence_curve),
        execution_time_ms=execution_time_ms,
    )


def si_cl_sdeo_optimize(
    active_nodes,
    block_size_mb: int = 128,
    replication_factors: tuple[int, ...] = (2, 3, 4),
    fixed_replication_factor: int | None = None,
    num_salps: int = 16,
    max_iter: int = 24,
    failure_rate: float = 0.01,
    seed: int | None = None,
) -> OptimizationResult:
    if fixed_replication_factor is not None:
        candidate_replication_factors = (fixed_replication_factor,)
    else:
        candidate_replication_factors = replication_factors

    if not active_nodes:
        return _invalid_result(candidate_replication_factors[0])

    average_load = float(np.mean([node.current_load for node in active_nodes]))
    best_result: OptimizationResult | None = None
    best_selection_score = float("inf")

    for offset, replication_factor in enumerate(candidate_replication_factors):
        candidate_result = _run_single_rf(
            active_nodes=active_nodes,
            replication_factor=replication_factor,
            block_size_mb=block_size_mb,
            num_salps=num_salps,
            max_iter=max_iter,
            seed=None if seed is None else seed + offset,
        )
        selection_score = candidate_result.fitness + _rf_selection_penalty(
            replication_factor,
            candidate_replication_factors,
            average_load,
            failure_rate,
        )

        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best_result = candidate_result

    return best_result if best_result is not None else _invalid_result(candidate_replication_factors[0])
