from __future__ import annotations

import numpy as np

from .fitness import OptimizationResult, PlacementMetrics, build_fitness_context, evaluate_fitness
from .timing import estimate_mbfoa_execution_time_ms


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
        algorithm="MBFOA",
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


def _run_single_rf(
    active_nodes,
    replication_factor: int,
    block_size_mb: int,
    num_bacteria: int,
    max_iter: int,
    failure_rate: float,
    seed: int | None,
) -> OptimizationResult:
    if not active_nodes or len(active_nodes) < replication_factor:
        return _invalid_result(replication_factor)

    population_size = max(6, num_bacteria)
    rng = np.random.default_rng(seed)
    context = build_fitness_context(active_nodes, block_size_mb, failure_rate=failure_rate)
    search_upper_bound = max(1.0, len(active_nodes) - 1)

    dimensions = replication_factor
    population = rng.uniform(0.0, search_upper_bound, size=(population_size, dimensions))
    step_size = 0.35
    d_attract = 0.1
    w_attract = 0.2
    h_repel = 0.12
    w_repel = 10.0

    fitness_values = np.array([evaluate_fitness(candidate, context).score for candidate in population])
    best_index = int(np.argmin(fitness_values))
    global_best = population[best_index].copy()
    global_best_score = float(fitness_values[best_index])
    convergence_curve: list[float] = [global_best_score]

    for iteration in range(max_iter):
        fitness_values = np.array([evaluate_fitness(candidate, context).score for candidate in population])
        best_index = int(np.argmin(fitness_values))
        if float(fitness_values[best_index]) < global_best_score:
            global_best_score = float(fitness_values[best_index])
            global_best = population[best_index].copy()

        swarming_cost = np.zeros(population_size)
        for left_index in range(population_size):
            for right_index in range(population_size):
                difference = population[left_index] - population[right_index]
                distance_sq = np.sum(difference ** 2)
                attraction = -d_attract * np.exp(-w_attract * distance_sq)
                repulsion = h_repel * np.exp(-w_repel * distance_sq)
                swarming_cost[left_index] += attraction + repulsion

        for candidate_index in range(population_size):
            delta = rng.uniform(-1.0, 1.0, dimensions)
            direction = delta / (np.linalg.norm(delta) + 1e-8)
            candidate = population[candidate_index] + (step_size * direction)
            candidate -= swarming_cost[candidate_index] * direction
            candidate += 0.18 * (global_best - population[candidate_index])
            candidate = np.clip(candidate, 0.0, search_upper_bound)

            candidate_score = evaluate_fitness(candidate, context).score
            if candidate_score <= fitness_values[candidate_index]:
                population[candidate_index] = candidate
                fitness_values[candidate_index] = candidate_score

        if (iteration + 1) % max(1, max_iter // 5) == 0:
            worst_indices = np.argsort(fitness_values)[-max(1, population_size // 6) :]
            population[worst_indices] = rng.uniform(0.0, search_upper_bound, size=(len(worst_indices), dimensions))

        convergence_curve.append(global_best_score)

    final_evaluation = evaluate_fitness(global_best, context)
    iterations_completed = max(1, len(convergence_curve) - 1)
    execution_time_ms = estimate_mbfoa_execution_time_ms(
        active_nodes=active_nodes,
        replication_factor=replication_factor,
        block_size_mb=block_size_mb,
        population_size=population_size,
        iterations_completed=iterations_completed,
        failure_rate=failure_rate,
    )

    return OptimizationResult(
        algorithm="MBFOA",
        replication_factor=replication_factor,
        selected_node_ids=tuple(node.node_id for node in final_evaluation.selected_nodes),
        fitness=final_evaluation.score,
        metrics=final_evaluation.metrics,
        convergence_curve=tuple(convergence_curve),
        execution_time_ms=execution_time_ms,
    )


def optimize_mbfoa(
    active_nodes,
    block_size_mb: int = 128,
    replication_factors: tuple[int, ...] = (2, 3, 4),
    fixed_replication_factor: int | None = None,
    num_bacteria: int = 16,
    max_iter: int = 28,
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
            num_bacteria=num_bacteria,
            max_iter=max_iter,
            failure_rate=failure_rate,
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
