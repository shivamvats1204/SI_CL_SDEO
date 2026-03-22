import json

import main
import numpy as np


def _summary(**overrides):
    payload = {
        "algorithm": "SI-CL-SDEO",
        "load_condition": "Medium",
        "dataset_size_gb": 500,
        "failure_rate": 0.05,
        "placements_completed": 6,
        "wall_clock_time_ms": 1200.0,
        "optimizer_time_ms": 1180.0,
        "average_optimizer_time_ms": 196.67,
        "average_replication_factor": 3.0,
        "average_read_latency_ms": 15.5,
        "average_write_latency_ms": 48.5,
        "data_availability_pct": 99.9,
        "fault_tolerance_pct": 86.2,
        "packet_delivery_ratio_pct": 74.4,
        "network_utilization_pct": 46.8,
    }
    payload.update(overrides)
    return main.ExperimentSummary(**payload)


def test_rack_sweep_uses_simulated_pdr(monkeypatch):
    def fake_run_algorithm_simulation(
        optimizer,
        algorithm_name,
        scenario,
        *,
        num_nodes=50,
        num_racks=10,
        fixed_replication_factor=None,
        cluster_seed=7,
        workload_seed=19,
        optimizer_seed=101,
        placement_request_limit=None,
    ):
        replication_factor = fixed_replication_factor or 0
        return _summary(
            average_replication_factor=float(replication_factor),
            data_availability_pct=float(num_racks + replication_factor),
            packet_delivery_ratio_pct=float((num_racks * 10) + replication_factor),
        )

    monkeypatch.setattr(main, "run_algorithm_simulation", fake_run_algorithm_simulation)

    racks, da_results, pdr_results = main.run_rack_sweep_experiment()

    assert racks.tolist() == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert da_results[3] == [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0]
    assert pdr_results[3] == [23.0, 43.0, 63.0, 83.0, 103.0, 123.0, 143.0, 163.0, 183.0, 203.0]


def test_select_blocks_for_simulation_spans_dataset():
    dataset = [f"block-{index}" for index in range(10)]

    sampled = main.select_blocks_for_simulation(dataset, 4)

    assert sampled == ["block-0", "block-3", "block-6", "block-9"]


def test_default_sampled_block_count_scales_with_dataset():
    assert main.default_sampled_block_count("Low", 800) == 8
    assert main.default_sampled_block_count("Medium", 4000) == 11
    assert main.default_sampled_block_count("High", 8000) == 14


def test_save_results_persists_only_simulation_fields(tmp_path, monkeypatch):
    results_path = tmp_path / "simulation_results.json"
    monkeypatch.setattr(main, "RESULTS_PATH", results_path)

    execution_results = {"SI-CL-SDEO": [_summary(load_condition="Low", optimizer_time_ms=321.0)]}
    rack_sweep = (np.array([2, 4]), {2: [99.1, 99.2]}, {2: [68.4, 69.2]})
    metric_summaries = {
        "fault_tolerance": {2: _summary(fault_tolerance_pct=82.5)},
        "read_latency": {2: _summary(average_read_latency_ms=12.3)},
        "write_latency": {2: _summary(average_write_latency_ms=34.5)},
        "network": {2: _summary(network_utilization_pct=44.4)},
    }

    main.save_results(execution_results, rack_sweep, metric_summaries)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    execution_summary = payload["execution_time_experiment"]["SI-CL-SDEO"][0]
    metric_summary = payload["replication_factor_metrics"]["read_latency"]["2"]

    assert execution_summary["optimizer_time_ms"] == 321.0
    assert execution_summary["dataset_block_count"] == 0
    assert execution_summary["sampled_block_count"] == 0
    assert "benchmark_execution_time_ms" not in execution_summary
    assert "benchmark_read_latency_ms" not in metric_summary
    assert payload["rack_sweep"]["packet_delivery_ratio_pct"]["2"] == [68.4, 69.2]


def test_print_summary_uses_simulation_values(capsys):
    execution_results = {"SI-CL-SDEO": [_summary(load_condition="Low", optimizer_time_ms=321.0)]}
    metric_summaries = {
        "fault_tolerance": {2: _summary(fault_tolerance_pct=82.5), 3: _summary(fault_tolerance_pct=86.4), 4: _summary(fault_tolerance_pct=88.1)},
        "read_latency": {
            2: _summary(average_read_latency_ms=12.3),
            3: _summary(average_read_latency_ms=15.6),
            4: _summary(average_read_latency_ms=18.9),
        },
        "write_latency": {
            2: _summary(average_write_latency_ms=34.5),
            3: _summary(average_write_latency_ms=48.7),
            4: _summary(average_write_latency_ms=62.1),
        },
        "network": {
            2: _summary(network_utilization_pct=44.4),
            3: _summary(network_utilization_pct=47.7),
            4: _summary(network_utilization_pct=51.2),
        },
    }

    main.print_summary(execution_results, metric_summaries)
    output = capsys.readouterr().out

    assert "ET= 321.00 ms" in output
    assert "ARL= 12.30 ms" in output
    assert "AWL= 34.50 ms" in output
    assert "FT= 82.50%" in output
    assert "NET= 44.40%" in output
