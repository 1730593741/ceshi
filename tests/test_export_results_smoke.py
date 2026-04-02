"""冒烟测试s 用于 result export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.export_results import collect_rows, export_results, export_split_results


def _write_summary(path: Path, method: str, benchmark: str, seed: int, hv: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "final_hv": hv,
        "final_igd_plus": 0.2,
        "final_spacing": 0.1,
        "final_spread": 0.3,
        "final_feasible_ratio": 1.0,
        "runtime_s": 0.5,
        "llm_overhead_s": 0.05,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_matched_run(
    root: Path,
    *,
    benchmark: str,
    seed: int,
    method: str,
    front: list[list[float]],
    summary_hv: float,
) -> None:
    run_dir = root / "matched" / benchmark / f"seed_{seed}" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_log = run_dir / "generation_metrics.jsonl"
    with generation_log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"generation": 1, "rank1_objectives": front}) + "\n")

    summary = {
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "final_hv": summary_hv,
        "best_hv": summary_hv,
        "hv_auc": summary_hv,
        "mean_hv": summary_hv,
        "final_igd_plus": 99.0,
        "final_spacing": 9.9,
        "final_spread": 9.9,
        "final_feasible_ratio": 1.0,
        "runtime_s": 1.0,
        "llm_overhead_s": 0.0,
        "generation_log_path": str(generation_log),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_export_results_smoke(tmp_path: Path) -> None:
    _write_summary(tmp_path / "matched" / "dwta_small" / "seed_1" / "rule_control" / "summary.json", "rule_control", "dwta_small", 1, 1.2)
    _write_summary(tmp_path / "matched" / "dwta_small" / "seed_2" / "mock_llm" / "summary.json", "mock_llm", "dwta_small", 2, 1.3)

    rows = collect_rows(tmp_path)
    assert len(rows) == 2

    outputs = export_results(runs_root=tmp_path, output_dir=tmp_path / "exports")
    for path in outputs.values():
        assert Path(path).exists()


def test_export_file_structure_and_csv_content(tmp_path: Path) -> None:
    import csv

    run_dir = tmp_path / "matched" / "dwta_small" / "seed_1" / "baseline_nsga2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "baseline_nsga2",
                "benchmark": "dwta_small",
                "seed": 1,
                "final_hv": 1.0,
                "final_igd_plus": 0.5,
                "final_spacing": 0.2,
                "final_spread": 0.4,
                "final_feasible_ratio": 0.9,
                "runtime_s": 1.2,
                "llm_overhead_s": 0.0,
            }
        ),
        encoding="utf-8",
    )

    outputs = export_results(runs_root=tmp_path, output_dir=tmp_path / "exports")

    raw_csv = Path(outputs["aggregated_runs_csv"])
    with raw_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader)
    assert first["method"] == "baseline_nsga2"
    assert first["benchmark"] == "dwta_small"

    paper_csv = Path(outputs["paper_table_method_csv"])
    assert "hv_mean" in paper_csv.read_text(encoding="utf-8")


def test_export_split_results_postprocesses_matched_metrics(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrix"
    _write_matched_run(
        matrix_root,
        benchmark="dwta_small",
        seed=1,
        method="baseline_nsga2",
        front=[[2.0, 2.0], [3.0, 1.0]],
        summary_hv=999.0,
    )
    _write_matched_run(
        matrix_root,
        benchmark="dwta_small",
        seed=1,
        method="mock_llm",
        front=[[1.0, 2.0], [2.0, 1.0]],
        summary_hv=888.0,
    )

    outputs = export_split_results(matrix_root=matrix_root, output_dir=tmp_path / "exports")
    matched_json = Path(outputs["matched"]["aggregated_runs_json"])
    payload = json.loads(matched_json.read_text(encoding="utf-8"))

    assert payload["export_mode"] == "matched_postprocessed"
    rows_by_method = {row["method"]: row for row in payload["rows"]}
    assert rows_by_method["mock_llm"]["reference_front_source"] == "empirical_matched_runs"
    assert rows_by_method["mock_llm"]["reference_front_is_comparable"] is True
    assert rows_by_method["baseline_nsga2"]["hv"] != 999.0
    assert rows_by_method["mock_llm"]["igd_plus"] != 99.0


def test_export_results_postprocesses_single_matched_benchmark(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrix"
    _write_matched_run(
        matrix_root,
        benchmark="dwta_small",
        seed=1,
        method="baseline_nsga2",
        front=[[2.0, 2.0], [3.0, 1.0]],
        summary_hv=999.0,
    )
    _write_matched_run(
        matrix_root,
        benchmark="dwta_small",
        seed=1,
        method="mock_llm",
        front=[[1.0, 2.0], [2.0, 1.0]],
        summary_hv=888.0,
    )

    benchmark_root = matrix_root / "matched" / "dwta_small"
    outputs = export_results(runs_root=benchmark_root, output_dir=tmp_path / "exports")
    payload = json.loads(Path(outputs["aggregated_runs_json"]).read_text(encoding="utf-8"))

    assert payload["export_mode"] == "matched_postprocessed_single_benchmark"
    rows_by_method = {row["method"]: row for row in payload["rows"]}
    assert rows_by_method["mock_llm"]["reference_front_source"] == "empirical_matched_runs"
    assert rows_by_method["mock_llm"]["reference_front_is_comparable"] is True
    assert rows_by_method["baseline_nsga2"]["hv"] != 999.0
