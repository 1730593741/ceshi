"""冒烟测试 用于 matched post-processing aggregation script."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.postprocess_matched import summarize_matched_runs


def _write_run(root: Path, seed: int, method: str, front: list[list[float]], hv: float) -> None:
    run_dir = root / f"seed_{seed}" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_log = run_dir / "generation_metrics.jsonl"
    with generation_log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"generation": 1, "rank1_objectives": front}) + "\n")

    summary = {
        "final_hv": hv,
        "best_hv": hv,
        "hv_auc": hv,
        "mean_hv": hv,
        "final_feasible_ratio": 1.0,
        "generation_log_path": str(generation_log),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f)


def test_summarize_matched_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, 1, "baseline_nsga2", [[2, 2], [3, 1]], 1.0)
    _write_run(tmp_path, 1, "rule_control", [[1.5, 2.0], [2.5, 1.2]], 1.1)
    _write_run(tmp_path, 1, "mock_llm", [[1.4, 2.1], [2.0, 1.4]], 1.2)

    _write_run(tmp_path, 2, "baseline_nsga2", [[2.1, 2.1], [2.9, 1.1]], 0.9)
    _write_run(tmp_path, 2, "rule_control", [[1.6, 1.9], [2.4, 1.1]], 1.0)
    _write_run(tmp_path, 2, "mock_llm", [[1.3, 2.0], [2.1, 1.3]], 1.3)

    payload = summarize_matched_runs(tmp_path)
    assert payload["reference_front"]["source"] == "empirical_matched_runs"
    assert payload["grouped"]["baseline_nsga2"]["final_igd"]["n"] == 2
    assert payload["grouped"]["mock_llm"]["best_hv"]["mean"] > 0.0
