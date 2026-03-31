"""对 matched 实验 做后处理 转换为 可复现的 paper-style 摘要 tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import igd, igd_plus, spacing, spread
from eval.reference_front import build_empirical_reference_front, read_final_front_from_generation_log

_METHODS = ("baseline_nsga2", "rule_control", "mock_llm", "real_llm", "hybrid_llm")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_run_summaries(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed_dir in sorted(runs_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        for method in _METHODS:
            summary_path = seed_dir / method / "summary.json"
            if summary_path.exists():
                by_method[method].append(_read_json(summary_path))
    return by_method


def _aggregate(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(arr.size)}


def summarize_matched_runs(runs_root: Path) -> dict[str, Any]:
    runs_by_method = _collect_run_summaries(runs_root)
    method_to_logs: dict[str, list[Path]] = {}
    for method, summaries in runs_by_method.items():
        method_to_logs[method] = [Path(s["generation_log_path"]) for s in summaries if s.get("generation_log_path")]

    reference = build_empirical_reference_front(method_to_logs)

    per_method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method, summaries in runs_by_method.items():
        for summary in summaries:
            generation_log_path = Path(summary["generation_log_path"])
            final_front = read_final_front_from_generation_log(generation_log_path)
            row = dict(summary)
            row["final_igd"] = igd(final_front, reference.points)
            row["final_igd_plus"] = igd_plus(final_front, reference.points)
            row["final_spacing"] = spacing(final_front)
            row["final_spread"] = spread(final_front, reference.points) if final_front and reference.points else 0.0
            row["reference_front"] = {
                "source": reference.source,
                "details": reference.details,
                "num_points": len(reference.points),
            }
            per_method_rows[method].append(row)

    metrics = [
        "final_hv",
        "best_hv",
        "hv_auc",
        "mean_hv",
        "final_feasible_ratio",
        "final_igd",
        "final_igd_plus",
        "final_spacing",
        "final_spread",
    ]

    grouped: dict[str, Any] = {}
    for method, rows in per_method_rows.items():
        grouped[method] = {
            metric: _aggregate([float(r[metric]) for r in rows if metric in r])
            for metric in metrics
        }

    return {
        "runs_root": str(runs_root),
        "methods": list(_METHODS),
        "reference_front": {
            "source": reference.source,
            "details": reference.details,
            "num_points": len(reference.points),
        },
        "grouped": grouped,
        "num_runs": {m: len(per_method_rows[m]) for m in _METHODS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched experiment metrics")
    parser.add_argument("--runs-root", required=True, help="Root directory containing seed_*/<method>/summary.json")
    parser.add_argument("--output", default=None, help="Output JSON path; default <runs-root>/paper_summary.json")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    payload = summarize_matched_runs(runs_root)
    out = Path(args.output) if args.output else runs_root / "paper_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
