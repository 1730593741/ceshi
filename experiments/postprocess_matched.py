"""对 matched 实验 做后处理 转换为 可复现的 paper-style 摘要 tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import igd, igd_plus, spacing, spread
from eval.reference_front import (
    build_empirical_reference_front,
    read_final_front_from_generation_log,
    read_generation_fronts_from_generation_log,
)
from sensing.hypervolume import compute_hypervolume

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


def _build_hypervolume_reference_point(points: list[tuple[float, ...]]) -> tuple[float, ...]:
    """Construct one shared HV reference point for all matched runs of a benchmark."""
    if not points:
        return (1.0, 1.0)
    matrix = np.asarray(points, dtype=float)
    maxima = matrix.max(axis=0)
    padding = np.maximum(np.abs(maxima) * 0.1, 1e-6)
    return tuple((maxima + padding).tolist())


def collect_matched_run_rows(runs_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect per-run rows with matched comparable metrics recomputed post hoc."""
    runs_by_method = _collect_run_summaries(runs_root)
    method_to_logs: dict[str, list[Path]] = {}
    for method, summaries in runs_by_method.items():
        method_to_logs[method] = [Path(s["generation_log_path"]) for s in summaries if s.get("generation_log_path")]

    reference = build_empirical_reference_front(method_to_logs)
    hv_reference_point = _build_hypervolume_reference_point(reference.points)
    reference_payload = {
        "source": reference.source,
        "is_comparable": True,
        "details": {
            **reference.details,
            "hv_reference_point": list(hv_reference_point),
        },
        "num_points": len(reference.points),
    }

    rows: list[dict[str, Any]] = []
    for method, summaries in runs_by_method.items():
        for summary in summaries:
            generation_log_path = Path(summary["generation_log_path"])
            generation_fronts = read_generation_fronts_from_generation_log(generation_log_path)
            final_front = generation_fronts[-1] if generation_fronts else read_final_front_from_generation_log(generation_log_path)
            hv_values = [compute_hypervolume(front, hv_reference_point) for front in generation_fronts]

            row = dict(summary)
            row["method"] = summary.get("method", method)
            row["benchmark"] = summary.get("benchmark", runs_root.name)
            row["summary_path"] = str(generation_log_path.parent / "summary.json")
            if hv_values:
                comparable_mean_hv = float(np.mean(np.asarray(hv_values, dtype=float)))
                row["final_hv"] = float(hv_values[-1])
                row["best_hv"] = float(max(hv_values))
                row["hv_auc"] = comparable_mean_hv
                row["mean_hv"] = comparable_mean_hv
            row["final_igd"] = igd(final_front, reference.points)
            row["final_igd_plus"] = igd_plus(final_front, reference.points)
            row["final_spacing"] = spacing(final_front)
            row["final_spread"] = spread(final_front, reference.points) if final_front and reference.points else 0.0
            row["reference_front"] = reference_payload
            rows.append(row)

    return rows, reference_payload


def summarize_matched_runs(runs_root: Path) -> dict[str, Any]:
    rows, reference_payload = collect_matched_run_rows(runs_root)
    per_method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_method_rows[str(row["method"])].append(row)

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
        "reference_front": reference_payload,
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
