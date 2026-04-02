"""汇总实验摘要 转换为 machine-readable exports 与 paper-table inputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.postprocess_matched import collect_matched_run_rows

_MATCHED_METHODS = frozenset(
    {
        "baseline_nsga2",
        "rule_control",
        "mock_llm",
        "real_llm",
        "hybrid_llm",
    }
)

EXPORT_FIELDS: tuple[str, ...] = (
    "method",
    "benchmark",
    "seed",
    "hv",
    "igd_plus",
    "spacing",
    "spread",
    "feasible_ratio",
    "runtime",
    "llm_overhead",
    "num_events",
    "wave_completion_rate",
    "event_triggered_actions",
    "reference_front_source",
    "reference_front_is_comparable",
    "summary_path",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_summary_paths(runs_root: Path) -> list[Path]:
    return sorted(path for path in runs_root.rglob("summary.json") if path.is_file())


def _to_row(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    dynamic = summary.get("dynamic_summary", {}) if isinstance(summary.get("dynamic_summary"), dict) else {}
    reference_front = summary.get("reference_front", {}) if isinstance(summary.get("reference_front"), dict) else {}
    return {
        "method": summary.get("method") or summary.get("controller_mode") or "unknown",
        "benchmark": summary.get("benchmark") or "unknown",
        "seed": int(summary.get("seed", -1)),
        "hv": float(summary.get("final_hv", 0.0)),
        "igd_plus": float(summary.get("final_igd_plus", 0.0)),
        "spacing": float(summary.get("final_spacing", 0.0)),
        "spread": float(summary.get("final_spread", 0.0)),
        "feasible_ratio": float(summary.get("final_feasible_ratio", 0.0)),
        "runtime": float(summary.get("runtime_s", 0.0)),
        "llm_overhead": float(summary.get("llm_overhead_s", 0.0)),
        "num_events": int(dynamic.get("num_events", 0)),
        "wave_completion_rate": float(dynamic.get("wave_completion_rate", 0.0)),
        "event_triggered_actions": int(dynamic.get("event_triggered_actions", 0)),
        "reference_front_source": str(reference_front.get("source", "")),
        "reference_front_is_comparable": bool(reference_front.get("is_comparable", False)),
        "summary_path": str(summary_path),
    }


def collect_rows(runs_root: str | Path) -> list[dict[str, Any]]:
    """收集 per-运行 扁平 rows 从 所有 摘要 artifacts 在 一个 运行 root."""
    root = Path(runs_root)
    rows: list[dict[str, Any]] = []
    for summary_path in _iter_summary_paths(root):
        rows.append(_to_row(_read_json(summary_path), summary_path))
    return rows

def collect_rows_within(runs_root: str | Path, relative_subdir: str | Path) -> list[dict[str, Any]]:
    """收集 runs_root 下某个子目录内的 summary rows（若目录不存在则返回空列表）。"""
    root = Path(runs_root)
    target = root / relative_subdir
    if not target.exists():
        return []
    return collect_rows(target)


def _aggregate(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(arr.size)}


def build_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 分组的 aggregates 用于 方法 与 方法+基准问题 views."""
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_method_benchmark: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_fields = (
        "hv",
        "igd_plus",
        "spacing",
        "spread",
        "feasible_ratio",
        "runtime",
        "llm_overhead",
        "num_events",
        "wave_completion_rate",
        "event_triggered_actions",
    )

    for row in rows:
        method = str(row["method"])
        benchmark = str(row["benchmark"])
        for field in metric_fields:
            value = float(row[field])
            by_method[method][field].append(value)
            by_method_benchmark[(method, benchmark)][field].append(value)

    method_agg = {
        method: {field: _aggregate(values) for field, values in field_map.items()}
        for method, field_map in by_method.items()
    }
    method_benchmark_agg = {
        f"{method}::{benchmark}": {field: _aggregate(values) for field, values in field_map.items()}
        for (method, benchmark), field_map in by_method_benchmark.items()
    }

    return {
        "num_runs": len(rows),
        "method": method_agg,
        "method_benchmark": method_benchmark_agg,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_exports(
    *,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist flat rows, grouped aggregates, and paper-table CSVs."""
    aggregates = build_aggregates(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_csv = out / "aggregated_runs.csv"
    raw_json = out / "aggregated_runs.json"
    by_method_csv = out / "paper_table_method.csv"
    by_method_benchmark_csv = out / "paper_table_method_benchmark.csv"

    _write_csv(raw_csv, rows, EXPORT_FIELDS)
    raw_payload = {"rows": rows, "aggregates": aggregates}
    if extra_payload:
        raw_payload.update(extra_payload)
    raw_json.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    method_rows: list[dict[str, Any]] = []
    for method, metrics in aggregates["method"].items():
        method_rows.append(
            {
                "method": method,
                "hv_mean": metrics["hv"]["mean"],
                "hv_std": metrics["hv"]["std"],
                "igd_plus_mean": metrics["igd_plus"]["mean"],
                "spacing_mean": metrics["spacing"]["mean"],
                "spread_mean": metrics["spread"]["mean"],
                "feasible_ratio_mean": metrics["feasible_ratio"]["mean"],
                "runtime_mean": metrics["runtime"]["mean"],
                "llm_overhead_mean": metrics["llm_overhead"]["mean"],
                "n": metrics["hv"]["n"],
            }
        )

    method_benchmark_rows: list[dict[str, Any]] = []
    for key, metrics in aggregates["method_benchmark"].items():
        method, benchmark = key.split("::", maxsplit=1)
        method_benchmark_rows.append(
            {
                "method": method,
                "benchmark": benchmark,
                "hv_mean": metrics["hv"]["mean"],
                "igd_plus_mean": metrics["igd_plus"]["mean"],
                "spacing_mean": metrics["spacing"]["mean"],
                "spread_mean": metrics["spread"]["mean"],
                "feasible_ratio_mean": metrics["feasible_ratio"]["mean"],
                "runtime_mean": metrics["runtime"]["mean"],
                "llm_overhead_mean": metrics["llm_overhead"]["mean"],
                "n": metrics["hv"]["n"],
            }
        )

    _write_csv(
        by_method_csv,
        method_rows,
        (
            "method",
            "hv_mean",
            "hv_std",
            "igd_plus_mean",
            "spacing_mean",
            "spread_mean",
            "feasible_ratio_mean",
            "runtime_mean",
            "llm_overhead_mean",
            "n",
        ),
    )
    _write_csv(
        by_method_benchmark_csv,
        method_benchmark_rows,
        (
            "method",
            "benchmark",
            "hv_mean",
            "igd_plus_mean",
            "spacing_mean",
            "spread_mean",
            "feasible_ratio_mean",
            "runtime_mean",
            "llm_overhead_mean",
            "n",
        ),
    )

    return {
        "aggregated_runs_csv": str(raw_csv),
        "aggregated_runs_json": str(raw_json),
        "paper_table_method_csv": str(by_method_csv),
        "paper_table_method_benchmark_csv": str(by_method_benchmark_csv),
    }


def export_results(*, runs_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Export 扁平 与 aggregated results 到 csv/json + paper-table input csv files."""
    root = Path(runs_root)
    if _looks_like_matched_benchmark_root(root):
        matched_rows, reference_front = collect_matched_run_rows(root)
        rows = [
            _to_row(row, Path(str(row.get("summary_path", root / "summary.json"))))
            for row in matched_rows
        ]
        return _write_exports(
            rows=rows,
            output_dir=output_dir,
            extra_payload={
                "export_mode": "matched_postprocessed_single_benchmark",
                "reference_fronts": {root.name: reference_front},
            },
        )

    rows = collect_rows(root)
    return _write_exports(rows=rows, output_dir=output_dir)


def _iter_benchmark_roots(runs_root: Path) -> list[Path]:
    """Return benchmark directories that directly contain ``seed_*`` sub-directories."""
    benchmark_roots: list[Path] = []
    for child in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        if not child.is_dir():
            continue
        if any(grandchild.is_dir() and grandchild.name.startswith("seed_") for grandchild in child.iterdir()):
            benchmark_roots.append(child)
    return benchmark_roots


def _looks_like_matched_benchmark_root(runs_root: Path) -> bool:
    """Detect a single matched benchmark directory shaped like ``seed_*/<method>/summary.json``."""
    if not runs_root.exists() or not runs_root.is_dir():
        return False

    seed_dirs = [
        child
        for child in sorted(runs_root.iterdir())
        if child.is_dir() and child.name.startswith("seed_")
    ]
    if not seed_dirs:
        return False

    method_dirs: set[str] = set()
    has_summary = False
    for seed_dir in seed_dirs:
        for method_dir in seed_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_dirs.add(method_dir.name)
            has_summary = has_summary or (method_dir / "summary.json").exists()

    return has_summary and bool(method_dirs) and method_dirs.issubset(_MATCHED_METHODS)


def export_matched_results(*, runs_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Export matched runs with benchmark-wise comparable metrics recomputed post hoc."""
    root = Path(runs_root)
    rows: list[dict[str, Any]] = []
    reference_fronts: dict[str, Any] = {}

    for benchmark_root in _iter_benchmark_roots(root):
        benchmark_rows, reference_front = collect_matched_run_rows(benchmark_root)
        reference_fronts[benchmark_root.name] = reference_front
        for row in benchmark_rows:
            summary_path = Path(str(row.get("summary_path", benchmark_root / "summary.json")))
            rows.append(_to_row(row, summary_path))

    return _write_exports(
        rows=rows,
        output_dir=output_dir,
        extra_payload={
            "export_mode": "matched_postprocessed",
            "reference_fronts": reference_fronts,
        },
    )


def export_split_results(*, matrix_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """
    按 matrix 目录结构分别导出 matched（对比）与 ablations（消融）汇总结果。

    目录约定：
    - {matrix_root}/matched
    - {matrix_root}/ablations
    """
    root = Path(matrix_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {
        "matrix_root": str(root),
        "matched": None,
        "ablations": None,
    }

    matched_rows = collect_rows_within(root, "matched")
    if matched_rows:
        matched_dir = out / "matched"
        outputs["matched"] = export_matched_results(runs_root=root / "matched", output_dir=matched_dir)

    ablation_rows = collect_rows_within(root, "ablations")
    if ablation_rows:
        ablation_dir = out / "ablations"
        outputs["ablations"] = export_results(runs_root=root / "ablations", output_dir=ablation_dir)

    outputs["has_data"] = bool(matched_rows or ablation_rows)
    return outputs



def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment runs into aggregate tables")
    parser.add_argument("--runs-root", help="Path to experiment runs root that directly contains summary.json descendants")
    parser.add_argument(
        "--matrix-root",
        help="Path to matrix output root that contains matched/ and ablations/ sub-directories",
    )
    parser.add_argument("--output-dir", default="experiments/exports")
    args = parser.parse_args()

    if bool(args.runs_root) == bool(args.matrix_root):
        raise SystemExit("Exactly one of --runs-root or --matrix-root must be provided.")

    if args.matrix_root:
        outputs = export_split_results(matrix_root=args.matrix_root, output_dir=args.output_dir)
    else:
        outputs = export_results(runs_root=args.runs_root, output_dir=args.output_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
