"""CLI 入口 用于 running toy/pilot/paper 实验 matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.ablations import run_ablation_matrix
from experiments.baselines.matched_runner import run_matched_matrix
from experiments.matrix import ABLATION_PRESETS, MATCHED_PRESETS


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_seed_override(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError("seeds override must include at least one integer")
    return tuple(int(item) for item in parts)


def run_matrix(
    *,
    preset: str,
    output_root: str | Path,
    include_ablation: bool,
    max_workers: int = 1,
    seeds_override: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """运行 selected 矩阵 preset 与 返回 manifest payload."""
    if preset not in MATCHED_PRESETS or preset not in ABLATION_PRESETS:
        raise ValueError(f"Unsupported preset '{preset}'; valid choices: {sorted(MATCHED_PRESETS)}")
    if max_workers <= 0:
        raise ValueError("max_workers must be >= 1")

    root = Path(output_root)
    matched = MATCHED_PRESETS[preset]
    ablation = ABLATION_PRESETS[preset]
    matched_seeds = seeds_override or matched.seeds
    ablation_seeds = seeds_override or ablation.seeds

    payload: dict[str, Any] = {
        "preset": preset,
        "matched": {
            "methods": list(matched.methods),
            "benchmarks": list(matched.benchmarks),
            "seeds": list(matched_seeds),
            "generations": matched.generations,
            "population_size": matched.population_size,
            "max_workers": max_workers,
            "runs_root": str(root / "matched"),
        },
        "ablation": None,
    }

    run_matched_matrix(
        output_root=root / "matched",
        methods=matched.methods,
        benchmarks=matched.benchmarks,
        seeds=matched_seeds,
        generations=matched.generations,
        population_size=matched.population_size,
        max_workers=max_workers,
    )

    if include_ablation:
        run_ablation_matrix(
            output_root=root / "ablations",
            benchmarks=ablation.benchmarks,
            seeds=ablation_seeds,
            generations=ablation.generations,
            population_size=ablation.population_size,
            tau_values=ablation.tau_values,
            memory_windows=ablation.memory_windows,
            max_workers=max_workers,
        )
        payload["ablation"] = {
            "benchmarks": list(ablation.benchmarks),
            "seeds": list(ablation_seeds),
            "generations": ablation.generations,
            "population_size": ablation.population_size,
            "tau_values": list(ablation.tau_values),
            "memory_windows": list(ablation.memory_windows),
            "max_workers": max_workers,
            "runs_root": str(root / "ablations"),
        }

    manifest_path = root / "matrix_manifest.json"
    _write_json(manifest_path, payload)
    payload["manifest_path"] = str(manifest_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix presets")
    parser.add_argument("--preset", choices=list(MATCHED_PRESETS.keys()), default="toy")
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of seeds to run in parallel")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed override, e.g. 2026,2027,2028")
    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_root}_{timestamp}"

    result = run_matrix(
        preset=args.preset,
        output_root=output_path,
        include_ablation=not args.skip_ablation,
        max_workers=args.max_workers,
        seeds_override=_parse_seed_override(args.seeds),
    )
    print(f"wrote manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
