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


def run_matrix(*, preset: str, output_root: str | Path, include_ablation: bool) -> dict[str, Any]:
    """运行 selected 矩阵 preset 与 返回 manifest payload."""
    if preset not in MATCHED_PRESETS or preset not in ABLATION_PRESETS:
        raise ValueError(f"Unsupported preset '{preset}'; valid choices: {sorted(MATCHED_PRESETS)}")

    root = Path(output_root)
    matched = MATCHED_PRESETS[preset]
    ablation = ABLATION_PRESETS[preset]

    payload: dict[str, Any] = {
        "preset": preset,
        "matched": {
            "methods": list(matched.methods),
            "benchmarks": list(matched.benchmarks),
            "seeds": list(matched.seeds),
            "generations": matched.generations,
            "population_size": matched.population_size,
            "runs_root": str(root / "matched"),
        },
        "ablation": None,
    }

    run_matched_matrix(
        output_root=root / "matched",
        methods=matched.methods,
        benchmarks=matched.benchmarks,
        seeds=matched.seeds,
        generations=matched.generations,
        population_size=matched.population_size,
    )

    if include_ablation:
        run_ablation_matrix(
            output_root=root / "ablations",
            benchmarks=ablation.benchmarks,
            seeds=ablation.seeds,
            generations=ablation.generations,
            population_size=ablation.population_size,
            tau_values=ablation.tau_values,
            memory_windows=ablation.memory_windows,
        )
        payload["ablation"] = {
            "benchmarks": list(ablation.benchmarks),
            "seeds": list(ablation.seeds),
            "generations": ablation.generations,
            "population_size": ablation.population_size,
            "tau_values": list(ablation.tau_values),
            "memory_windows": list(ablation.memory_windows),
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
    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_root}_{timestamp}"

    result = run_matrix(preset=args.preset, output_root=output_path, include_ablation=not args.skip_ablation)
    print(f"wrote manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
