"""运行 ablation 矩阵 实验 通过 deriving configs 从 一个 基线 方法 配置."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from main import run_experiment

_BASE_METHOD_CONFIG = "experiments/configs/rule_control.yaml"
_BENCHMARK_CONFIGS: dict[str, str] = {
    "dwta_small_smoke": "experiments/configs/dwta_small_smoke.yaml",
    "dwta_small": "experiments/configs/dwta_small.yaml",
    "dwta_medium": "experiments/configs/dwta_medium.yaml",
    "dwta_hard": "experiments/configs/dwta_hard_realworld.yaml",
    "dwta_large_static": "experiments/configs/dwta_large_static.yaml",
    "dwta_hard_realworld": "experiments/configs/dwta_hard_realworld.yaml",
}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _ablation_specs(tau_values: tuple[int, ...], memory_windows: tuple[int, ...]) -> list[tuple[str, dict[str, Any]]]:
    # NOTE: All patches must use the current unified config schema:
    #   - memory fields live under "controller" (memory_enabled, memory_window, experience_log_path)
    #   - rule-controller fields live under "controller.rule"
    # The legacy top-level "memory" key is only recognized by load_config's setdefault path
    # and will NOT override values already present in the base config.
    specs: list[tuple[str, dict[str, Any]]] = [
        (
            "no_pareto_state_deep_features",
            {"controller": {"rule": {"improvement_threshold": 1.0, "diversity_low": 0.0}}},
        ),
        (
            "no_experience_pool",
            {"controller": {"memory_enabled": False, "experience_log_path": None}},
        ),
        (
            "binary_state_machine",
            {"controller": {"rule": {"feasible_ratio_low": -1.0, "diversity_low": -1.0}}},
        ),
        (
            "four_state_machine",
            {"controller": {"rule": {"diversity_low": 0.12, "feasible_ratio_low": 0.6, "improvement_threshold": 0.0001}}},
        ),
        ("pc_pm_only", {"controller": {"rule": {"min_repair_prob": 0.0, "max_repair_prob": 0.0}}}),
        ("extended_action_space", {"controller": {"rule": {"min_repair_prob": 0.0, "max_repair_prob": 1.0}}}),
    ]
    specs.extend((f"tau_{tau}", {"controller": {"rule": {"control_interval": tau}}}) for tau in tau_values)
    specs.extend((f"memory_window_{window}", {"controller": {"memory_window": window}}) for window in memory_windows)
    return specs


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_ablation_matrix(
    *,
    output_root: str | Path,
    benchmarks: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    generations: int,
    population_size: int,
    tau_values: tuple[int, ...] = (1, 3, 5, 10),
    memory_windows: tuple[int, ...] = (5, 20, 50),
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """运行 基准问题 x 种子 x ablation 矩阵 从 该 规则-控制 base 配置."""
    base_payload = _load_yaml(_BASE_METHOD_CONFIG)
    results: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    root = Path(output_root)

    for benchmark in benchmarks:
        if benchmark not in _BENCHMARK_CONFIGS:
            raise ValueError(f"Unsupported benchmark '{benchmark}'")
        benchmark_problem = _load_yaml(_BENCHMARK_CONFIGS[benchmark])["problem"]
        benchmark_results: dict[int, dict[str, dict[str, Any]]] = {}

        for seed in seeds:
            seed_results: dict[str, dict[str, Any]] = {}
            for ablation_name, patch in _ablation_specs(tau_values, memory_windows):
                payload = _deep_merge(base_payload, patch)
                payload["problem"] = copy.deepcopy(benchmark_problem)
                payload.setdefault("solver", {})["seed"] = seed
                payload["solver"]["generations"] = generations
                payload["solver"]["population_size"] = population_size
                payload.setdefault("experiment", {})["seed"] = seed
                payload["experiment"]["name"] = f"ablation_{ablation_name}_{benchmark}_seed{seed}"
                payload["experiment"]["method"] = ablation_name
                payload["experiment"]["benchmark"] = benchmark
                payload.setdefault("logging", {})["output_dir"] = str(root / benchmark / f"seed_{seed}" / ablation_name)

                tmp_path = root / benchmark / f"seed_{seed}" / f"{ablation_name}.tmp.yaml"
                _dump_yaml(tmp_path, payload)
                try:
                    seed_results[ablation_name] = run_experiment(str(tmp_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            benchmark_results[seed] = seed_results
        results[benchmark] = benchmark_results
    return results


__all__ = ["run_ablation_matrix"]
