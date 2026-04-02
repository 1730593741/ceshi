"""Helpers 到 运行 matched 实验，并带有 aligned seeds/基准问题/优化器 settings."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from main import run_experiment

_MATCHED_CONFIGS: dict[str, str] = {
    "baseline_nsga2": "experiments/configs/baseline_nsga2.yaml",
    "rule_control": "experiments/configs/rule_control.yaml",
    "mock_llm": "experiments/configs/mock_llm.yaml",
    "real_llm": "experiments/configs/real_llm.yaml",
    "hybrid_llm": "experiments/configs/hybrid_llm.yaml",
}

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _resolve_methods(methods: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = tuple(methods) if methods else tuple(_MATCHED_CONFIGS.keys())
    unsupported = [method for method in selected if method not in _MATCHED_CONFIGS]
    if unsupported:
        raise ValueError(f"Unsupported matched methods: {unsupported}")
    return selected


def _resolve_max_workers(max_workers: int, num_jobs: int) -> int:
    if max_workers <= 0:
        raise ValueError("max_workers must be >= 1")
    return min(max_workers, max(1, num_jobs))


def _build_matched_payload(
    *,
    method: str,
    benchmark: str,
    seed: int,
    generations: int,
    population_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(_load_yaml(_MATCHED_CONFIGS[method]))
    benchmark_payload = _load_yaml(_BENCHMARK_CONFIGS[benchmark])

    payload["problem"] = copy.deepcopy(benchmark_payload["problem"])

    # Normalize solver key: benchmark configs may use legacy "optimizer" key.
    # The caller-supplied generations/population_size always win for matched experiments
    # (ensuring all methods run under identical optimizer settings).
    # However, benchmark-specific solver knobs (eta_c, eta_m, local_search_prob, etc.)
    # are intentionally NOT inherited from the benchmark config so that matched
    # comparisons use only the method config's defaults — keeping comparisons fair.
    if "solver" not in payload and "optimizer" in payload:
        payload["solver"] = payload.pop("optimizer")
    payload.setdefault("solver", {})["seed"] = seed
    payload["solver"]["generations"] = generations
    payload["solver"]["population_size"] = population_size

    payload.setdefault("experiment", {})["seed"] = seed
    payload["experiment"]["name"] = f"matched_{method}_{benchmark}_seed{seed}"
    payload["experiment"]["method"] = method
    payload["experiment"]["benchmark"] = benchmark

    payload.setdefault("logging", {})["output_dir"] = str(output_dir)
    return payload


def run_matched_experiments(
    *,
    output_root: str | Path,
    seed: int,
    generations: int,
    population_size: int = 24,
    benchmark: str = "dwta_small",
    methods: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """运行 matched 方法 comparisons 在 一个 单个 种子 + 基准问题 setting."""
    if benchmark not in _BENCHMARK_CONFIGS:
        raise ValueError(f"Unsupported benchmark '{benchmark}'")

    selected_methods = _resolve_methods(methods)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    for method in selected_methods:
        log_dir = output_root_path / method
        matched_payload = _build_matched_payload(
            method=method,
            benchmark=benchmark,
            seed=seed,
            generations=generations,
            population_size=population_size,
            output_dir=log_dir,
        )
        config_tmp_path = output_root_path / f"{method}.matched.tmp.yaml"
        _dump_yaml(config_tmp_path, matched_payload)
        try:
            results[method] = run_experiment(str(config_tmp_path))
        finally:
            if config_tmp_path.exists():
                config_tmp_path.unlink()

    return results


def _run_matched_seed_job(job: tuple[str, int, int, int, str, tuple[str, ...]]) -> tuple[int, dict[str, dict[str, Any]]]:
    output_root, seed, generations, population_size, benchmark, methods = job
    return (
        seed,
        run_matched_experiments(
            output_root=Path(output_root) / f"seed_{seed}",
            seed=seed,
            generations=generations,
            population_size=population_size,
            benchmark=benchmark,
            methods=methods,
        ),
    )


def run_matched_seed_sweep(
    *,
    output_root: str | Path,
    seeds: list[int] | tuple[int, ...],
    generations: int,
    population_size: int = 24,
    benchmark: str = "dwta_small",
    methods: list[str] | tuple[str, ...] | None = None,
    max_workers: int = 1,
) -> dict[int, dict[str, dict[str, Any]]]:
    """运行 matched comparisons 用于 multiple seeds 在 一个 基准问题."""
    root = Path(output_root)
    selected_methods = _resolve_methods(methods)
    seed_list = [int(seed) for seed in seeds]
    results: dict[int, dict[str, dict[str, Any]]] = {}

    if len(seed_list) <= 1 or max_workers == 1:
        for seed in seed_list:
            results[seed] = run_matched_experiments(
                output_root=root / f"seed_{seed}",
                seed=seed,
                generations=generations,
                population_size=population_size,
                benchmark=benchmark,
                methods=selected_methods,
            )
        return results

    worker_count = _resolve_max_workers(max_workers, len(seed_list))
    jobs = [
        (str(root), seed, generations, population_size, benchmark, selected_methods)
        for seed in seed_list
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for seed, seed_result in executor.map(_run_matched_seed_job, jobs):
            results[seed] = seed_result
    return results


def run_matched_matrix(
    *,
    output_root: str | Path,
    benchmarks: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    generations: int,
    population_size: int,
    methods: list[str] | tuple[str, ...] | None = None,
    max_workers: int = 1,
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """运行 full matched 矩阵 跨 基准问题 x 种子 x 方法."""
    root = Path(output_root)
    matrix_results: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for benchmark in benchmarks:
        benchmark_root = root / benchmark
        matrix_results[benchmark] = run_matched_seed_sweep(
            output_root=benchmark_root,
            seeds=seeds,
            generations=generations,
            population_size=population_size,
            benchmark=benchmark,
            methods=methods,
            max_workers=max_workers,
        )
    return matrix_results


__all__ = [
    "run_matched_experiments",
    "run_matched_seed_sweep",
    "run_matched_matrix",
]
