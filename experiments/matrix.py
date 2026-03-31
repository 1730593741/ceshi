"""可复用的实验矩阵规格 用于 matched 与 ablation runs."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MATCHED_METHODS: tuple[str, ...] = (
    "baseline_nsga2",
    "rule_control",
    "mock_llm",
    "real_llm",
    "hybrid_llm",
)

DEFAULT_MATCHED_BENCHMARKS: tuple[str, ...] = (
    "dwta_small",
    "dwta_medium",
)

DEFAULT_MATCHED_SEEDS: tuple[int, ...] = (11, 23, 37)


@dataclass(frozen=True, slots=True)
class MatchedMatrix:
    """Matched 矩阵 axes 用于 paper-ready 方法 comparisons."""

    methods: tuple[str, ...] = DEFAULT_MATCHED_METHODS
    benchmarks: tuple[str, ...] = DEFAULT_MATCHED_BENCHMARKS
    seeds: tuple[int, ...] = DEFAULT_MATCHED_SEEDS
    generations: int = 20
    population_size: int = 24


@dataclass(frozen=True, slots=True)
class AblationMatrix:
    """Ablation 矩阵 axes 用于 方法 component removal/comparison."""

    seeds: tuple[int, ...] = DEFAULT_MATCHED_SEEDS
    benchmarks: tuple[str, ...] = DEFAULT_MATCHED_BENCHMARKS
    generations: int = 20
    population_size: int = 24
    tau_values: tuple[int, ...] = (1, 3, 5, 10)
    memory_windows: tuple[int, ...] = (5, 20, 50)


MATCHED_PRESETS: dict[str, MatchedMatrix] = {
    # quick preset: single seed, halved generations, halved population.
    # Covers both large-scale benchmarks for fast exploratory runs.
    # dwta_large_static:   pop 96->48, gen 120->60
    # dwta_hard_realworld: pop 128->64, gen 160->80
    "quick": MatchedMatrix(
        methods=DEFAULT_MATCHED_METHODS,
        benchmarks=("dwta_large_static", "dwta_hard_realworld"),
        seeds=(2026,),
        generations=60,
        population_size=48,
    ),
    "long_single": MatchedMatrix(
        methods=DEFAULT_MATCHED_METHODS,
        benchmarks=("dwta_hard_realworld",),
        seeds=(2026,),
        generations=160,
        population_size=96,
    ),
    "toy": MatchedMatrix(
        methods=("baseline_nsga2", "rule_control", "mock_llm"),
        benchmarks=("dwta_small_smoke",),
        seeds=(7,),
        generations=4,
        population_size=12,
    ),
    "pilot": MatchedMatrix(
        methods=DEFAULT_MATCHED_METHODS,
        benchmarks=("dwta_small",),
        seeds=(11, 23),
        generations=10,
        population_size=20,
    ),
    "paper": MatchedMatrix(),
    # full preset: adds large-scale benchmarks on top of the paper matrix.
    # dwta_large_static  — 12 weapons x 24 targets, 120 generations, pop 96.
    # dwta_hard_realworld — 8-16 weapons x 10-15 targets, 160 generations,
    #                       5 scripted wave events, pop 128.
    # These benchmarks use their own optimizer settings (population_size /
    # generations) defined in their YAML files; the values below are
    # intentionally matched to those configs.
    "full": MatchedMatrix(
        methods=DEFAULT_MATCHED_METHODS,
        benchmarks=("dwta_small", "dwta_medium", "dwta_large_static", "dwta_hard_realworld"),
        seeds=DEFAULT_MATCHED_SEEDS,
        generations=120,
        population_size=96,
    ),
}

ABLATION_PRESETS: dict[str, AblationMatrix] = {
    # quick preset: ablation skipped by default (use --skip-ablation).
    # Stub entry required so run_matrix validation passes for this preset.
    "quick": AblationMatrix(
        seeds=(2026,),
        benchmarks=("dwta_large_static", "dwta_hard_realworld"),
        generations=60,
        population_size=48,
        tau_values=(5,),
        memory_windows=(20,),
    ),
    "long_single": AblationMatrix(
        seeds=(2026,),
        benchmarks=("dwta_hard_realworld",),
        generations=160,
        population_size=128,
        tau_values=(3, 5, 10),
        memory_windows=(5, 20, 40),
    ),
    "toy": AblationMatrix(
        seeds=(7,),
        benchmarks=("dwta_small_smoke",),
        generations=4,
        population_size=12,
        tau_values=(1, 3),
        memory_windows=(5, 20),
    ),
    "pilot": AblationMatrix(
        seeds=(11,),
        benchmarks=("dwta_small",),
        generations=10,
        population_size=20,
    ),
    "paper": AblationMatrix(),
    # full ablation: same large benchmarks, reduced seed count to control cost.
    "full": AblationMatrix(
        seeds=(11, 23),
        benchmarks=("dwta_small", "dwta_medium", "dwta_large_static", "dwta_hard_realworld"),
        generations=120,
        population_size=96,
        tau_values=(1, 3, 5, 10),
        memory_windows=(5, 20, 50),
    ),
}
