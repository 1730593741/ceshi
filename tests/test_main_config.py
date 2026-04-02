"""用于测试 main configuration (DWTA-only)."""

from pathlib import Path

import pytest

from main import load_config, parse_cli_args


def test_load_config_rejects_non_dwta_problem(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
problem:
  n_tasks: 2
  n_resources: 2
  cost_matrix:
    - [1.0, 2.0]
    - [2.0, 1.0]
  task_loads: [1.0, 1.0]
  capacities: [1.5, 1.5]
solver:
  population_size: 6
  generations: 2
  crossover_prob: 0.9
  mutation_prob: 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-DWTA"):
        load_config(config_path)


def test_parse_cli_args_supports_config_flag_and_positional() -> None:
    assert parse_cli_args(["--config", "experiments/configs/dwta_small_smoke.yaml"]) == "experiments/configs/dwta_small_smoke.yaml"
    assert parse_cli_args(["experiments/configs/default.yaml"]) == "experiments/configs/default.yaml"


def test_load_dwta_precomputed_config() -> None:
    cfg = load_config("experiments/configs/dwta_medium.yaml")

    assert cfg.problem.problem_type == "dwta"
    assert cfg.problem.precomputed is not None
    assert cfg.problem.precomputed.ammo_capacities == [4, 4, 3, 3]


def test_load_dwta_scripted_config() -> None:
    cfg = load_config("experiments/configs/dwta_scripted_minimal.yaml")

    assert cfg.problem.problem_type == "dwta"
    assert cfg.problem.scenario_mode == "scripted_waves"
    assert cfg.problem.max_weapons == 4
    assert cfg.problem.max_targets == 6
    assert len(cfg.problem.waves) == 2


def test_load_dwta_hard_realworld_config() -> None:
    cfg = load_config("experiments/configs/dwta_hard_realworld.yaml")
    assert cfg.problem.problem_type == "dwta"
    assert cfg.problem.scenario_mode == "scripted_waves"
    assert cfg.problem.max_weapons == 16
    assert cfg.problem.max_targets == 32
    assert len(cfg.problem.targets or []) == 10


def test_load_hybrid_llm_hard_realworld_event_bypass_config() -> None:
    cfg = load_config("experiments/configs/hybrid_llm_hard_realworld_event_bypass.yaml")

    assert cfg.problem.problem_type == "dwta"
    assert cfg.problem.scenario_mode == "scripted_waves"
    assert cfg.controller.mode == "hybrid_llm"
    assert cfg.solver.population_size == 96
    assert cfg.solver.generations == 160
    assert cfg.controller.rule.control_interval == 5
    assert cfg.controller.rule.forced_control_on_major_event is True
