"""用于测试 real_llm 模式 配置 wiring 与 graceful fallback behavior."""

from __future__ import annotations

from pathlib import Path

from controller.closed_loop import ClosedLoopRunner, LLMChainController
from main import build_controller, build_solver, load_config
from memory.experience_pool import ExperiencePool
from sensing.pareto_state import ParetoStateSensor


def test_real_llm_yaml_loads_and_wires_runtime_fields() -> None:
    config = load_config("experiments/configs/real_llm.yaml")
    assert config.controller_mode.mode == "real_llm"
    assert config.llm.api_key_env == "OPENAI_API_KEY"
    assert config.llm.base_url_env == "OPENAI_BASE_URL"
    assert config.llm.model_env == "OPENAI_MODEL"
    assert config.llm.fallback_mode in {"mock_llm", "hold"}


def test_real_llm_mode_runs_with_missing_key_via_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = load_config("experiments/configs/real_llm.yaml")
    config.log_path = str(tmp_path / "events.jsonl")
    config.memory.experience_log_path = str(tmp_path / "exp.jsonl")

    solver = build_solver(config.problem, config.optimizer)
    controller = build_controller(config)
    assert isinstance(controller, LLMChainController)

    runner = ClosedLoopRunner(
        solver=solver,
        sensor=ParetoStateSensor(),
        controller=controller,
        experience_pool=ExperiencePool(config.memory.memory_window),
    )
    states = runner.run(generations=4)

    assert states[-1].generation == 4
    assert 0.0 <= solver.config.mutation_prob <= 1.0
    assert 0.0 <= solver.config.crossover_prob <= 1.0
