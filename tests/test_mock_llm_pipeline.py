"""集成测试 用于 mock_llm 闭环 模式."""

from pathlib import Path
from types import SimpleNamespace

from main import build_controller, build_solver, load_config
from controller.closed_loop import ClosedLoopRunner, LLMChainController
from memory.experience_pool import ExperiencePool
from sensing.pareto_state import ParetoStateSensor


def test_mock_llm_mode_runs_closed_loop(tmp_path: Path) -> None:
    config = load_config("experiments/configs/mock_llm.yaml")
    config.log_path = str(tmp_path / "events.jsonl")
    config.memory.experience_log_path = str(tmp_path / "exp.jsonl")

    solver = build_solver(config.problem, config.optimizer)
    controller = build_controller(config)
    assert isinstance(controller, LLMChainController)

    experience_pool = ExperiencePool(config.memory.memory_window)
    runner = ClosedLoopRunner(
        solver=solver,
        sensor=ParetoStateSensor(),
        controller=controller,
        experience_pool=experience_pool,
    )
    states = runner.run(generations=6)

    assert states[-1].generation == 6
    assert 0.0 <= solver.config.mutation_prob <= 1.0
    assert 0.0 <= solver.config.crossover_prob <= 1.0

    records = experience_pool.recent(10)
    assert records
    assert all("control_state" in record.action for record in records)

def test_build_controller_compatible_with_legacy_llm_config_without_read_timeout() -> None:
    config = load_config("experiments/configs/mock_llm.yaml")
    legacy_llm = SimpleNamespace(
        provider=config.llm.provider,
        model=config.llm.model,
        timeout_s=config.llm.timeout_s,
        max_retries=config.llm.max_retries,
        api_key_env=config.llm.api_key_env,
        base_url_env=config.llm.base_url_env,
        model_env=config.llm.model_env,
        base_url=config.llm.base_url,
        fallback_mode=config.llm.fallback_mode,
    )
    config.llm = legacy_llm

    controller = build_controller(config)

    assert isinstance(controller, LLMChainController)