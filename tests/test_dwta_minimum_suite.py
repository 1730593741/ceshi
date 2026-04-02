"""论文导向 DWTA 最小可靠测试套件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from main import load_config, run_experiment


def _write_config_with_overrides(
    source_config: str,
    destination: Path,
    *,
    output_dir: Path,
    generations: int,
    population_size: int = 12,
) -> Path:
    payload: dict[str, Any] = yaml.safe_load(Path(source_config).read_text(encoding="utf-8"))
    payload.setdefault("logging", {})["output_dir"] = str(output_dir)

    if "solver" in payload:
        payload["solver"]["generations"] = generations
        payload["solver"]["population_size"] = population_size
    elif "optimizer" in payload:
        payload["optimizer"]["generations"] = generations
        payload["optimizer"]["population_size"] = population_size
    else:
        raise AssertionError("配置文件必须包含 solver 或 optimizer")

    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def test_tiny_dwta_baseline_smoke(tmp_path: Path) -> None:
    config_path = _write_config_with_overrides(
        "experiments/configs/baseline_nsga2.yaml",
        tmp_path / "baseline_tiny.yaml",
        output_dir=tmp_path / "baseline_out",
        generations=3,
        population_size=10,
    )

    summary = run_experiment(str(config_path))

    assert summary["controller_mode"] == "rule"
    assert summary["final_generation"] == 3
    assert summary["benchmark"] == "unknown"
    assert Path(summary["summary_path"]).exists()


def test_tiny_dynamic_event_application(tmp_path: Path) -> None:
    config_path = _write_config_with_overrides(
        "experiments/configs/dwta_scripted_waves_smoke.yaml",
        tmp_path / "dynamic_tiny.yaml",
        output_dir=tmp_path / "dynamic_out",
        generations=5,
        population_size=10,
    )

    summary = run_experiment(str(config_path))
    dynamic_summary = summary["dynamic_summary"]

    assert summary["final_generation"] == 5
    assert dynamic_summary["num_events"] >= 2
    assert "disable_weapons" in dynamic_summary["event_types_seen"]
    assert Path(summary["events_path"]).exists()


def test_mock_llm_closed_loop_smoke(tmp_path: Path) -> None:
    config_path = _write_config_with_overrides(
        "experiments/configs/mock_llm.yaml",
        tmp_path / "mock_llm_tiny.yaml",
        output_dir=tmp_path / "mock_llm_out",
        generations=4,
        population_size=12,
    )

    summary = run_experiment(str(config_path))

    assert summary["controller_mode"] == "mock_llm"
    assert summary["final_generation"] == 4
    assert summary["num_actions"] >= 1
    assert summary["num_experiences"] >= 1


def test_config_loading_for_dwta_modes() -> None:
    baseline_cfg = load_config("experiments/configs/baseline_nsga2.yaml")
    dynamic_cfg = load_config("experiments/configs/dwta_scripted_waves_smoke.yaml")

    assert baseline_cfg.problem.problem_type == "dwta"
    assert baseline_cfg.problem.scenario_mode == "static"
    assert dynamic_cfg.problem.scenario_mode == "scripted_waves"
    assert len(dynamic_cfg.problem.waves) >= 2


def test_experiment_runner_output_artifacts(tmp_path: Path) -> None:
    config_path = _write_config_with_overrides(
        "experiments/configs/baseline_nsga2.yaml",
        tmp_path / "artifact_tiny.yaml",
        output_dir=tmp_path / "artifact_out",
        generations=2,
        population_size=8,
    )

    summary = run_experiment(str(config_path))

    required_paths = [
        summary["events_path"],
        summary["generation_log_path"],
        summary["action_log_path"],
        summary["config_snapshot_path"],
        summary["summary_path"],
    ]
    for path in required_paths:
        assert Path(path).exists()

    summary_payload = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert summary_payload["run_id"].startswith("run-")
    assert summary_payload["config_fingerprint"]
    assert summary_payload["final_generation"] == 2
