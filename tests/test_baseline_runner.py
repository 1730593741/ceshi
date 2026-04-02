"""用于测试 M7 基线 运行器s."""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.baselines.runner import run_baseline_nsga2, run_no_memory_baseline


def test_baseline_nsga2_runner_executes(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("experiments/configs/baseline_nsga2.yaml").read_text(encoding="utf-8"))
    payload.setdefault("logging", {})["output_dir"] = str(tmp_path / "baseline_logs")
    payload.setdefault("solver", {})["generations"] = 2
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summary = run_baseline_nsga2(str(config_path))
    assert summary["final_generation"] == 2
    assert summary["experiences_path"] is None
    assert (tmp_path / "baseline_logs" / "generation_metrics.jsonl").exists()
    assert (tmp_path / "baseline_logs" / "actions.jsonl").exists()


def test_no_memory_baseline_uses_ablation_switch(tmp_path: Path) -> None:
    config_path = tmp_path / "rule.yaml"
    payload = yaml.safe_load(Path("experiments/configs/rule_control.yaml").read_text(encoding="utf-8"))
    payload["logging"]["output_dir"] = str(tmp_path / "no_memory_logs")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summary = run_no_memory_baseline(str(config_path))
    assert summary["controller_mode"] == "rule"
    assert summary["experiences_path"] is None
