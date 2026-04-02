"""冒烟测试s 用于 matched 实验 辅助工具."""

from __future__ import annotations

from pathlib import Path

from experiments.baselines.matched_runner import run_matched_experiments


def test_run_matched_experiments_smoke(tmp_path: Path) -> None:
    results = run_matched_experiments(
        output_root=tmp_path / "matched",
        seed=33,
        generations=3,
        benchmark="dwta_small_smoke",
        population_size=12,
        methods=("baseline_nsga2", "rule_control", "mock_llm"),
    )

    assert set(results.keys()) == {"baseline_nsga2", "rule_control", "mock_llm"}
    for mode, summary in results.items():
        assert summary["seed"] == 33
        assert summary["generations"] == 3
        assert summary["benchmark"] == "dwta_small_smoke"
        assert summary["controller_mode"] in {"rule", "mock_llm"}
        assert Path(summary["summary_path"]).exists()
        assert str(tmp_path / "matched" / mode) in summary["summary_path"]
