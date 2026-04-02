"""冒烟测试 用于 preset 矩阵 运行器."""

from __future__ import annotations

from pathlib import Path

import pytest

import experiments.run_matrix as run_matrix_module
from experiments.run_matrix import run_matrix


def test_run_matrix_toy_smoke(tmp_path: Path) -> None:
    payload = run_matrix(preset="toy", output_root=tmp_path / "runs", include_ablation=False)

    manifest = Path(payload["manifest_path"])
    assert manifest.exists()
    assert payload["matched"]["benchmarks"] == ["dwta_small_smoke"]
    assert payload["matched"]["methods"] == ["baseline_nsga2", "rule_control", "mock_llm"]
    assert payload["matched"]["max_workers"] == 1


def test_run_matrix_forwards_max_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    def fake_run_matched_matrix(**kwargs):
        calls["matched"] = kwargs["max_workers"]
        return {}

    def fake_run_ablation_matrix(**kwargs):
        calls["ablation"] = kwargs["max_workers"]
        return {}

    monkeypatch.setattr(run_matrix_module, "run_matched_matrix", fake_run_matched_matrix)
    monkeypatch.setattr(run_matrix_module, "run_ablation_matrix", fake_run_ablation_matrix)

    payload = run_matrix(preset="toy", output_root=tmp_path / "runs", include_ablation=True, max_workers=3)

    assert calls == {"matched": 3, "ablation": 3}
    assert payload["matched"]["max_workers"] == 3
    assert payload["ablation"]["max_workers"] == 3


def test_run_matrix_rejects_non_positive_max_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        run_matrix(preset="toy", output_root=tmp_path / "runs", include_ablation=False, max_workers=0)


def test_run_matrix_supports_seed_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, list[int]] = {}

    def fake_run_matched_matrix(**kwargs):
        seen["matched"] = list(kwargs["seeds"])
        return {}

    def fake_run_ablation_matrix(**kwargs):
        seen["ablation"] = list(kwargs["seeds"])
        return {}

    monkeypatch.setattr(run_matrix_module, "run_matched_matrix", fake_run_matched_matrix)
    monkeypatch.setattr(run_matrix_module, "run_ablation_matrix", fake_run_ablation_matrix)

    payload = run_matrix(
        preset="long_single",
        output_root=tmp_path / "runs",
        include_ablation=True,
        max_workers=2,
        seeds_override=(2026, 2027, 2028),
    )

    assert seen == {"matched": [2026, 2027, 2028], "ablation": [2026, 2027, 2028]}
    assert payload["matched"]["seeds"] == [2026, 2027, 2028]
    assert payload["ablation"]["seeds"] == [2026, 2027, 2028]
