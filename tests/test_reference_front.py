"""用于测试 reference-前沿 construction 与 provenance."""

from __future__ import annotations

import json
from pathlib import Path

from eval.reference_front import (
    build_empirical_reference_front,
    build_intra_run_reference_front,
    build_true_reference_front,
    read_final_front_from_generation_log,
)


def test_build_true_reference_front_filters_dominated() -> None:
    ref = build_true_reference_front([(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)], name="toy")
    assert ref.source == "true_pareto_front"
    assert (3.0, 3.0) not in ref.points


def test_read_final_front_from_generation_log(tmp_path: Path) -> None:
    log = tmp_path / "generation_metrics.jsonl"
    rows = [
        {"generation": 0, "rank1_objectives": [[3, 3]]},
        {"generation": 1, "rank1_objectives": [[1, 2], [2, 1]]},
    ]
    with log.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    front = read_final_front_from_generation_log(log)
    assert set(front) == {(1.0, 2.0), (2.0, 1.0)}


def test_build_empirical_reference_front(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    with a.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"generation": 1, "rank1_objectives": [[1, 3], [2, 2]]}) + "\n")
    with b.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"generation": 1, "rank1_objectives": [[3, 1], [2.5, 2.5]]}) + "\n")

    ref = build_empirical_reference_front({"m1": [a], "m2": [b]})
    assert ref.source == "empirical_matched_runs"
    assert len(ref.points) >= 2


def test_build_intra_run_reference_front() -> None:
    events = [
        {"rank1_objectives": [[3, 3]]},
        {"rank1_objectives": [[2, 2], [1.5, 2.2]]},
    ]
    ref = build_intra_run_reference_front(events)
    assert ref.source == "intra_run_union_front"
    assert len(ref.points) >= 1
