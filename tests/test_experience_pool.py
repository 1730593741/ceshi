"""用于测试 M5 experience pool behavior."""

import pytest

from memory.experience_pool import ExperiencePool, ExperienceRecord


def _record(idx: int) -> ExperienceRecord:
    return ExperienceRecord(
        state={"generation": idx},
        action={"generation": idx, "mutation_prob": 0.1, "crossover_prob": 0.9, "reason": "test"},
        reward=float(idx),
        next_state={"generation": idx + 1},
    )


def test_experience_pool_append_and_recent() -> None:
    pool = ExperiencePool(max_size=3)
    for idx in range(3):
        pool.append(_record(idx))

    recent = pool.recent(2)
    assert len(pool) == 3
    assert [item.state["generation"] for item in recent] == [1, 2]


def test_experience_pool_sliding_window_trim() -> None:
    pool = ExperiencePool(max_size=2)
    pool.append(_record(0))
    pool.append(_record(1))
    pool.append(_record(2))

    assert len(pool) == 2
    assert [item.state["generation"] for item in pool.get_recent(5)] == [1, 2]


def test_experience_pool_rejects_non_positive_max_size() -> None:
    with pytest.raises(ValueError, match="max_size"):
        ExperiencePool(max_size=0)


def test_experience_pool_recent_with_non_positive_n_returns_empty() -> None:
    pool = ExperiencePool(max_size=2)
    pool.append(_record(0))
    assert pool.recent(0) == []
    assert pool.recent(-1) == []
