"""用于测试 论文级 evaluation 指标."""

from __future__ import annotations

import math

from eval.metrics import igd, igd_plus, spacing, spread


def test_igd_and_igd_plus_zero_on_same_front() -> None:
    front = [(1.0, 2.0), (2.0, 1.0)]
    assert igd(front, front) == 0.0
    assert igd_plus(front, front) == 0.0


def test_igd_plus_not_larger_than_igd_on_simple_case() -> None:
    obtained = [(3.0, 3.0)]
    reference = [(1.0, 1.0), (2.0, 2.0)]
    assert igd_plus(obtained, reference) <= igd(obtained, reference)


def test_spacing_stability_for_small_or_duplicate_fronts() -> None:
    assert spacing([]) == 0.0
    assert spacing([(1.0, 1.0)]) == 0.0
    assert spacing([(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]) >= 0.0


def test_spread_on_bi_objective_front() -> None:
    reference = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
    obtained = [(1.0, 4.0), (2.5, 2.5), (4.0, 1.0)]
    value = spread(obtained, reference)
    assert 0.0 <= value <= 1.0


def test_igd_empty_obtained_is_inf() -> None:
    result = igd([], [(1.0, 1.0)])
    assert math.isinf(result)
