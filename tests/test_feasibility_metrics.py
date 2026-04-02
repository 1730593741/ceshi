"""用于测试 可行性 与 约束-违反 感知 指标."""

from sensing.feasibility_metrics import (
    feasible_ratio,
    max_constraint_violation,
    mean_constraint_violation,
)


def test_feasibility_metric_values() -> None:
    violations = [0.0, 0.0, 1.5, 2.5]

    assert feasible_ratio(violations) == 0.5
    assert mean_constraint_violation(violations) == 1.0
    assert max_constraint_violation(violations) == 2.5


def test_feasibility_metrics_empty_input() -> None:
    assert feasible_ratio([]) == 0.0
    assert mean_constraint_violation([]) == 0.0
    assert max_constraint_violation([]) == 0.0
