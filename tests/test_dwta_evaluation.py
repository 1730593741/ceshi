"""用于测试 DWTA evaluate_allocation 封装。"""

from __future__ import annotations

from src.dwta.evaluation import evaluate_allocation


def test_evaluate_allocation_returns_consistent_objectives_and_feasibility() -> None:
    result = evaluate_allocation(
        [2, 1, 0, 1],
        n_weapons=2,
        n_targets=2,
        required_damage=[5.0, 3.0],
        lethality_matrix=[[2.0, 1.0], [1.0, 2.0]],
        ammo_capacities=[3, 1],
        compatibility_matrix=[[1, 1], [1, 0]],
    )

    assert result.objectives == (1.0, 4.0)
    assert result.constraint_breakdown.compatibility == 1.0
    assert result.constraint_violation == 1.0
    assert result.feasible is False
