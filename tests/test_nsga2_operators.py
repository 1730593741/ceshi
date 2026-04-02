"""用于测试 NSGA-II variation 算子."""

import random

from optimizers.nsga2.operators import (
    matrix_block_crossover,
    mutate_assignment,
    mutate_dwta_allocation,
    one_point_crossover,
)
from src.dwta.constraints import constraint_breakdown
from src.dwta.repair import repair_allocation


def test_one_point_crossover_respects_genome_length_and_values() -> None:
    rng = random.Random(0)
    p1 = [0, 0, 0, 0]
    p2 = [1, 1, 1, 1]

    c1, c2 = one_point_crossover(p1, p2, crossover_prob=1.0, rng=rng)

    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert set(c1).issubset({0, 1})
    assert set(c2).issubset({0, 1})


def test_mutation_keeps_values_in_resource_range() -> None:
    rng = random.Random(1)
    genome = [0, 1, 2, 1, 0]

    mutated = mutate_assignment(genome, n_resources=3, mutation_prob=1.0, rng=rng)

    assert len(mutated) == len(genome)
    assert all(0 <= gene < 3 for gene in mutated)


def test_matrix_block_crossover_preserves_integer_matrix_structure() -> None:
    rng = random.Random(3)
    parent_a = [0, 1, 2, 0, 1, 3]
    parent_b = [3, 0, 1, 2, 0, 0]

    child_a, child_b = matrix_block_crossover(
        parent_a,
        parent_b,
        n_weapons=2,
        n_targets=3,
        crossover_prob=1.0,
        rng=rng,
        eta_c=100.0,
    )

    assert len(child_a) == len(parent_a)
    assert len(child_b) == len(parent_b)
    assert all(isinstance(value, int) for value in child_a)
    assert all(isinstance(value, int) for value in child_b)
    assert set(child_a).issubset(set(parent_a) | set(parent_b))
    assert set(child_b).issubset(set(parent_a) | set(parent_b))


def test_dwta_mutation_never_changes_incompatible_cells() -> None:
    rng = random.Random(11)
    compatibility_matrix = [[1, 0, 1], [0, 1, 1]]
    genome = [1, 0, 2, 0, 2, 1]

    mutated = mutate_dwta_allocation(
        genome,
        n_weapons=2,
        n_targets=3,
        compatibility_matrix=compatibility_matrix,
        mutation_prob=1.0,
        rng=rng,
        mutation_step=3.0,
        eta_m=5.0,
        local_search_prob=1.0,
    )

    assert mutated[1] == genome[1]
    assert mutated[3] == genome[3]


def test_dwta_mutation_plus_repair_restores_feasibility() -> None:
    rng = random.Random(21)
    compatibility_matrix = [[1, 1], [1, 0]]
    ammo_capacities = [3, 1]
    genome = [2, 1, 1, 0]

    mutated = mutate_dwta_allocation(
        genome,
        n_weapons=2,
        n_targets=2,
        compatibility_matrix=compatibility_matrix,
        mutation_prob=1.0,
        rng=rng,
        mutation_step=4.0,
        eta_m=0.0,
        local_search_prob=0.0,
    )
    repaired = repair_allocation(
        mutated,
        ammo_capacities=ammo_capacities,
        compatibility_matrix=compatibility_matrix,
        n_targets=2,
        rng=random.Random(22),
    )
    breakdown = constraint_breakdown(
        repaired,
        ammo_capacities=ammo_capacities,
        compatibility_matrix=compatibility_matrix,
        n_targets=2,
    )

    assert breakdown.total == 0.0
