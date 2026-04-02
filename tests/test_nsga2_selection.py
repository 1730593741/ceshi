"""用于测试 NSGA-II sorting 与 选择 routines."""

import math
import random

from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import (
    assign_crowding_distance,
    environmental_selection,
    non_dominated_sort,
    parent_selection,
)


def test_non_dominated_sort_assigns_expected_first_front() -> None:
    a = Individual(genome=[0], objectives=(1.0, 2.0), feasible=True)
    b = Individual(genome=[1], objectives=(2.0, 1.0), feasible=True)
    c = Individual(genome=[2], objectives=(3.0, 3.0), feasible=True)

    fronts = non_dominated_sort([a, b, c])

    assert a in fronts[0]
    assert b in fronts[0]
    assert c in fronts[1]


def test_crowding_distance_marks_boundaries_infinite() -> None:
    front = [
        Individual(genome=[0], objectives=(1.0, 3.0), feasible=True),
        Individual(genome=[1], objectives=(2.0, 2.0), feasible=True),
        Individual(genome=[2], objectives=(3.0, 1.0), feasible=True),
    ]

    assign_crowding_distance(front)

    inf_count = sum(1 for ind in front if math.isinf(ind.crowding_distance))
    assert inf_count == 2


def test_parent_and_environment_selection_return_target_sizes() -> None:
    rng = random.Random(5)
    inds = [
        Individual(genome=[i], objectives=(float(i), float(10 - i)), feasible=True)
        for i in range(6)
    ]
    fronts = non_dominated_sort(inds)
    for front in fronts:
        assign_crowding_distance(front)

    parents = parent_selection(inds, n_parents=6, rng=rng)
    next_pop = environmental_selection(inds, population_size=4)

    assert len(parents) == 6
    assert len(next_pop) == 4
