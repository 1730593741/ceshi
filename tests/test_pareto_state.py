"""用于测试 Pareto-状态 感知 over NSGA-II populations."""

import pytest

from optimizers.nsga2.population import Individual
from sensing.pareto_state import ParetoStateSensor


def test_pareto_state_sensor_outputs_expected_fields() -> None:
    population = [
        Individual(genome=[0], objectives=(1.0, 4.0), constraint_violation=0.0, feasible=True),
        Individual(genome=[1], objectives=(2.0, 2.0), constraint_violation=0.0, feasible=True),
        Individual(genome=[2], objectives=(4.0, 1.0), constraint_violation=1.0, feasible=False),
    ]

    sensor = ParetoStateSensor()
    state = sensor.sense(generation=1, population=population, reference_point=(5.0, 5.0))

    assert state.generation == 1
    assert state.hv == 11.0
    assert state.delta_hv == 11.0
    assert state.feasible_ratio == 2 / 3
    assert state.rank1_ratio == 2 / 3
    assert state.mean_cv == 1 / 3
    assert state.diversity_score > 0.0
    assert 0.0 <= state.crowding_entropy <= 1.0
    assert 0.0 <= state.d_dec <= 1.0
    assert 0.0 <= state.d_front <= 1.0
    assert state.stagnation_len == 0
    assert state.to_dict()["hv"] == 11.0
    assert "crowding_entropy" in state.to_dict()
    assert "d_dec" in state.to_dict()
    assert "d_front" in state.to_dict()


def test_pareto_state_stagnation_counter_increments() -> None:
    population = [
        Individual(genome=[0], objectives=(2.0, 2.0), constraint_violation=0.0, feasible=True),
        Individual(genome=[1], objectives=(2.5, 2.5), constraint_violation=0.0, feasible=True),
    ]

    sensor = ParetoStateSensor()
    state1 = sensor.sense(generation=1, population=population, reference_point=(5.0, 5.0))
    state2 = sensor.sense(
        generation=2,
        population=population,
        previous_state=state1,
        reference_point=(5.0, 5.0),
    )

    assert state2.delta_hv == 0.0
    assert state2.stagnation_len == 1


def test_pareto_state_sensor_does_not_mutate_population_ranks() -> None:
    population = [
        Individual(genome=[0], objectives=(1.0, 4.0), rank=7, constraint_violation=0.0, feasible=True),
        Individual(genome=[1], objectives=(2.0, 2.0), rank=8, constraint_violation=0.0, feasible=True),
    ]

    sensor = ParetoStateSensor()
    _ = sensor.sense(generation=1, population=population, reference_point=(5.0, 5.0))

    assert [ind.rank for ind in population] == [7, 8]


def test_crowding_entropy_uniform_front_higher_than_collapsed_front() -> None:
    sensor = ParetoStateSensor()

    uniform_front = [
        Individual(genome=[0, 0], objectives=(1.0, 9.0), feasible=True),
        Individual(genome=[0, 1], objectives=(3.0, 7.0), feasible=True),
        Individual(genome=[1, 0], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[1, 1], objectives=(7.0, 3.0), feasible=True),
        Individual(genome=[2, 0], objectives=(9.0, 1.0), feasible=True),
    ]
    collapsed_front = [
        Individual(genome=[0, 0], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[0, 1], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[1, 0], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[1, 1], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[2, 0], objectives=(5.0, 5.0), feasible=True),
    ]

    state_uniform = sensor.sense(generation=1, population=uniform_front, reference_point=(10.0, 10.0))
    state_collapsed = sensor.sense(generation=1, population=collapsed_front, reference_point=(10.0, 10.0))

    assert state_uniform.crowding_entropy > 0.0
    assert state_collapsed.crowding_entropy == 0.0
    assert state_uniform.crowding_entropy > state_collapsed.crowding_entropy


@pytest.mark.parametrize(
    ("population", "expected"),
    [
        (
            [
                Individual(genome=[0, 0, 0], objectives=(1.0, 1.0), feasible=True),
                Individual(genome=[0, 0, 0], objectives=(1.2, 1.2), feasible=True),
                Individual(genome=[0, 0, 0], objectives=(1.4, 1.4), feasible=True),
            ],
            0.0,
        ),
        (
            [
                Individual(genome=[0, 0, 0], objectives=(1.0, 1.0), feasible=True),
                Individual(genome=[1, 1, 1], objectives=(1.2, 1.2), feasible=True),
                Individual(genome=[2, 2, 2], objectives=(1.4, 1.4), feasible=True),
            ],
            1.0,
        ),
    ],
)
def test_d_dec_reflects_assignment_diversity(population: list[Individual], expected: float) -> None:
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=1, population=population, reference_point=(3.0, 3.0))

    assert state.d_dec == pytest.approx(expected)


def test_d_front_separation_higher_when_front_and_dominated_are_far() -> None:
    sensor = ParetoStateSensor()

    separated = [
        Individual(genome=[0], objectives=(1.0, 9.0), feasible=True),
        Individual(genome=[1], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[2], objectives=(9.0, 1.0), feasible=True),
        Individual(genome=[3], objectives=(20.0, 20.0), feasible=False, constraint_violation=1.0),
        Individual(genome=[4], objectives=(22.0, 19.0), feasible=False, constraint_violation=1.2),
    ]
    mixed = [
        Individual(genome=[0], objectives=(1.0, 9.0), feasible=True),
        Individual(genome=[1], objectives=(5.0, 5.0), feasible=True),
        Individual(genome=[2], objectives=(9.0, 1.0), feasible=True),
        Individual(genome=[3], objectives=(5.2, 5.1), feasible=False, constraint_violation=1.0),
        Individual(genome=[4], objectives=(4.9, 5.4), feasible=False, constraint_violation=1.2),
    ]

    state_separated = sensor.sense(generation=1, population=separated, reference_point=(30.0, 30.0))
    state_mixed = sensor.sense(generation=1, population=mixed, reference_point=(30.0, 30.0))

    assert state_separated.d_front > state_mixed.d_front
    assert state_separated.d_front > 0.5


def test_d_front_all_rank1_fallback_is_one() -> None:
    sensor = ParetoStateSensor()
    all_rank1 = [
        Individual(genome=[0], objectives=(1.0, 3.0), feasible=True),
        Individual(genome=[1], objectives=(2.0, 2.0), feasible=True),
        Individual(genome=[2], objectives=(3.0, 1.0), feasible=True),
    ]

    state = sensor.sense(generation=1, population=all_rank1, reference_point=(5.0, 5.0))
    assert state.rank1_ratio == 1.0
    assert state.d_front == 1.0
