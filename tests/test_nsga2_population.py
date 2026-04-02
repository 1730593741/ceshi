"""用于测试 NSGA-II 种群 structures."""

from optimizers.nsga2.population import Individual, Population


def test_individual_copy_preserves_values_and_copies_genome() -> None:
    ind = Individual(
        genome=[0, 1, 1],
        objectives=(1.0, 2.0),
        constraint_violation=0.0,
        feasible=True,
        rank=1,
        crowding_distance=0.5,
    )

    clone = ind.copy()
    clone.genome[0] = 2

    assert clone.objectives == (1.0, 2.0)
    assert ind.genome[0] == 0
    assert clone.genome[0] == 2


def test_population_wrapper_append_extend_and_len() -> None:
    pop = Population()
    pop.append(Individual(genome=[0]))
    pop.extend([Individual(genome=[1]), Individual(genome=[2])])

    assert len(pop) == 3
    assert [ind.genome[0] for ind in pop] == [0, 1, 2]
