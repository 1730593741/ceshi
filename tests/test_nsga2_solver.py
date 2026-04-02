"""集成测试 用于 最小 NSGA-II 求解器."""

from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from controller.operator_space import OperatorParams
from src.dwta.precompute import build_precomputed_matrices
from src.dwta.model import MunitionType, Target, Weapon


def test_dwta_solver_matrix_operators_run_for_multiple_generations() -> None:
    dwta_data = build_precomputed_matrices(
        munitions=[MunitionType(id="m1", max_range=10.0, flight_speed=2.0, lethality=2.0)],
        weapons=[
            Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3),
            Weapon(id="w2", x=1.0, y=0.0, munition_type_id="m1", ammo_capacity=2),
        ],
        targets=[
            Target(id="t1", x=2.0, y=0.0, required_damage=2.0, time_window=(0.0, 6.0)),
            Target(id="t2", x=3.0, y=0.0, required_damage=2.5, time_window=(0.0, 6.0)),
            Target(id="t3", x=4.0, y=0.0, required_damage=1.5, time_window=(0.0, 6.0)),
        ],
    )

    solver = NSGA2Solver(
        n_tasks=0,
        n_resources=1,
        cost_matrix=[],
        task_loads=[],
        capacities=[],
        dwta_data=dwta_data,
        config=NSGA2Config(
            population_size=12,
            generations=4,
            crossover_prob=0.9,
            mutation_prob=0.35,
            repair_prob=1.0,
            eta_c=75.0,
            eta_m=20.0,
            local_search_prob=0.4,
            seed=5,
        ),
    )

    population = solver.run()

    assert len(population) == 12
    assert all(len(ind.genome) == dwta_data.n_weapons * dwta_data.n_targets for ind in population)
    assert all(all(isinstance(value, int) for value in ind.genome) for ind in population)
    assert all(ind.constraint_violation == 0.0 for ind in population)


def test_dwta_solver_exposes_and_applies_unified_optional_operator_params() -> None:
    dwta_data = build_precomputed_matrices(
        munitions=[MunitionType(id="m1", max_range=8.0, flight_speed=2.0, lethality=1.5)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=2)],
        targets=[Target(id="t1", x=2.0, y=0.0, required_damage=1.0, time_window=(0.0, 3.0))],
    )

    solver = NSGA2Solver(
        n_tasks=0,
        n_resources=1,
        cost_matrix=[],
        task_loads=[],
        capacities=[],
        dwta_data=dwta_data,
        config=NSGA2Config(population_size=4, generations=1, seed=9),
    )

    caps = solver.get_operator_capabilities()
    assert caps.supports_eta_c is True
    assert caps.supports_eta_m is True
    assert caps.supports_local_search_prob is True

    solver.set_operator_params(
        OperatorParams(
            mutation_prob=0.3,
            crossover_prob=0.8,
            eta_c=60.0,
            eta_m=15.0,
            repair_prob=0.9,
            local_search_prob=0.25,
        )
    )
    params = solver.get_operator_params()
    assert params.eta_c == 60.0
    assert params.eta_m == 15.0
    assert params.local_search_prob == 0.25
