"""用于测试 M4/M5 规则-based 闭环 运行器."""

import pytest

from controller.closed_loop import (
    ClosedLoopRunner,
    ControlAction,
    NSGA2Adaptation,
    Observation,
    RewardConfig,
    RuleBasedController,
    RuleControllerConfig,
)
from controller.control_semantics import ControlState
from controller.operator_space import OperatorCapabilities, OperatorParams
from memory.experience_pool import ExperiencePool
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from src.dwta.live_cache import DWTALiveCache
from src.dwta.model import DWTAWaveEvent, MunitionType, Target, Weapon
from src.dwta.scenario_builder import build_dynamic_scenario
from sensing.pareto_state import ParetoStateSensor


class InMemoryLogger:
    """收集 事件 用于 assertions 在 测试."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log(self, event: dict[str, object]) -> None:
        self.events.append(event)


class InMemoryExperienceLogger:
    """收集 experience records 用于 assertions 在 测试."""

    def __init__(self) -> None:
        self.records: list[object] = []

    def log(self, record: object) -> None:
        self.records.append(record)


class RequestedAppliedDivergenceController:
    """Controller 该 emits divergent requested/applied params 用于 regression 测试."""

    control_interval = 1

    def decide(
        self,
        *,
        state,
        recent_experiences,
        current_params: OperatorParams,
        capabilities: OperatorCapabilities,
    ) -> ControlAction:
        del recent_experiences, capabilities
        return ControlAction(
            generation=state.generation,
            mutation_prob=current_params.mutation_prob,
            crossover_prob=current_params.crossover_prob,
            reason="test",
            control_state=ControlState.MAINTAIN_BALANCE,
            requested_params={
                "mutation_prob": current_params.mutation_prob,
                "crossover_prob": current_params.crossover_prob,
                "repair_prob": 1.0,
            },
            applied_params={
                "mutation_prob": current_params.mutation_prob,
                "crossover_prob": current_params.crossover_prob,
                "repair_prob": 0.2,
            },
        )


def _build_solver() -> NSGA2Solver:
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=12.0, flight_speed=3.0, lethality=2.0)],
        weapons=[
            Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3),
            Weapon(id="w2", x=2.0, y=0.0, munition_type_id="m1", ammo_capacity=2),
        ],
        targets=[
            Target(id="t1", x=3.0, y=0.0, required_damage=3.0, time_window=(0.0, 4.0)),
            Target(id="t2", x=6.0, y=0.0, required_damage=2.0, time_window=(0.0, 4.0)),
        ],
    )
    cache = DWTALiveCache(scenario)
    cache.refresh()
    return NSGA2Solver(
        n_tasks=0,
        n_resources=1,
        cost_matrix=[],
        task_loads=[],
        capacities=[],
        dwta_data=scenario.base_data,
        dwta_live_cache=cache,
        config=NSGA2Config(
            population_size=12,
            generations=6,
            crossover_prob=0.9,
            mutation_prob=0.1,
            seed=11,
            repair_prob=1.0,
        ),
    )


def test_closed_loop_runs_and_logs_actions() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    sensor = ParetoStateSensor()
    logger = InMemoryLogger()

    runner = ClosedLoopRunner(solver=solver, sensor=sensor, controller=controller, logger=logger)
    states = runner.run(generations=6)

    assert len(states) == 7
    assert states[-1].generation == 6
    assert any(event["event"] == "action" for event in logger.events)
    action_events = [event for event in logger.events if event["event"] == "action"]
    assert all("control_state" in event for event in action_events)
    assert all("reason_detail" in event for event in action_events)
    state_events = [event for event in logger.events if event["event"] == "state"]
    assert state_events
    latest_state_event = state_events[-1]
    assert "crowding_entropy" in latest_state_event
    assert "d_dec" in latest_state_event
    assert "d_front" in latest_state_event
    assert "active_targets_count" in latest_state_event
    assert "active_weapons_count" in latest_state_event
    assert "cache_refresh_count" in latest_state_event
    assert 0.0 <= solver.config.mutation_prob <= 1.0
    assert 0.0 <= solver.config.crossover_prob <= 1.0


def test_closed_loop_records_experience_chain() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    sensor = ParetoStateSensor()
    experience_pool = ExperiencePool(max_size=10)
    experience_logger = InMemoryExperienceLogger()

    runner = ClosedLoopRunner(
        solver=solver,
        sensor=sensor,
        controller=controller,
        experience_pool=experience_pool,
        experience_logger=experience_logger,
        reward_config=RewardConfig(alpha=1.0, beta=0.1),
    )
    runner.run(generations=6)

    assert len(experience_pool) == 2
    first = experience_pool.recent(1)[0]
    assert set(first.to_dict().keys()) == {
        "state",
        "action",
        "reward",
        "next_state",
        "stage_id",
        "wave_id",
        "trigger_event_id",
    }
    assert "crowding_entropy" in first.state
    assert "d_dec" in first.state
    assert "d_front" in first.state
    assert "crowding_entropy" in first.next_state
    assert "d_dec" in first.next_state
    assert "d_front" in first.next_state
    assert len(experience_logger.records) == 2


def test_rule_controller_config_validates_control_interval() -> None:
    with pytest.raises(ValueError, match="control_interval"):
        RuleControllerConfig(control_interval=0)

    with pytest.raises(ValueError, match="event_control_cooldown"):
        RuleControllerConfig(event_control_cooldown=-1)


def test_rule_controller_decide_backwards_compatible_signature() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)

    action = controller.decide(
        state=state,
        current_mutation=solver.config.mutation_prob,
        current_crossover=solver.config.crossover_prob,
    )

    assert action.generation == 0
    assert 0.0 <= action.mutation_prob <= 1.0
    assert 0.0 <= action.crossover_prob <= 1.0


def test_rule_controller_decide_with_shared_observation() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)
    observation = Observation(
        state=state,
        recent_experiences=[],
        current_params=solver.get_operator_params(),
        capabilities=solver.get_operator_capabilities(),
    )

    action = controller.decide(observation=observation)

    assert action.generation == 0
    assert 0.0 <= action.mutation_prob <= 1.0
    assert 0.0 <= action.crossover_prob <= 1.0


def test_closed_loop_skips_terminal_generation_action() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=3))
    sensor = ParetoStateSensor()
    logger = InMemoryLogger()

    runner = ClosedLoopRunner(solver=solver, sensor=sensor, controller=controller, logger=logger)
    runner.run(generations=6)

    action_generations = [int(event["generation"]) for event in logger.events if event["event"] == "action"]
    assert action_generations == [3]


def test_rule_controller_maps_to_feasibility_state() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2, feasible_ratio_low=0.95))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)
    state.feasible_ratio = 0.2

    action = controller.decide(
        state=state,
        current_mutation=solver.config.mutation_prob,
        current_crossover=solver.config.crossover_prob,
    )

    assert action.control_state == ControlState.INCREASE_FEASIBILITY


def test_rule_controller_maps_to_diversity_state() -> None:
    solver = _build_solver()
    controller = RuleBasedController(RuleControllerConfig(control_interval=2, diversity_low=0.95, feasible_ratio_low=0.0))
    population = solver.initialize_population()
    sensor = ParetoStateSensor()
    state = sensor.sense(generation=0, population=population, previous_state=None, reference_point=None)
    state.diversity_score = 0.0

    action = controller.decide(
        state=state,
        current_mutation=solver.config.mutation_prob,
        current_crossover=solver.config.crossover_prob,
    )

    assert action.control_state == ControlState.INCREASE_DIVERSITY


def test_closed_loop_applies_applied_params_not_requested_params() -> None:
    solver = _build_solver()
    sensor = ParetoStateSensor()
    runner = ClosedLoopRunner(
        solver=solver,
        sensor=sensor,
        controller=RequestedAppliedDivergenceController(),
    )

    runner.run(generations=2)

    assert solver.config.repair_prob == 0.2


def test_nsga2_adaptation_applies_explicit_action_space() -> None:
    solver = _build_solver()
    adaptation = NSGA2Adaptation(solver)
    before = solver.get_operator_params()
    action = ControlAction(
        generation=1,
        mutation_prob=0.22,
        crossover_prob=0.88,
        reason="test",
        applied_params={"mutation_prob": 0.22, "crossover_prob": 0.88},
    )

    applied = adaptation.apply_action(action)

    assert applied.mutation_prob == 0.22
    assert applied.crossover_prob == 0.88
    assert solver.get_operator_params().mutation_prob == 0.22
    assert before.mutation_prob != solver.get_operator_params().mutation_prob


def test_closed_loop_applies_dwta_runtime_events_and_logs() -> None:
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=9.0, flight_speed=3.0, lethality=2.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3)],
        targets=[Target(id="t1", x=2.0, y=0.0, required_damage=2.0, time_window=(0.0, 4.0))],
        scenario_mode="scripted_waves",
        max_weapons=2,
        max_targets=2,
        waves=[
            DWTAWaveEvent(
                wave_id="disable_w1",
                trigger_generation=1,
                event_type="disable_weapons",
                payload={"weapon_ids": ["w1"]},
            ),
            DWTAWaveEvent(
                wave_id="inject_t2",
                trigger_generation=2,
                event_type="inject_targets",
                payload={"targets": [{"id": "t2", "x": 3.0, "y": 0.0, "required_damage": 2.0, "time_window": [0.0, 4.0]}]},
            ),
        ],
    )
    live_cache = DWTALiveCache(scenario)
    live_cache.refresh()

    solver = NSGA2Solver(
        n_tasks=0,
        n_resources=1,
        cost_matrix=[],
        task_loads=[],
        capacities=[],
        config=NSGA2Config(population_size=8, generations=3, crossover_prob=0.9, mutation_prob=0.1, seed=3),
        dwta_data=scenario.base_data,
        dwta_live_cache=live_cache,
    )
    runner = ClosedLoopRunner(
        solver=solver,
        sensor=ParetoStateSensor(),
        controller=RuleBasedController(RuleControllerConfig(control_interval=1)),
        logger=InMemoryLogger(),
    )
    states = runner.run(generations=3)
    runtime_events = [event for event in runner.logger.events if event["event"] == "runtime_event"]  # type: ignore[union-attr]

    assert len(states) == 4
    assert len(runtime_events) == 2
    assert all(event["live_cache_invalidated"] is True for event in runtime_events)
    assert all(int(event["cache_refresh_count"]) >= 2 for event in runtime_events)


def test_closed_loop_supports_periodic_and_event_trigger_with_cooldown() -> None:
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=9.0, flight_speed=3.0, lethality=2.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3)],
        targets=[Target(id="t1", x=2.0, y=0.0, required_damage=2.0, time_window=(0.0, 4.0))],
        scenario_mode="scripted_waves",
        max_weapons=2,
        max_targets=2,
        waves=[
            DWTAWaveEvent(
                wave_id="disable_w1",
                trigger_generation=1,
                event_type="disable_weapons",
                payload={"weapon_ids": ["w1"]},
            ),
            DWTAWaveEvent(
                wave_id="inject_t2",
                trigger_generation=2,
                event_type="inject_targets",
                payload={"targets": [{"id": "t2", "x": 3.0, "y": 0.0, "required_damage": 2.0, "time_window": [0.0, 4.0]}]},
            ),
        ],
    )
    live_cache = DWTALiveCache(scenario)
    live_cache.refresh()

    solver = NSGA2Solver(
        n_tasks=0,
        n_resources=1,
        cost_matrix=[],
        task_loads=[],
        capacities=[],
        config=NSGA2Config(population_size=8, generations=5, crossover_prob=0.9, mutation_prob=0.1, seed=3),
        dwta_data=scenario.base_data,
        dwta_live_cache=live_cache,
    )
    logger = InMemoryLogger()
    runner = ClosedLoopRunner(
        solver=solver,
        sensor=ParetoStateSensor(),
        controller=RuleBasedController(
            RuleControllerConfig(
                control_interval=2,
                event_triggered_control=True,
                event_control_cooldown=1,
                forced_control_on_major_event=False,
            )
        ),
        logger=logger,
    )
    runner.run(generations=5)

    action_events = [event for event in logger.events if event["event"] == "action"]
    skip_events = [event for event in logger.events if event["event"] == "control_skip"]

    trigger_types = {str(event["trigger_type"]) for event in action_events}
    assert "periodic" in trigger_types
    assert "event" in trigger_types
    assert any(bool(event["cooldown_skipped"]) for event in skip_events)
    event_actions = [event for event in action_events if event["trigger_type"] == "event"]
    assert all(event["trigger_event_id"] for event in event_actions)


def test_closed_loop_forces_major_event_through_cooldown() -> None:
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=9.0, flight_speed=3.0, lethality=2.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3)],
        targets=[Target(id="t1", x=2.0, y=0.0, required_damage=2.0, time_window=(0.0, 4.0))],
        scenario_mode="scripted_waves",
        max_weapons=2,
        max_targets=2,
        waves=[
            DWTAWaveEvent(
                wave_id="disable_w1",
                trigger_generation=1,
                event_type="disable_weapons",
                payload={"weapon_ids": ["w1"]},
            ),
            DWTAWaveEvent(
                wave_id="inject_t2",
                trigger_generation=2,
                event_type="inject_targets",
                payload={"targets": [{"id": "t2", "x": 3.0, "y": 0.0, "required_damage": 2.0, "time_window": [0.0, 4.0]}]},
            ),
        ],
    )
    live_cache = DWTALiveCache(scenario)
    live_cache.refresh()

    solver = NSGA2Solver(
        n_tasks=0,
        n_resources=1,
        cost_matrix=[],
        task_loads=[],
        capacities=[],
        config=NSGA2Config(population_size=8, generations=5, crossover_prob=0.9, mutation_prob=0.1, seed=3),
        dwta_data=scenario.base_data,
        dwta_live_cache=live_cache,
    )
    logger = InMemoryLogger()
    runner = ClosedLoopRunner(
        solver=solver,
        sensor=ParetoStateSensor(),
        controller=RuleBasedController(
            RuleControllerConfig(
                control_interval=2,
                event_triggered_control=True,
                event_control_cooldown=1,
                forced_control_on_major_event=True,
            )
        ),
        logger=logger,
    )
    runner.run(generations=5)

    action_events = [event for event in logger.events if event["event"] == "action"]
    event_actions = [event for event in action_events if event["trigger_type"] == "event"]
    skip_events = [event for event in logger.events if event["event"] == "control_skip"]

    assert [int(event["generation"]) for event in event_actions] == [1, 2]
    assert not any(int(event["generation"]) == 2 and event["trigger_type"] == "event" for event in skip_events)
