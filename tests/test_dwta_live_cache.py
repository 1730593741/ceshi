"""Unit tests for DWTA live cache refresh semantics."""

from __future__ import annotations

from src.dwta.live_cache import DWTALiveCache
from src.dwta.model import DWTAWaveEvent, MunitionType, Target, Weapon
from src.dwta.scenario_builder import build_dynamic_scenario


def test_live_cache_refresh_after_environment_change() -> None:
    """Cache should rebuild matrices after invalidation and environment update."""
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=10.0, flight_speed=2.0, lethality=3.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=3)],
        targets=[Target(id="t1", x=4.0, y=0.0, required_damage=2.0, time_window=(0.0, 3.0))],
        scenario_mode="scripted_waves",
    )
    cache = DWTALiveCache(scenario)

    first = cache.refresh()
    assert first.compatibility_mask.astype(int).tolist() == [[1]]
    assert first.flight_time_matrix.tolist()[0][0] == 2.0

    scenario.targets = [
        Target(
            id="t1",
            x=9.0,
            y=0.0,
            required_damage=2.0,
            time_window=(0.0, 3.0),
        )
    ]
    scenario.state_epoch += 1

    cache.invalidate()
    second = cache.refresh()
    assert second.compatibility_mask.astype(int).tolist() == [[0]]
    assert second.flight_time_matrix.tolist()[0][0] == 4.5


def test_live_cache_refresh_after_runtime_event() -> None:
    """Scripted runtime events should mutate env and force cache rebuild."""
    scenario = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=8.0, flight_speed=2.0, lethality=2.0)],
        weapons=[
            Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=2),
            Weapon(id="w2", x=1.0, y=0.0, munition_type_id="m1", ammo_capacity=2),
        ],
        targets=[Target(id="t1", x=2.0, y=0.0, required_damage=1.0, time_window=(0.0, 4.0))],
        scenario_mode="scripted_waves",
        max_weapons=3,
        max_targets=3,
    )
    cache = DWTALiveCache(scenario)
    first = cache.refresh()
    assert first.n_targets == 3
    assert first.ammo_capacities.tolist()[:2] == [2, 2]

    disable_event = DWTAWaveEvent(
        wave_id="disable",
        trigger_generation=1,
        event_type="disable_weapons",
        payload={"weapon_ids": ["w2"]},
    )
    scenario.apply_wave_event(disable_event)
    cache.invalidate()
    after_disable = cache.refresh()
    assert after_disable.ammo_capacities.tolist()[:2] == [2, 0]

    inject_event = DWTAWaveEvent(
        wave_id="inject",
        trigger_generation=2,
        event_type="inject_targets",
        payload={"targets": [{"id": "t2", "x": 3.0, "y": 0.0, "required_damage": 2.0, "time_window": [0.0, 3.0]}]},
    )
    scenario.apply_wave_event(inject_event)
    cache.invalidate()
    after_inject = cache.refresh()
    assert after_inject.n_targets == 3
    assert "t2" in [target.id for target in scenario.targets]
    assert after_inject.target_active_mask.astype(int).tolist().count(1) >= 2
