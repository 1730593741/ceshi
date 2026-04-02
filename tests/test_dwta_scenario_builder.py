"""用于测试 DWTA scenario setup-time preprocessing 与 schema validation."""

from __future__ import annotations

import pytest

from main import ProblemConfig
from src.dwta.model import DWTAWaveEvent, MunitionType, Target, Weapon
from src.dwta.scenario_builder import build_dynamic_scenario, build_scenario_matrices


def test_scenario_builder_compatibility_and_lethality_correctness() -> None:
    data = build_scenario_matrices(
        munition_types=[MunitionType(id="m1", max_range=10.0, flight_speed=2.0, lethality=3.5)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=5)],
        targets=[
            Target(id="t1", x=6.0, y=8.0, required_damage=3.0, time_window=(4.0, 6.0)),  # 距离=10, flight=5
            Target(id="t2", x=3.0, y=4.0, required_damage=2.0, time_window=(1.0, 3.0)),  # 距离=5, flight=2.5
        ],
    )

    assert data.compatibility_matrix == [[1, 1]]
    assert data.lethality_matrix == [[3.5, 3.5]]


def test_scenario_builder_marks_out_of_range_as_incompatible() -> None:
    data = build_scenario_matrices(
        munition_types=[MunitionType(id="m1", max_range=4.9, flight_speed=10.0, lethality=1.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=1)],
        targets=[Target(id="t1", x=3.0, y=4.0, required_damage=1.0, time_window=(0.0, 10.0))],
    )

    assert data.compatibility_matrix == [[0]]


def test_scenario_builder_marks_time_window_mismatch_as_incompatible() -> None:
    data = build_scenario_matrices(
        munition_types=[MunitionType(id="m1", max_range=100.0, flight_speed=1.0, lethality=1.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=1)],
        targets=[Target(id="t1", x=3.0, y=4.0, required_damage=1.0, time_window=(0.0, 4.0))],  # flight=5
    )

    assert data.compatibility_matrix == [[0]]


def test_scenario_builder_rejects_invalid_munition_reference() -> None:
    with pytest.raises(ValueError, match="unknown munition_type_id"):
        build_scenario_matrices(
            munition_types=[MunitionType(id="m1", max_range=10.0, flight_speed=2.0, lethality=1.0)],
            weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="missing", ammo_capacity=1)],
            targets=[Target(id="t1", x=1.0, y=1.0, required_damage=1.0, time_window=(0.0, 10.0))],
        )


def test_dynamic_scenario_builder_assembles_environment_and_script() -> None:
    env = build_dynamic_scenario(
        munition_types=[MunitionType(id="m1", max_range=10.0, flight_speed=2.0, lethality=2.0)],
        weapons=[Weapon(id="w1", x=0.0, y=0.0, munition_type_id="m1", ammo_capacity=2)],
        targets=[Target(id="t1", x=2.0, y=0.0, required_damage=2.0, time_window=(0.0, 2.0))],
        scenario_mode="scripted_waves",
        max_weapons=8,
        max_targets=12,
        waves=[DWTAWaveEvent(wave_id="w0", trigger_generation=0, target_damage_scale=1.0)],
    )

    assert env.scenario_mode == "scripted_waves"
    assert env.max_weapons == 8
    assert env.max_targets == 12
    assert env.script is not None
    assert len(env.script.waves) == 1
    assert env.base_data.n_weapons == 8
    assert env.base_data.n_targets == 12
    assert env.weapons[-1].ammo_capacity == 0
    assert env.targets[-1].required_damage == 0.0


def test_problem_config_accepts_new_munition_types_key() -> None:
    cfg = ProblemConfig.model_validate(
        {
            "problem_type": "dwta",
            "munition_types": [{"id": "m1", "max_range": 10.0, "flight_speed": 2.0, "lethality": 1.5}],
            "weapons": [{"id": "w1", "x": 0.0, "y": 0.0, "munition_type_id": "m1", "ammo_capacity": 2}],
            "targets": [{"id": "t1", "x": 1.0, "y": 1.0, "required_damage": 2.0, "time_window": [0.0, 2.0]}],
        }
    )

    assert cfg.problem_type == "dwta"
    assert len(cfg.resolved_munition_types) == 1


@pytest.mark.parametrize(
    "payload,error_pattern",
    [
        (
            {
                "problem_type": "dwta",
                "munition_types": [{"id": "m1", "max_range": 10.0, "flight_speed": 2.0, "lethality": 1.0}],
                "weapons": [{"id": "w1", "x": 0.0, "y": 0.0, "munition_type_id": "m1", "ammo_capacity": -1}],
                "targets": [{"id": "t1", "x": 1.0, "y": 1.0, "required_damage": 2.0, "time_window": [0.0, 2.0]}],
            },
            "ammo_capacity",
        ),
        (
            {
                "problem_type": "dwta",
                "munition_types": [{"id": "m1", "max_range": 10.0, "flight_speed": 2.0, "lethality": 1.0}],
                "weapons": [{"id": "w1", "x": "bad", "y": 0.0, "munition_type_id": "m1", "ammo_capacity": 1}],
                "targets": [{"id": "t1", "x": 1.0, "y": 1.0, "required_damage": 2.0, "time_window": [0.0, 2.0]}],
            },
            "Input should be a valid number",
        ),
        (
            {
                "problem_type": "dwta",
                "munition_types": [{"id": "m1", "max_range": 10.0, "flight_speed": 2.0, "lethality": 1.0}],
                "weapons": [{"id": "w1", "x": 0.0, "y": 0.0, "munition_type_id": "m1", "ammo_capacity": 1}],
                "targets": [{"id": "t1", "x": 1.0, "y": 1.0, "required_damage": 2.0, "time_window": [3.0, 2.0]}],
            },
            "time_window",
        ),
        (
            {
                "problem_type": "dwta",
                "munition_types": [{"id": "m1", "max_range": 10.0, "flight_speed": 2.0, "lethality": 1.0}],
                "weapons": [{"id": "w1", "x": 0.0, "y": 0.0, "munition_type_id": "missing", "ammo_capacity": 1}],
                "targets": [{"id": "t1", "x": 1.0, "y": 1.0, "required_damage": 2.0, "time_window": [0.0, 2.0]}],
            },
            "munition_type_id",
        ),
    ],
)
def test_problem_config_validates_invalid_dwta_configs(payload: dict, error_pattern: str) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        ProblemConfig.model_validate(payload)
