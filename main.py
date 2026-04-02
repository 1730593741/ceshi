"""运行可配置的闭环 NSGA-II 实验."""

from __future__ import annotations

import hashlib
import json
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from eval.metrics import igd, igd_plus, spacing, spread
from eval.reference_front import (
    ReferenceFront,
    build_intra_run_reference_front,
    build_true_reference_front,
)

import yaml
import math

from pydantic import BaseModel, Field, model_validator

from controller.closed_loop import (
    ClosedLoopRunner,
    HybridController,
    LLMChainController,
    RewardConfig,
    RuleBasedController,
    RuleControllerConfig,
)
from controller.operator_space import OperatorParams
from infra.llm_client import LLMClient, LLMClientConfig
from infra.storage import ExperienceJsonlLogger, JsonlLogger
from llm.actuator import Actuator
from llm.analyst import Analyst
from llm.strategist import Strategist
from memory.experience_pool import ExperiencePool
from experiments.logging import split_event_stream
from optimizers.nsga2.solver import NSGA2Config, NSGA2Solver
from src.dwta.model import (
    DWTABenchmarkData,
    DWTAWaveEvent,
    MunitionType as DWTAMunitionType,
    Target as DWTATarget,
    Weapon as DWTAWeapon,
)
from src.dwta.live_cache import DWTALiveCache
from src.dwta.scenario_builder import build_dynamic_scenario
from sensing.pareto_state import ParetoStateSensor


class DWTAMunitionConfig(BaseModel):
    """DWTA Munition 配置项."""

    id: str
    max_range: float
    flight_speed: float
    lethality: float

    @model_validator(mode="after")
    def _validate_non_negative(self) -> "DWTAMunitionConfig":
        if not math.isfinite(self.max_range) or self.max_range < 0:
            raise ValueError("munition max_range must be finite and >= 0")
        if not math.isfinite(self.flight_speed) or self.flight_speed <= 0:
            raise ValueError("munition flight_speed must be finite and > 0")
        if not math.isfinite(self.lethality) or self.lethality < 0:
            raise ValueError("munition lethality must be finite and >= 0")
        return self


class DWTAWeaponConfig(BaseModel):
    """DWTA Weapon 配置项."""

    id: str
    x: float
    y: float
    munition_type_id: str
    ammo_capacity: int

    @model_validator(mode="after")
    def _validate_weapon(self) -> "DWTAWeaponConfig":
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("weapon coordinates must be finite numbers")
        if self.ammo_capacity < 0:
            raise ValueError("weapon ammo_capacity must be >= 0")
        return self


class DWTATargetConfig(BaseModel):
    """DWTA Target 配置项."""

    id: str
    x: float
    y: float
    required_damage: float
    time_window: list[float]

    @model_validator(mode="after")
    def _validate_target(self) -> "DWTATargetConfig":
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("target coordinates must be finite numbers")
        if not math.isfinite(self.required_damage) or self.required_damage < 0:
            raise ValueError("target required_damage must be finite and >= 0")
        if len(self.time_window) != 2:
            raise ValueError("target time_window must contain [start, end]")
        if not all(math.isfinite(value) for value in self.time_window):
            raise ValueError("target time_window values must be finite")
        if float(self.time_window[0]) > float(self.time_window[1]):
            raise ValueError("target time_window start must be <= end")
        return self


class DWTAPrecomputedConfig(BaseModel):
    """DWTA precomputed 矩阵 payload 用于 direct 运行时 loading."""

    ammo_capacities: list[int]
    compatibility_matrix: list[list[int]]
    lethality_matrix: list[list[float]]
    required_damage: list[float]

    @model_validator(mode="after")
    def _validate_shapes(self) -> "DWTAPrecomputedConfig":
        n_weapons = len(self.ammo_capacities)
        if n_weapons == 0:
            raise ValueError("dwta.precomputed.ammo_capacities must not be empty")
        if len(self.compatibility_matrix) != n_weapons:
            raise ValueError("dwta.precomputed.compatibility_matrix row count must equal ammo_capacities length")
        if len(self.lethality_matrix) != n_weapons:
            raise ValueError("dwta.precomputed.lethality_matrix row count must equal ammo_capacities length")

        n_targets = len(self.required_damage)
        if n_targets == 0:
            raise ValueError("dwta.precomputed.required_damage must not be empty")

        for row in self.compatibility_matrix:
            if len(row) != n_targets:
                raise ValueError("dwta.precomputed.compatibility_matrix column count must equal required_damage length")
            if any(value not in (0, 1) for value in row):
                raise ValueError("dwta.precomputed.compatibility_matrix values must be 0 or 1")
        for row in self.lethality_matrix:
            if len(row) != n_targets:
                raise ValueError("dwta.precomputed.lethality_matrix column count must equal required_damage length")
            if any((not math.isfinite(value)) or value < 0 for value in row):
                raise ValueError("dwta.precomputed.lethality_matrix values must be finite and >= 0")
        if any(capacity < 0 for capacity in self.ammo_capacities):
            raise ValueError("dwta.precomputed.ammo_capacities values must be >= 0")
        if any((not math.isfinite(damage)) or damage < 0 for damage in self.required_damage):
            raise ValueError("dwta.precomputed.required_damage values must be finite and >= 0")
        return self


class DWTAWaveEventConfig(BaseModel):
    """DWTA 脚本波次配置项（支持运行期事件注入）."""

    wave_id: str
    trigger_generation: int = 0
    event_type: str = "legacy_target_damage_scale"
    payload: dict[str, Any] = Field(default_factory=dict)
    target_damage_scale: float | None = None
    compatibility_override: list[list[int]] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _validate_wave(self) -> "DWTAWaveEventConfig":
        supported_event_types = {
            "activate_targets",
            "inject_targets",
            "disable_weapons",
            "ammo_delta",
            "target_priority_update",
            "time_window_update",
            "legacy_target_damage_scale",
        }
        if self.trigger_generation < 0:
            raise ValueError("dwta.waves.trigger_generation must be >= 0")
        if self.event_type not in supported_event_types:
            raise ValueError(f"dwta.waves.event_type must be one of {sorted(supported_event_types)}")
        if self.target_damage_scale is not None and (
            (not math.isfinite(self.target_damage_scale)) or self.target_damage_scale <= 0
        ):
            raise ValueError("dwta.waves.target_damage_scale must be finite and > 0 when provided")
        if self.compatibility_override is not None:
            for row in self.compatibility_override:
                if any(value not in (0, 1) for value in row):
                    raise ValueError("dwta.waves.compatibility_override values must be 0 or 1")
        return self


class ProblemConfig(BaseModel):
    """问题规格 用于 任务-分配 与 DWTA 基准问题."""

    n_tasks: int = 0
    n_resources: int = 1
    cost_matrix: list[list[float]] = Field(default_factory=list)
    task_loads: list[float] = Field(default_factory=list)
    capacities: list[float] = Field(default_factory=list)
    task_time_windows: list[list[float]] | None = None
    resource_time_windows: list[list[float]] | None = None
    compatibility_matrix: list[list[int]] | None = None
    resource_stage_levels: list[int] | None = None
    stage_transitions: list[list[int]] | None = None

    munition_types: list[DWTAMunitionConfig] | None = None
    munitions: list[DWTAMunitionConfig] | None = None
    weapons: list[DWTAWeaponConfig] | None = None
    targets: list[DWTATargetConfig] | None = None
    precomputed: DWTAPrecomputedConfig | None = None
    scenario_mode: Literal["static", "scripted_waves"] = "static"
    max_weapons: int | None = None
    max_targets: int | None = None
    waves: list[DWTAWaveEventConfig] = Field(default_factory=list)

    @property
    def problem_type(self) -> Literal["dwta"]:
        """当前仓库仅支持 DWTA 问题类型。"""
        return "dwta"

    @property
    def is_dwta(self) -> bool:
        return any(
            value
            for value in (
                self.precomputed is not None,
                bool(self.resolved_munition_types),
                bool(self.weapons),
                bool(self.targets),
                bool(self.waves),
                self.scenario_mode != "static",
                self.max_weapons is not None,
                self.max_targets is not None,
            )
        )

    @property
    def resolved_munition_types(self) -> list[DWTAMunitionConfig]:
        """返回 DWTA Munition list，并带有 向后兼容 用于 old key names."""
        return self.munition_types or self.munitions or []

    @model_validator(mode="after")
    def _validate_shapes(self) -> "ProblemConfig":
        if self.is_dwta:
            if self.scenario_mode not in {"static", "scripted_waves"}:
                raise ValueError("dwta.scenario_mode must be one of static/scripted_waves")
            if self.max_weapons is not None and self.max_weapons <= 0:
                raise ValueError("dwta.max_weapons must be > 0 when provided")
            if self.max_targets is not None and self.max_targets <= 0:
                raise ValueError("dwta.max_targets must be > 0 when provided")
            if self.scenario_mode == "static" and self.waves:
                raise ValueError("dwta.waves requires scenario_mode=scripted_waves")
            if self.scenario_mode == "scripted_waves":
                if self.max_weapons is None or self.max_targets is None:
                    raise ValueError("scripted_waves mode requires max_weapons and max_targets for fixed canvas")
            if self.precomputed is not None:
                return self
            if not self.resolved_munition_types or not self.weapons or not self.targets:
                raise ValueError("dwta requires either precomputed matrices or munition_types (or munitions) + weapons + targets")
            munition_ids = {munition.id for munition in self.resolved_munition_types}
            if len(munition_ids) != len(self.resolved_munition_types):
                raise ValueError("munition type ids must be unique")
            weapon_ids = {weapon.id for weapon in self.weapons}
            if len(weapon_ids) != len(self.weapons):
                raise ValueError("weapon ids must be unique")
            target_ids = {target.id for target in self.targets}
            if len(target_ids) != len(self.targets):
                raise ValueError("target ids must be unique")
            for weapon in self.weapons:
                if weapon.munition_type_id not in munition_ids:
                    raise ValueError("weapon.munition_type_id must reference a defined munition")
            if self.scenario_mode == "scripted_waves":
                if self.max_weapons is not None and self.max_weapons < len(self.weapons):
                    raise ValueError("dwta.max_weapons must be >= number of configured weapons")
                if self.max_targets is not None and self.max_targets < len(self.targets):
                    raise ValueError("dwta.max_targets must be >= number of configured targets")
            return self

        raise ValueError(
            "non-DWTA problem definitions were removed; configure dwta.precomputed or munition_types+weapons+targets"
        )


class MemoryConfig(BaseModel):
    """M5 记忆 与 reward-related configuration."""

    enabled: bool = True
    memory_window: int = 100
    experience_log_path: str | None = None
    reward_alpha: float = 1.0
    reward_beta: float = 0.1


class ControllerModeConfig(BaseModel):
    """Controller 模式 switch 与 模式-specific options."""

    mode: Literal["rule", "mock_llm", "real_llm", "hybrid_llm"] = "rule"
    experience_lookback: int = 5


class LLMRuntimeConfig(BaseModel):
    """Config schema 用于 LLM 运行时 backend."""

    provider: str = "openai"
    model: str = "qwen3-max"
    timeout_s: float = 60.0
    min_read_timeout_s: float = 60.0
    max_retries: int = 2
    # Environment variable name for API key; never hardcode secrets here.
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model_env: str = "OPENAI_MODEL"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    fallback_mode: Literal["mock_llm", "hold"] = "mock_llm"


class LoggingConfig(BaseModel):
    """Output layout 用于 可复现的 实验 artifacts."""

    output_dir: str = "experiments/logs/default"
    events_file: str = "events.jsonl"
    experiences_file: str = "experiences.jsonl"
    summary_file: str = "summary.json"
    config_snapshot_file: str = "config_snapshot.yaml"
    generation_log_file: str = "generation_metrics.jsonl"
    action_log_file: str = "actions.jsonl"


class ExperimentMetaConfig(BaseModel):
    """Lightweight 元数据 persisted 转换为 摘要 输出."""

    name: str = "default"
    seed: int | None = None
    method: str | None = None
    benchmark: str | None = None


class EvaluationConfig(BaseModel):
    """Evaluation settings 用于 论文级 指标 与 reference-前沿 provenance."""

    reference_front_mode: Literal["auto", "true_front_file", "intra_run"] = "auto"
    true_pareto_front_path: str | None = None


class AblationConfig(BaseModel):
    """用于配置观测/历史/动作维度消融开关."""

    disabled_observations: list[str] = Field(default_factory=list)
    disable_history: bool = False
    disabled_action_dimensions: list[str] = Field(default_factory=list)


class ControllerConfig(BaseModel):
    """实验导向控制配置：统一 rule/llm/memory 参数."""

    rule: RuleControllerConfig = Field(default_factory=RuleControllerConfig)
    mode: Literal["rule", "mock_llm", "real_llm", "hybrid_llm"] = "rule"
    experience_lookback: int = 5
    memory_enabled: bool = True
    memory_window: int = 100
    experience_log_path: str | None = None
    reward_alpha: float = 1.0
    reward_beta: float = 0.1
    # Hybrid-specific: stagnation threshold to wake up LLM controller
    hybrid_stagnation_llm_threshold: int = 5

    # NOTE: __getattr__ / __setattr__ overrides are intentionally omitted.
    # In Pydantic v2, overriding these magic methods on a BaseModel subclass
    # is unsafe: __getattr__ is called during model initialisation (before
    # fields are populated) and can cause infinite recursion or AttributeError
    # when the delegated field (self.rule) is not yet set.
    #
    # Rule-controller fields must be accessed via ``config.controller.rule.<field>``
    # directly. The RuleControllerConfig dataclass is always fully constructed
    # before ControllerConfig validation completes, so direct attribute access
    # on ``self.rule`` is always safe after model_validate() returns.


class ExperimentConfig(BaseModel):
    """从 YAML 解析得到的顶层实验配置."""

    solver: NSGA2Config
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    llm: LLMRuntimeConfig = Field(default_factory=LLMRuntimeConfig)
    problem: ProblemConfig
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentMetaConfig = Field(default_factory=ExperimentMetaConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    log_path: str | None = None

    @property
    def optimizer(self) -> NSGA2Config:
        """向后兼容：旧代码仍可读取 cfg.optimizer."""
        return self.solver

    @property
    def controller_mode(self) -> "_ControllerModeProxy":
        """向后兼容：旧代码仍可读取 cfg.controller_mode."""
        return _ControllerModeProxy(self.controller)

    @property
    def memory(self) -> "_MemoryProxy":
        """向后兼容：旧代码仍可读取 cfg.memory."""
        return _MemoryProxy(self.controller)


class _ControllerModeProxy:
    def __init__(self, controller: ControllerConfig) -> None:
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name: str) -> Any:
        mapping = {"mode": "mode", "experience_lookback": "experience_lookback"}
        if name not in mapping:
            raise AttributeError(name)
        return getattr(self._controller, mapping[name])

    def __setattr__(self, name: str, value: Any) -> None:
        mapping = {"mode": "mode", "experience_lookback": "experience_lookback"}
        if name not in mapping:
            raise AttributeError(name)
        setattr(self._controller, mapping[name], value)


class _MemoryProxy:
    def __init__(self, controller: ControllerConfig) -> None:
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name: str) -> Any:
        mapping = {
            "enabled": "memory_enabled",
            "memory_window": "memory_window",
            "experience_log_path": "experience_log_path",
            "reward_alpha": "reward_alpha",
            "reward_beta": "reward_beta",
        }
        if name not in mapping:
            raise AttributeError(name)
        return getattr(self._controller, mapping[name])

    def __setattr__(self, name: str, value: Any) -> None:
        mapping = {
            "enabled": "memory_enabled",
            "memory_window": "memory_window",
            "experience_log_path": "experience_log_path",
            "reward_alpha": "reward_alpha",
            "reward_beta": "reward_beta",
        }
        if name not in mapping:
            raise AttributeError(name)
        setattr(self._controller, mapping[name], value)


@dataclass(slots=True)
class RunArtifacts:
    """已解析的文件路径 用于 一个 concrete 运行."""

    output_dir: Path
    events_path: Path
    experiences_path: Path | None
    summary_path: Path
    config_snapshot_path: Path
    generation_log_path: Path
    action_log_path: Path


@dataclass(slots=True)
class RuntimeBundle:
    """组装好的运行时组件 用于 main 与 基线 wrappers."""

    config: ExperimentConfig
    solver: NSGA2Solver
    runner: ClosedLoopRunner
    artifacts: RunArtifacts


def load_config(path: str | Path) -> ExperimentConfig:
    """从 YAML 文件加载实验配置."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "solver" not in raw and "optimizer" in raw:
        raw["solver"] = raw.pop("optimizer")

    controller_raw = dict(raw.get("controller") or {})
    legacy_mode = raw.pop("controller_mode", None)
    if isinstance(legacy_mode, dict):
        controller_raw.setdefault("mode", legacy_mode.get("mode", "rule"))
        controller_raw.setdefault("experience_lookback", legacy_mode.get("experience_lookback", 5))
    legacy_memory = raw.pop("memory", None)
    if isinstance(legacy_memory, dict):
        controller_raw.setdefault("memory_enabled", legacy_memory.get("enabled", True))
        controller_raw.setdefault("memory_window", legacy_memory.get("memory_window", 100))
        controller_raw.setdefault("experience_log_path", legacy_memory.get("experience_log_path"))
        controller_raw.setdefault("reward_alpha", legacy_memory.get("reward_alpha", 1.0))
        controller_raw.setdefault("reward_beta", legacy_memory.get("reward_beta", 0.1))
    rule_fields = set(RuleControllerConfig.__dataclass_fields__.keys())
    nested_rule = dict(controller_raw.get("rule") or {})
    for key in list(controller_raw.keys()):
        if key in rule_fields:
            nested_rule.setdefault(key, controller_raw.pop(key))
    controller_raw["rule"] = nested_rule
    raw["controller"] = controller_raw

    return ExperimentConfig.model_validate(raw)


def build_solver(problem: ProblemConfig, optimizer: NSGA2Config) -> NSGA2Solver:
    """从结构化配置创建 NSGA-II 求解器."""
    if problem.is_dwta:
        dwta_live_cache: DWTALiveCache | None = None
        if problem.precomputed is not None:
            dwta_data = DWTABenchmarkData(
                n_weapons=len(problem.precomputed.ammo_capacities),
                n_targets=len(problem.precomputed.required_damage),
                ammo_capacities=problem.precomputed.ammo_capacities,
                compatibility_matrix=problem.precomputed.compatibility_matrix,
                lethality_matrix=problem.precomputed.lethality_matrix,
                required_damage=problem.precomputed.required_damage,
            )
        else:
            scenario = build_dynamic_scenario(
                [
                    DWTAMunitionType(
                        id=item.id,
                        max_range=item.max_range,
                        flight_speed=item.flight_speed,
                        lethality=item.lethality,
                    )
                    for item in problem.resolved_munition_types
                ],
                [
                    DWTAWeapon(
                        id=item.id,
                        x=item.x,
                        y=item.y,
                        munition_type_id=item.munition_type_id,
                        ammo_capacity=item.ammo_capacity,
                    )
                    for item in problem.weapons or []
                ],
                [
                    DWTATarget(
                        id=item.id,
                        x=item.x,
                        y=item.y,
                        required_damage=item.required_damage,
                        time_window=(float(item.time_window[0]), float(item.time_window[1])),
                    )
                    for item in problem.targets or []
                ],
                scenario_mode=problem.scenario_mode,
                max_weapons=problem.max_weapons,
                max_targets=problem.max_targets,
                waves=[
                    DWTAWaveEvent(
                        wave_id=item.wave_id,
                        trigger_generation=item.trigger_generation,
                        event_type=item.event_type,
                        payload=item.payload,
                        target_damage_scale=item.target_damage_scale,
                        compatibility_override=item.compatibility_override,
                        note=item.note,
                    )
                    for item in problem.waves
                ],
            )
            dwta_live_cache = DWTALiveCache(scenario)
            dwta_live_cache.refresh()
            dwta_data = scenario.base_data
        return NSGA2Solver(
            n_tasks=0,
            n_resources=1,
            cost_matrix=[],
            task_loads=[],
            capacities=[],
            config=optimizer,
            dwta_data=dwta_data,
            dwta_live_cache=dwta_live_cache,
        )

    raise ValueError(
        "non-DWTA solver mode was removed; provide DWTA matrices or DWTA scenario entities in config.problem"
    )


def build_controller(config: ExperimentConfig) -> RuleBasedController | LLMChainController | HybridController:
    """按模式构建规则、LLM 或混合控制器，而不改变运行器 API."""
    mode = config.controller.mode
    rule = config.controller.rule
    if mode == "rule":
        return RuleBasedController(rule)

    def _build_llm_chain(llm_mode: str = "real_llm") -> LLMChainController:
        """内部辅助：构建 LLMChainController 实例."""
        llm_client = LLMClient(
            LLMClientConfig(
                mode=llm_mode,
                provider=config.llm.provider,
                model=config.llm.model,
                timeout_s=config.llm.timeout_s,
                min_read_timeout_s=getattr(config.llm, "min_read_timeout_s", config.llm.timeout_s),
                max_retries=config.llm.max_retries,
                api_key_env=config.llm.api_key_env,
                base_url_env=config.llm.base_url_env,
                model_env=config.llm.model_env,
                base_url=config.llm.base_url,
                fallback_mode=config.llm.fallback_mode,
            )
        )
        analyst = Analyst(llm_client)
        actuator = Actuator(
            llm_client,
            min_mutation_prob=rule.min_mutation_prob,
            max_mutation_prob=rule.max_mutation_prob,
            min_crossover_prob=rule.min_crossover_prob,
            max_crossover_prob=rule.max_crossover_prob,
            min_eta_c=rule.min_eta_c,
            max_eta_c=rule.max_eta_c,
            min_eta_m=rule.min_eta_m,
            max_eta_m=rule.max_eta_m,
            min_repair_prob=rule.min_repair_prob,
            max_repair_prob=rule.max_repair_prob,
            min_local_search_prob=rule.min_local_search_prob,
            max_local_search_prob=rule.max_local_search_prob,
        )
        return LLMChainController(
            control_interval=rule.control_interval,
            experience_lookback=config.controller.experience_lookback,
            event_triggered_control=rule.event_triggered_control,
            event_control_cooldown=rule.event_control_cooldown,
            forced_control_on_major_event=rule.forced_control_on_major_event,
            analyst=analyst,
            actuator=actuator,
        )

    if mode == "hybrid_llm":
        rule_ctrl = RuleBasedController(rule)
        llm_ctrl = _build_llm_chain("real_llm")
        return HybridController(
            rule_controller=rule_ctrl,
            llm_controller=llm_ctrl,
            control_interval=rule.control_interval,
            stagnation_llm_threshold=config.controller.hybrid_stagnation_llm_threshold,
            event_triggered_control=rule.event_triggered_control,
            event_control_cooldown=rule.event_control_cooldown,
            forced_control_on_major_event=rule.forced_control_on_major_event,
        )

    return _build_llm_chain(mode)


def resolve_artifacts(config: ExperimentConfig) -> RunArtifacts:
    """为一次运行解析并创建输出路径."""
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = Path(config.log_path) if config.log_path else output_dir / config.logging.events_file
    experiences_path: Path | None = None
    if config.controller.memory_enabled:
        if config.controller.experience_log_path:
            experiences_path = Path(config.controller.experience_log_path)
        else:
            experiences_path = output_dir / config.logging.experiences_file

    return RunArtifacts(
        output_dir=output_dir,
        events_path=events_path,
        experiences_path=experiences_path,
        summary_path=output_dir / config.logging.summary_file,
        config_snapshot_path=output_dir / config.logging.config_snapshot_file,
        generation_log_path=output_dir / config.logging.generation_log_file,
        action_log_path=output_dir / config.logging.action_log_file,
    )


def build_runtime(config: ExperimentConfig) -> RuntimeBundle:
    """组装 求解器、控制器、运行器 与输出产物."""
    artifacts = resolve_artifacts(config)
    _reset_run_logs(artifacts)
    solver = build_solver(config.problem, config.solver)
    sensor = ParetoStateSensor()
    controller = build_controller(config)
    logger = JsonlLogger(artifacts.events_path)

    experience_pool = ExperiencePool(config.controller.memory_window) if config.controller.memory_enabled else None
    experience_logger = ExperienceJsonlLogger(artifacts.experiences_path) if artifacts.experiences_path else None
    reward_config = RewardConfig(alpha=config.controller.reward_alpha, beta=config.controller.reward_beta)

    runner = ClosedLoopRunner(
        solver=solver,
        sensor=sensor,
        controller=controller,
        logger=logger,
        experience_pool=experience_pool,
        experience_logger=experience_logger,
        reward_config=reward_config,
        stagnation_tolerance=config.controller.rule.improvement_threshold,
    )
    return RuntimeBundle(config=config, solver=solver, runner=runner, artifacts=artifacts)




def _config_identity(config: ExperimentConfig, config_path: str | Path) -> tuple[str, str]:
    """根据配置快照输入构建确定性的指纹与运行 ID."""
    material = {
        "source_config_path": str(config_path),
        "resolved_config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    run_id = f"run-{fingerprint[:12]}"
    return fingerprint, run_id


def _write_config_snapshot(path: Path, config: ExperimentConfig, config_path: str | Path, *, config_fingerprint: str, run_id: str) -> None:
    payload = {
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "source_config_path": str(config_path),
        "resolved_config": config.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)






def _reset_run_logs(artifacts: RunArtifacts) -> None:
    """重置追加日志，使每次运行的 指标/动作 计数彼此隔离."""
    for path in [
        artifacts.events_path,
        artifacts.generation_log_path,
        artifacts.action_log_path,
        artifacts.experiences_path,
    ]:
        if path is not None and path.exists():
            path.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _count_control_states(action_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in action_events:
        state = event.get("control_state")
        if not isinstance(state, str):
            continue
        counts[state] = counts.get(state, 0) + 1
    return counts


def _build_dynamic_summary(
    *,
    config: ExperimentConfig,
    generation_events: list[dict[str, Any]],
    action_events: list[dict[str, Any]],
    runtime_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build dynamic DWTA metrics while preserving legacy summary fields."""
    event_types_seen = sorted(
        {
            str(event.get("event_type"))
            for event in runtime_events
            if isinstance(event.get("event_type"), str) and str(event.get("event_type"))
        }
    )
    num_events = len(runtime_events)
    event_triggered_actions = sum(1 for event in action_events if event.get("trigger_type") == "event")

    # Aggregate generation-level proxies for dynamic pressure/consumption.
    # cumulative_unmet_damage: sum of remaining_survivability (objective 0) across all generations.
    # cumulative_resource_consumption: weapons lost relative to generation-0 baseline.
    cumulative_unmet_damage = 0.0
    cumulative_resource_consumption = 0.0
    if generation_events:
        baseline_active_weapons = generation_events[0].get("active_weapons_count")
        if isinstance(baseline_active_weapons, (int, float)):
            baseline = float(baseline_active_weapons)
        else:
            baseline = 0.0
        for event in generation_events:
            # Use the MINIMUM of the first objective (remaining_survivability) as the unmet-damage proxy.
            # This ensures stability across different Pareto front orderings.
            rank1_objs = event.get("rank1_objectives")
            if isinstance(rank1_objs, list) and rank1_objs:
                all_survivability = [
                    float(point[0])
                    for point in rank1_objs
                    if isinstance(point, (list, tuple)) and len(point) >= 1
                ]
                if all_survivability:
                    cumulative_unmet_damage += min(all_survivability)
            active_weapons = event.get("active_weapons_count")
            if isinstance(active_weapons, (int, float)):
                cumulative_resource_consumption += max(0.0, baseline - float(active_weapons))

    total_waves = len(config.problem.waves)
    wave_completion_rate = (float(num_events) / float(total_waves)) if total_waves > 0 else 0.0

    post_event_recovery = _post_event_recovery_generations_mean(generation_events, runtime_events)

    return {
        "num_events": num_events,
        "event_types_seen": event_types_seen,
        "post_event_recovery_generations_mean": post_event_recovery,
        "cumulative_unmet_damage": cumulative_unmet_damage,
        "cumulative_resource_consumption": cumulative_resource_consumption,
        "wave_completion_rate": wave_completion_rate,
        "event_triggered_actions": event_triggered_actions,
    }


def _post_event_recovery_generations_mean(
    generation_events: list[dict[str, Any]], runtime_events: list[dict[str, Any]]
) -> float:
    """Estimate mean generations needed to re-enter positive-HV progress after events."""
    if not generation_events or not runtime_events:
        return 0.0
    by_generation: dict[int, dict[str, Any]] = {}
    for event in generation_events:
        generation = event.get("generation")
        if isinstance(generation, int):
            by_generation[generation] = event

    horizons: list[float] = []
    max_generation = max(by_generation.keys(), default=0)
    for event in runtime_events:
        g0 = event.get("generation")
        if not isinstance(g0, int):
            continue
        steps: float | None = None
        for gen in range(g0 + 1, max_generation + 1):
            candidate = by_generation.get(gen)
            if candidate is None:
                continue
            delta_hv = candidate.get("delta_hv")
            if isinstance(delta_hv, (int, float)) and float(delta_hv) > 0.0:
                steps = float(gen - g0)
                break
        if steps is None:
            steps = float(max(0, max_generation - g0))
        horizons.append(steps)

    if not horizons:
        return 0.0
    return float(sum(horizons) / len(horizons))


def _load_true_reference_front(path: Path) -> list[tuple[float, ...]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("true pareto front file must be a JSON list of points")
    points: list[tuple[float, ...]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("each true front point must be a list")
        points.append(tuple(float(v) for v in row))
    return points


def _resolve_reference_front(config: ExperimentConfig, generation_events: list[dict[str, Any]]) -> ReferenceFront:
    mode = config.evaluation.reference_front_mode
    true_path = config.evaluation.true_pareto_front_path
    if mode in {"auto", "true_front_file"} and true_path:
        path = Path(true_path)
        return build_true_reference_front(_load_true_reference_front(path), name=str(path))
    return build_intra_run_reference_front(generation_events)


def run_experiment(config_path: str = "experiments/configs/default.yaml") -> dict[str, Any]:
    """运行一次实验并持久化快照与摘要输出."""
    config = load_config(config_path)
    runtime = build_runtime(config)
    try:
        config_fingerprint, run_id = _config_identity(config, config_path)
        _write_config_snapshot(
            runtime.artifacts.config_snapshot_path,
            config,
            config_path,
            config_fingerprint=config_fingerprint,
            run_id=run_id,
        )

        start_ts = time.perf_counter()
        states = runtime.runner.run(generations=config.solver.generations)
        runtime_s = time.perf_counter() - start_ts
        split_event_stream(
            events_path=runtime.artifacts.events_path,
            state_log_path=runtime.artifacts.generation_log_path,
            action_log_path=runtime.artifacts.action_log_path,
        )

        final = states[-1]
        best_state = max(states, key=lambda state: state.hv)
        hv_auc = sum(state.hv for state in states) / len(states) if states else 0.0

        generation_events = _read_jsonl(runtime.artifacts.generation_log_path)
        action_events = _read_jsonl(runtime.artifacts.action_log_path)
        runtime_events = [
            event for event in _read_jsonl(runtime.artifacts.events_path) if event.get("event") == "runtime_event"
        ]
        final_generation_event = generation_events[-1] if generation_events else {}
        num_experiences = 0
        if runtime.artifacts.experiences_path and runtime.artifacts.experiences_path.exists():
            num_experiences = len(_read_jsonl(runtime.artifacts.experiences_path))

        reference_front = _resolve_reference_front(config, generation_events)
        final_front_raw = final_generation_event.get("rank1_objectives", final.rank1_objectives)
        final_front = [tuple(float(v) for v in point) for point in final_front_raw] if isinstance(final_front_raw, list) else []

        final_igd = igd(final_front, reference_front.points)
        final_igd_plus = igd_plus(final_front, reference_front.points)
        final_spacing = spacing(final_front)
        final_spread = spread(final_front, reference_front.points) if final_front and reference_front.points else 0.0

        llm_overhead_s = sum(
            float(event.get("decision_runtime_s", 0.0))
            for event in action_events
            if event.get("is_llm_decision")
        )
        dynamic_summary = _build_dynamic_summary(
            config=config,
            generation_events=generation_events,
            action_events=action_events,
            runtime_events=runtime_events,
        )

        summary = {
            "experiment": config.experiment.model_dump(mode="json"),
            "controller_mode": config.controller.mode,
            "method": config.experiment.method or config.experiment.name,
            "benchmark": config.experiment.benchmark or "unknown",
            "seed": config.experiment.seed if config.experiment.seed is not None else config.solver.seed,
            "source_config_path": str(config_path),
            "run_id": run_id,
            "config_fingerprint": config_fingerprint,
            "generations": config.solver.generations,
            "final_generation": final.generation,
            "final_hv": final.hv,
            "best_hv": best_state.hv,
            "best_generation": best_state.generation,
            "hv_auc": hv_auc,
            "mean_hv": hv_auc,
            "final_feasible_ratio": final.feasible_ratio,
            "final_rank1_ratio": final.rank1_ratio,
            "final_igd": final_igd,
            "final_igd_plus": final_igd_plus,
            "final_spacing": final_spacing,
            "final_spread": final_spread,
            "reference_front": {
                "source": reference_front.source,
                "is_comparable": reference_front.source in {"true_pareto_front", "empirical_matched_runs"},
                "details": reference_front.details,
                "num_points": len(reference_front.points),
            },
            "final_mutation_prob": final_generation_event.get("mutation_prob", runtime.solver.config.mutation_prob),
            "final_crossover_prob": final_generation_event.get("crossover_prob", runtime.solver.config.crossover_prob),
            "final_operator_params": (
                final_generation_event.get("operator_params")
                or runtime.solver.get_operator_params().to_dict()
            ),
            "final_effective_params": (
                OperatorParams(**(final_generation_event.get("operator_params") or runtime.solver.get_operator_params().to_dict()))
                .active_params(runtime.solver.get_operator_capabilities())
            ),
            "operator_capabilities": runtime.solver.get_operator_capabilities().to_dict(),
            "events_path": str(runtime.artifacts.events_path),
            "experiences_path": str(runtime.artifacts.experiences_path) if runtime.artifacts.experiences_path else None,
            "num_actions": len(action_events),
            "runtime_s": runtime_s,
            "llm_overhead_s": llm_overhead_s,
            "control_state_counts": _count_control_states(action_events),
            "num_experiences": num_experiences,
            # Dynamic metrics are nested to keep legacy top-level summary schema stable.
            "dynamic_summary": dynamic_summary,
            "config_snapshot_path": str(runtime.artifacts.config_snapshot_path),
            "generation_log_path": str(runtime.artifacts.generation_log_path),
            "action_log_path": str(runtime.artifacts.action_log_path),
        }
        _write_summary(runtime.artifacts.summary_path, summary)
        summary["summary_path"] = str(runtime.artifacts.summary_path)
        return summary
    finally:
        runtime.runner.logger.close()
        if runtime.runner.experience_logger:
            runtime.runner.experience_logger.close()


def main(config_path: str = "experiments/configs/default.yaml") -> None:
    """运行闭环并打印最终摘要."""
    summary = run_experiment(config_path)
    print(
        "Run complete | "
        f"generation={summary['final_generation']} hv={summary['final_hv']:.4f} "
        f"mutation_prob={summary['final_mutation_prob']:.3f} "
        f"crossover_prob={summary['final_crossover_prob']:.3f} "
        f"log_path={summary['events_path']}"
    )


def parse_cli_args(argv: list[str] | None = None) -> str:
    """解析 CLI 参数并返回解析后的配置路径."""
    parser = argparse.ArgumentParser(description="Run closed-loop NSGA-II experiment")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to YAML experiment config")
    parser.add_argument("--config", dest="config_flag", default=None, help="Path to YAML experiment config")
    args = parser.parse_args(argv)
    return args.config_flag or args.config_path or "experiments/configs/default.yaml"


if __name__ == "__main__":
    main(parse_cli_args())
