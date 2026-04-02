"""Closed-loop orchestration for adaptive NSGA-II experiments.

This module connects four responsibilities that are intentionally kept separate
elsewhere in the repository:

1. ``NSGA2Solver`` performs search and owns the mutable operator parameters.
2. ``ParetoStateSensor`` compresses a population into a structured search state.
3. ``ControlPolicy`` implementations decide how to react to the sensed state.
4. ``ExperiencePool`` / loggers persist the state-action-reward transition chain.

The implementation here is deliberately lightweight: it does not embed problem-
specific logic or LLM transport code. Instead, it defines a stable coordination
layer that can drive the current rule-based controller and the later
analyst/strategist/actuator chain without changing the optimizer itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Protocol

from controller.control_semantics import ControlState
from controller.operator_space import OperatorCapabilities, OperatorParams
from memory.experience_pool import ExperiencePool, ExperienceRecord
from optimizers.nsga2.solver import NSGA2Solver
from sensing.pareto_state import ParetoState, ParetoStateSensor


class EventLogger(Protocol):
    """Protocol for a structured event sink used by the runner.

    Implementations typically write JSONL records, but the runner only requires a
    ``log`` method so that storage remains replaceable.
    """

    def log(self, event: dict[str, Any]) -> None:
        """Persist a single structured event payload."""


class ExperienceLogger(Protocol):
    """Protocol for optional experience persistence backends."""

    def log(self, record: ExperienceRecord) -> None:
        """Persist a single experience record."""


@dataclass(slots=True)
class RuleControllerConfig:
    """Thresholds and step sizes for the rule-based controller.

    The controller only manipulates optimizer operator parameters. Bounds are
    centralized here so that experiments can safely tune heuristics through
    configuration files instead of hardcoded magic numbers.
    """

    control_interval: int = 5
    event_triggered_control: bool = False
    event_control_cooldown: int = 0
    forced_control_on_major_event: bool = False
    min_mutation_prob: float = 0.02
    max_mutation_prob: float = 0.8
    min_crossover_prob: float = 0.4
    max_crossover_prob: float = 0.98
    min_eta_c: float = 5.0
    max_eta_c: float = 40.0
    min_eta_m: float = 5.0
    max_eta_m: float = 80.0
    min_repair_prob: float = 0.0
    max_repair_prob: float = 1.0
    min_local_search_prob: float = 0.0
    max_local_search_prob: float = 1.0
    mutation_step: float = 0.05
    crossover_step: float = 0.04
    eta_c_step: float = 2.0
    eta_m_step: float = 4.0
    repair_step: float = 0.08
    local_search_step: float = 0.05
    feasible_ratio_low: float = 0.55
    diversity_low: float = 0.08
    improvement_threshold: float = 1e-3
    # --- Bug-fix additions ---
    stagnation_threshold: int = 3  # consecutive stagnant generations before convergence push
    recovery_ratio: float = 0.5   # fraction to recover toward initial mutation when delta_hv > 0
    default_eta_c: float = 15.0   # used when eta_c is None but caps support it
    default_eta_m: float = 20.0   # used when eta_m is None but caps support it
    default_local_search_prob: float = 0.0  # used when local_search_prob is None but supported

    def __post_init__(self) -> None:
        """Validate controller bounds to protect runtime experiments."""
        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if self.event_control_cooldown < 0:
            raise ValueError("event_control_cooldown must be >= 0")
        if not 0.0 <= self.min_mutation_prob <= self.max_mutation_prob <= 1.0:
            raise ValueError("mutation probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_crossover_prob <= self.max_crossover_prob <= 1.0:
            raise ValueError("crossover probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_repair_prob <= self.max_repair_prob <= 1.0:
            raise ValueError("repair probability bounds must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.min_local_search_prob <= self.max_local_search_prob <= 1.0:
            raise ValueError("local-search probability bounds must satisfy 0 <= min <= max <= 1")


@dataclass(slots=True)
class RewardConfig:
    """Weights used when converting state transitions into scalar rewards."""

    alpha: float = 1.0
    beta: float = 0.1


@dataclass(slots=True)
class ControlAction:
    """A controller decision to be applied to optimizer operator parameters.

    ``requested_params`` captures the ideal operator setting proposed by the
    controller. ``applied_params`` stores the capability-aware subset that the
    current problem/optimizer combination actually supports.
    """

    generation: int
    mutation_prob: float
    crossover_prob: float
    reason: str
    requested_params: dict[str, float | None] | None = None
    applied_params: dict[str, float] | None = None
    capabilities: dict[str, bool] | None = None
    control_state: ControlState = ControlState.MAINTAIN_BALANCE
    reason_detail: str = ""
    decision_runtime_s: float = 0.0
    is_llm_decision: bool = False
    trigger_type: str = "periodic"
    trigger_event_id: str | None = None
    cooldown_skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action for logging and experience replay."""
        return asdict(self)


@dataclass(slots=True)
class Observation:
    """统一控制观测载体，供 rule/llm 控制器共享."""

    state: ParetoState
    recent_experiences: list[ExperienceRecord]
    current_params: OperatorParams
    capabilities: OperatorCapabilities
    trigger_type: str = "periodic"
    trigger_event_id: str | None = None


class NSGA2Adaptation:
    """将控制层与 NSGA-II solver 参数写入解耦的窄适配层."""

    def __init__(self, solver: NSGA2Solver) -> None:
        self.solver = solver

    def build_observation(
        self,
        *,
        state: ParetoState,
        recent_experiences: list[ExperienceRecord],
        trigger_type: str,
        trigger_event_id: str | None,
    ) -> Observation:
        """从求解器快照构造统一 Observation."""
        return Observation(
            state=state,
            recent_experiences=recent_experiences,
            current_params=self.solver.get_operator_params(),
            capabilities=self.solver.get_operator_capabilities(),
            trigger_type=trigger_type,
            trigger_event_id=trigger_event_id,
        )

    def apply_action(self, action: ControlAction) -> OperatorParams:
        """只通过显式 action->operator 参数映射更新求解器."""
        applied_params = action.applied_params or {}
        next_params = OperatorParams(
            mutation_prob=action.mutation_prob,
            crossover_prob=action.crossover_prob,
            eta_c=float(applied_params["eta_c"]) if "eta_c" in applied_params else None,
            eta_m=float(applied_params["eta_m"]) if "eta_m" in applied_params else None,
            repair_prob=float(applied_params["repair_prob"]) if "repair_prob" in applied_params else None,
            local_search_prob=(
                float(applied_params["local_search_prob"]) if "local_search_prob" in applied_params else None
            ),
        )
        self.solver.set_operator_params(next_params)
        return next_params


class ControlPolicy(Protocol):
    """Shared controller interface for rule-based and LLM-based policies."""

    control_interval: int

    def decide(self, *, observation: Observation) -> ControlAction:
        """Return the next control action based on a shared observation."""


class RuleBasedController:
    """Simple heuristic controller used by the MVP closed loop.

    The rule set intentionally mirrors the project specification:

    - low feasibility -> push the search toward safer, more repair-heavy moves;
    - low diversity -> increase exploration pressure;
    - stagnation / weak HV progress -> strengthen convergence pressure;
    - otherwise keep the current balance (with diversity recovery when needed).
    """

    def __init__(self, config: RuleControllerConfig) -> None:
        self.config = config
        self.control_interval = config.control_interval
        self.event_triggered_control = config.event_triggered_control
        self.event_control_cooldown = config.event_control_cooldown
        self.forced_control_on_major_event = config.forced_control_on_major_event
        # Remember the initial mutation/crossover values so we can recover toward them.
        self._initial_mutation: float | None = None
        self._initial_crossover: float | None = None

    def decide(
        self,
        *,
        observation: Observation | None = None,
        state: ParetoState | None = None,
        recent_experiences: list[ExperienceRecord] | None = None,
        current_params: OperatorParams | None = None,
        capabilities: OperatorCapabilities | None = None,
        current_mutation: float | None = None,
        current_crossover: float | None = None,
    ) -> ControlAction:
        """Derive new operator parameters from the sensed Pareto state.

        ``current_mutation`` / ``current_crossover`` are retained for backward
        compatibility with earlier tests, while ``current_params`` is the
        preferred code path used by the runner.
        """
        if observation is not None:
            state = observation.state
            recent_experiences = observation.recent_experiences
            current_params = observation.current_params
            capabilities = observation.capabilities
        if state is None:
            raise ValueError("state must be provided directly or through observation")
        del recent_experiences
        if current_params is not None:
            mutation = current_params.mutation_prob
            crossover = current_params.crossover_prob
        elif current_mutation is not None and current_crossover is not None:
            mutation = current_mutation
            crossover = current_crossover
        else:
            msg = "Either current_params or both current_mutation/current_crossover must be provided"
            raise ValueError(msg)
        capabilities = capabilities or OperatorCapabilities()

        # Snapshot initial values on first call so recovery has a target.
        if self._initial_mutation is None:
            self._initial_mutation = mutation
        if self._initial_crossover is None:
            self._initial_crossover = crossover

        control_state = ControlState.MAINTAIN_BALANCE
        reason_details: list[str] = []

        # --- Initialise optional params with defaults when supported but None ---
        eta_c = current_params.eta_c if current_params else None
        eta_m = current_params.eta_m if current_params else None
        repair_prob = current_params.repair_prob if current_params else None
        local_search_prob = current_params.local_search_prob if current_params else None

        if capabilities.supports_eta_c and eta_c is None:
            eta_c = self.config.default_eta_c
            reason_details.append("init_eta_c_default")
        if capabilities.supports_eta_m and eta_m is None:
            eta_m = self.config.default_eta_m
            reason_details.append("init_eta_m_default")
        if capabilities.supports_local_search_prob and local_search_prob is None:
            local_search_prob = self.config.default_local_search_prob
            reason_details.append("init_local_search_default")

        # First decide the high-level control intent from the observed state.
        if state.feasible_ratio < self.config.feasible_ratio_low:
            control_state = ControlState.INCREASE_FEASIBILITY
            mutation -= self.config.mutation_step * 0.5
            crossover += self.config.crossover_step
            reason_details.append("feasible_ratio_below_threshold")
        elif state.diversity_score < self.config.diversity_low:
            control_state = ControlState.INCREASE_DIVERSITY
            mutation += self.config.mutation_step
            crossover -= self.config.crossover_step
            reason_details.append("diversity_score_below_threshold")
        elif state.stagnation_len >= self.config.stagnation_threshold:
            # Bug-fix: only trigger convergence after N consecutive stagnant
            # generations, not on any single-generation dip.
            control_state = ControlState.INCREASE_CONVERGENCE
            mutation -= self.config.mutation_step * 0.5
            crossover += self.config.crossover_step * 0.5
            reason_details.append(
                f"stagnation_len({state.stagnation_len})>=threshold({self.config.stagnation_threshold})"
            )
        else:
            # Bug-fix: when maintaining balance AND mutation has been pushed below
            # its initial value, partially recover toward the initial value so that
            # exploration pressure does not monotonically decay.
            control_state = ControlState.MAINTAIN_BALANCE
            if state.delta_hv > 0 and mutation < self._initial_mutation:
                recovery_amount = (self._initial_mutation - mutation) * self.config.recovery_ratio
                mutation += recovery_amount
                reason_details.append(
                    f"diversity_recovery(+{recovery_amount:.4f})"
                )
            else:
                reason_details.append("metrics_in_expected_range")

        # --- eta adjustments tied to convergence/diversity intent ---
        if capabilities.supports_eta_c and eta_c is not None:
            if control_state == ControlState.INCREASE_CONVERGENCE:
                eta_c += self.config.eta_c_step  # larger eta → smaller perturbation
                reason_details.append("raise_eta_c_for_convergence")
            elif control_state == ControlState.INCREASE_DIVERSITY:
                eta_c -= self.config.eta_c_step  # smaller eta → larger perturbation
                reason_details.append("lower_eta_c_for_diversity")
            eta_c = _clip(eta_c, self.config.min_eta_c, self.config.max_eta_c)

        if capabilities.supports_eta_m and eta_m is not None:
            if control_state == ControlState.INCREASE_CONVERGENCE:
                eta_m += self.config.eta_m_step
                reason_details.append("raise_eta_m_for_convergence")
            elif control_state == ControlState.INCREASE_DIVERSITY:
                eta_m -= self.config.eta_m_step
                reason_details.append("lower_eta_m_for_diversity")
            eta_m = _clip(eta_m, self.config.min_eta_m, self.config.max_eta_m)

        # Then clip values into the safe experimental operating range.
        mutation = _clip(mutation, self.config.min_mutation_prob, self.config.max_mutation_prob)
        crossover = _clip(crossover, self.config.min_crossover_prob, self.config.max_crossover_prob)

        # Apply optional parameter adjustments only when the optimizer supports
        # the corresponding operator dimension.
        if capabilities.supports_repair_prob and repair_prob is not None:
            if control_state == ControlState.INCREASE_FEASIBILITY:
                repair_prob += self.config.repair_step
                reason_details.append("raise_repair_for_feasibility")
            elif control_state == ControlState.INCREASE_DIVERSITY:
                repair_prob -= self.config.repair_step * 0.5
                reason_details.append("lower_repair_for_exploration")
            repair_prob = _clip(repair_prob, self.config.min_repair_prob, self.config.max_repair_prob)

        if capabilities.supports_local_search_prob and local_search_prob is not None:
            if control_state == ControlState.INCREASE_CONVERGENCE:
                local_search_prob += self.config.local_search_step
                reason_details.append("raise_local_search_for_convergence")
            elif control_state == ControlState.INCREASE_DIVERSITY:
                local_search_prob -= self.config.local_search_step * 0.5
                reason_details.append("lower_local_search_for_diversity")
            local_search_prob = _clip(
                local_search_prob, self.config.min_local_search_prob, self.config.max_local_search_prob
            )

        reason = control_state.value
        reason_detail = ";".join(reason_details)
        requested_params = OperatorParams(
            mutation_prob=mutation,
            crossover_prob=crossover,
            eta_c=eta_c,
            eta_m=eta_m,
            repair_prob=repair_prob,
            local_search_prob=local_search_prob,
        )
        return ControlAction(
            generation=state.generation,
            mutation_prob=mutation,
            crossover_prob=crossover,
            reason=reason,
            requested_params=requested_params.to_dict(),
            applied_params=requested_params.active_params(capabilities),
            capabilities=capabilities.to_dict(),
            control_state=control_state,
            reason_detail=reason_detail,
        )


class LLMChainController:
    """Composable controller built from analyst -> actuator (strategist merged into analyst)."""

    def __init__(
        self,
        *,
        control_interval: int,
        experience_lookback: int,
        event_triggered_control: bool = False,
        event_control_cooldown: int = 0,
        forced_control_on_major_event: bool = False,
        analyst: Any,
        strategist: Any = None,
        actuator: Any,
    ) -> None:
        if control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if event_control_cooldown < 0:
            raise ValueError("event_control_cooldown must be >= 0")
        self.control_interval = control_interval
        self.experience_lookback = max(1, experience_lookback)
        self.event_triggered_control = event_triggered_control
        self.event_control_cooldown = event_control_cooldown
        self.forced_control_on_major_event = forced_control_on_major_event
        self.analyst = analyst
        self.actuator = actuator

    def decide(
        self,
        *,
        observation: Observation,
    ) -> ControlAction:
        """Run the two-step reasoning chain (analyst -> actuator) and return a structured action.

        The Strategist LLM call is eliminated: Analyst now outputs both diagnosis
        and strategy intent in a single call. StrategyDecision is constructed
        from AnalysisResult with zero additional network round-trips.
        """
        from llm.strategist import StrategyDecision

        diagnosis = self.analyst.analyze(
            state=observation.state,
            recent_experiences=observation.recent_experiences,
        )
        strategy = StrategyDecision.from_analysis(diagnosis)
        action = self.actuator.act(
            generation=observation.state.generation,
            strategy=strategy,
            current_params=observation.current_params,
            capabilities=observation.capabilities,
        )
        action.is_llm_decision = True
        return action


class HybridController:
    """混合控制器：默认使用 RuleBasedController，在事件触发或长期停滞时唤醒 LLMChainController.

    设计思路：
    - 绝大多数代使用轻量级 rule_control（零 LLM 延迟）。
    - 当 observation 的 trigger_type == "event"（环境巨变）或
      state.stagnation_len >= stagnation_llm_threshold 时，切换到 LLM 推理。
    - 这样在静态/平稳场景下性能接近 rule_control，
      在动态/停滞场景下获得 real_llm 的自适应优势。
    """

    def __init__(
        self,
        *,
        rule_controller: RuleBasedController,
        llm_controller: LLMChainController,
        control_interval: int,
        stagnation_llm_threshold: int = 5,
        event_triggered_control: bool = True,
        event_control_cooldown: int = 0,
        forced_control_on_major_event: bool = True,
    ) -> None:
        if control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        self.rule_controller = rule_controller
        self.llm_controller = llm_controller
        self.control_interval = control_interval
        self.stagnation_llm_threshold = max(1, stagnation_llm_threshold)
        self.event_triggered_control = event_triggered_control
        self.event_control_cooldown = event_control_cooldown
        self.forced_control_on_major_event = forced_control_on_major_event
        # Expose experience_lookback for ClosedLoopRunner._recent_experiences
        self.experience_lookback = llm_controller.experience_lookback
        # 统计 LLM 被唤醒的次数（用于日志/分析）
        self.llm_wakeup_count: int = 0
        self.rule_decision_count: int = 0

    def decide(self, *, observation: Observation) -> ControlAction:
        """根据触发类型和停滞状态选择 rule 或 llm 控制器."""
        use_llm = False
        wakeup_reasons: list[str] = []

        # 条件 1: 事件触发（环境巨变）
        if observation.trigger_type == "event":
            use_llm = True
            wakeup_reasons.append(f"event_triggered(id={observation.trigger_event_id})")

        # 条件 2: 长时间停滞
        if observation.state.stagnation_len >= self.stagnation_llm_threshold:
            use_llm = True
            wakeup_reasons.append(
                f"stagnation({observation.state.stagnation_len})>="
                f"threshold({self.stagnation_llm_threshold})"
            )

        wakeup_reason = "+".join(wakeup_reasons)

        if use_llm:
            self.llm_wakeup_count += 1
            action = self.llm_controller.decide(observation=observation)
            action.is_llm_decision = True
            action.reason_detail = f"hybrid_llm_wakeup:{wakeup_reason};{action.reason_detail}"
            return action

        self.rule_decision_count += 1
        action = self.rule_controller.decide(observation=observation)
        action.is_llm_decision = False
        action.reason_detail = f"hybrid_rule;{action.reason_detail}"
        return action


@dataclass(slots=True)
class _PendingExperience:
    """Internal placeholder for a transition awaiting its next state."""

    state: ParetoState
    action: ControlAction
    stage_id: str | None = None
    wave_id: str | None = None
    trigger_event_id: str | None = None


class ClosedLoopRunner:
    """Coordinate optimizer, sensor, controller, and lightweight logging.

    The runner owns the experiment timeline. It does not inspect low-level
    genomes or encode problem-specific semantics; it only advances generations,
    senses state, triggers control at configured intervals, and records the
    resulting trajectory.
    """

    def __init__(
        self,
        *,
        solver: NSGA2Solver,
        sensor: ParetoStateSensor,
        controller: ControlPolicy,
        logger: EventLogger | None = None,
        experience_pool: ExperiencePool | None = None,
        experience_logger: ExperienceLogger | None = None,
        reward_config: RewardConfig | None = None,
        stagnation_tolerance: float = 1e-12,
    ) -> None:
        self.solver = solver
        self.sensor = sensor
        self.controller = controller
        self.adaptation = NSGA2Adaptation(solver)
        self.logger = logger
        self.experience_pool = experience_pool
        self.experience_logger = experience_logger
        self.reward_config = reward_config or RewardConfig()
        self.stagnation_tolerance = stagnation_tolerance

    def run(self, *, generations: int, reference_point: tuple[float, ...] | None = None) -> list[ParetoState]:
        """Execute the closed-loop optimization process and return sensed states."""
        if generations < 0:
            raise ValueError("generations must be >= 0")
        population = self.solver.initialize_population()
        previous_state: ParetoState | None = None
        states: list[ParetoState] = []
        pending_experience: _PendingExperience | None = None
        last_control_generation: int | None = None
        latest_event_id: str | None = None
        current_wave_id: str | None = None

        # Allow scripted generation-0 events before baseline sensing so that
        # initial state reflects the post-event environment snapshot.
        initial_events = self.solver.apply_runtime_events(generation=0)
        if initial_events:
            population = self.solver.reevaluate_population(population)
            for event in initial_events:
                self._log_runtime_event(event, generation=0)
            current_wave_id = _resolve_wave_id(initial_events[-1])

        # Generation 0 is sensed before any controller intervention so the run
        # always has a baseline state for logging and reward computation.
        initial_state = self.sensor.sense(
            generation=0,
            population=population,
            previous_state=None,
            reference_point=reference_point,
            stagnation_tolerance=self.stagnation_tolerance,
        )
        states.append(initial_state)
        self._log_state(initial_state, self.solver.get_operator_params())
        previous_state = initial_state

        for generation in range(1, generations + 1):
            runtime_events = self.solver.apply_runtime_events(generation=generation)
            if runtime_events:
                population = self.solver.reevaluate_population(population)
                for event in runtime_events:
                    self._log_runtime_event(event, generation=generation)
                latest_event_id = _runtime_event_id(runtime_events[-1], generation=generation)
                current_wave_id = _resolve_wave_id(runtime_events[-1]) or current_wave_id
            population = self.solver.evolve_one_generation(population)
            state = self.sensor.sense(
                generation=generation,
                population=population,
                previous_state=previous_state,
                reference_point=reference_point,
                stagnation_tolerance=self.stagnation_tolerance,
            )
            states.append(state)
            self._log_state(state, self.solver.get_operator_params(), current_wave_id=current_wave_id)

            # A state-action pair becomes a full experience only after the next
            # state has been observed.
            if pending_experience is not None:
                self._finalize_experience(pending_experience, state)
                pending_experience = None

            # Do not schedule a new action after the final generation because no
            # subsequent state would exist to complete the transition.
            trigger_type, cooldown_skipped, suppressed_trigger = self._resolve_trigger(
                generation=generation,
                generations=generations,
                runtime_events=runtime_events,
                last_control_generation=last_control_generation,
            )

            if cooldown_skipped:
                self._log_control_skip(
                    generation=generation,
                    trigger_type=suppressed_trigger,
                    event_id=latest_event_id,
                    last_control_generation=last_control_generation,
                )

            if trigger_type is not None:
                recent_experiences = self._recent_experiences()
                observation = self.adaptation.build_observation(
                    state=state,
                    recent_experiences=recent_experiences,
                    trigger_type=trigger_type,
                    trigger_event_id=latest_event_id if trigger_type == "event" else None,
                )
                action_started = time.perf_counter()
                action = self._decide_action(observation=observation)
                action.decision_runtime_s = time.perf_counter() - action_started
                action.trigger_type = trigger_type
                action.trigger_event_id = latest_event_id if trigger_type == "event" else None
                action.cooldown_skipped = False
                self.adaptation.apply_action(action)
                self._log_action(action)
                stage_id = self._resolve_stage_id(generation=generation, wave_id=current_wave_id)
                pending_experience = _PendingExperience(
                    state=state,
                    action=action,
                    stage_id=stage_id,
                    wave_id=current_wave_id,
                    trigger_event_id=action.trigger_event_id,
                )
                last_control_generation = generation

            previous_state = state

        return states

    def _decide_action(self, *, observation: Observation) -> ControlAction:
        """优先使用统一 observation 接口，兼容旧 controller 签名."""
        try:
            return self.controller.decide(observation=observation)
        except TypeError:
            return self.controller.decide(
                state=observation.state,
                recent_experiences=observation.recent_experiences,
                current_params=observation.current_params,
                capabilities=observation.capabilities,
            )

    def _recent_experiences(self) -> list[ExperienceRecord]:
        """Return the recent experience window expected by the controller."""
        if self.experience_pool is None:
            return []
        lookback = getattr(self.controller, "experience_lookback", 5)
        return self.experience_pool.recent(int(lookback))

    def _log_state(self, state: ParetoState, params: OperatorParams, *, current_wave_id: str | None = None) -> None:
        """Emit a state event enriched with the active operator snapshot."""
        if self.logger is None:
            return
        dynamic = self._dynamic_context()
        self.logger.log(
            {
                "event": "state",
                **state.to_dict(),
                "mutation_prob": params.mutation_prob,
                "crossover_prob": params.crossover_prob,
                "operator_params": params.to_dict(),
                "operator_capabilities": self.solver.get_operator_capabilities().to_dict(),
                "current_wave_id": current_wave_id,
                **dynamic,
            }
        )

    def _log_action(self, action: ControlAction) -> None:
        """Emit a controller action event when structured logging is enabled."""
        if self.logger is None:
            return
        self.logger.log({"event": "action", **action.to_dict()})

    def _log_runtime_event(self, runtime_event: dict[str, Any], *, generation: int) -> None:
        """Emit a DWTA runtime-event record for scripted wave playback."""
        if self.logger is None:
            return
        self.logger.log(
            {
                "event": "runtime_event",
                "generation": generation,
                "runtime_event_id": _runtime_event_id(runtime_event, generation=generation),
                **runtime_event,
                "live_cache_invalidated": True,
                "cache_refresh_count": self._cache_refresh_count(),
            }
        )

    def _resolve_trigger(
        self,
        *,
        generation: int,
        generations: int,
        runtime_events: list[dict[str, Any]],
        last_control_generation: int | None,
    ) -> tuple[str | None, bool, str | None]:
        """Determine whether controller should run and record cooldown skips.

        Returns a 3-tuple of (trigger_type, cooldown_skipped, suppressed_trigger):
        - trigger_type: the trigger to act on (None if skipped)
        - cooldown_skipped: True if a trigger was suppressed by cooldown
        - suppressed_trigger: the would-be trigger type when cooldown suppresses

        The final generation is intentionally excluded from controller triggering
        because no subsequent state would exist to close the experience transition.
        """
        if generation >= generations:
            periodic_due = generation % self.controller.control_interval == 0
            event_enabled = bool(getattr(self.controller, "event_triggered_control", False))
            forced_on_major = bool(getattr(self.controller, "forced_control_on_major_event", False))
            event_due = bool(runtime_events) and (event_enabled or forced_on_major)
            if periodic_due or event_due:
                self._log_final_generation_skip(
                    generation=generation,
                    trigger_type="event" if event_due else "periodic",
                )
            return None, False, None

        periodic_due = generation % self.controller.control_interval == 0
        event_due = False
        event_enabled = bool(getattr(self.controller, "event_triggered_control", False))
        forced_on_major = bool(getattr(self.controller, "forced_control_on_major_event", False))

        if runtime_events and (event_enabled or forced_on_major):
            event_due = True

        if not periodic_due and not event_due:
            return None, False, None

        cooldown = int(getattr(self.controller, "event_control_cooldown", 0))
        if (
            event_due
            and forced_on_major
            and last_control_generation is not None
            and generation - last_control_generation <= cooldown
        ):
            # Major events are allowed to bypass cooldown so hybrid controllers
            # can react immediately to disruptive environment shifts.
            return "event", False, None

        if last_control_generation is not None and generation - last_control_generation <= cooldown:
            # Cooldown active: suppress control but record the would-be trigger.
            suppressed = "event" if event_due else "periodic"
            return None, True, suppressed

        if event_due:
            return "event", False, None
        if periodic_due:
            return "periodic", False, None
        return None, False, None

    def _log_control_skip(
        self,
        *,
        generation: int,
        trigger_type: str | None,
        event_id: str | None,
        last_control_generation: int | None,
    ) -> None:
        """Log skipped control windows when cooldown suppresses triggering."""
        if self.logger is None:
            return
        self.logger.log(
            {
                "event": "control_skip",
                "generation": generation,
                "trigger_type": trigger_type,
                "trigger_event_id": event_id,
                "cooldown_skipped": True,
                "last_control_generation": last_control_generation,
            }
        )

    def _log_final_generation_skip(
        self,
        *,
        generation: int,
        trigger_type: str,
    ) -> None:
        """Log that the final generation suppressed a would-be control trigger.

        Control is intentionally skipped on the last generation because there
        is no subsequent state to close the experience transition tuple. This
        event lets post-hoc analysis account for the reduced ``num_actions``
        count in short experiments (e.g. toy preset with 4 generations).
        """
        if self.logger is None:
            return
        self.logger.log(
            {
                "event": "control_skip",
                "generation": generation,
                "trigger_type": trigger_type,
                "trigger_event_id": None,
                "cooldown_skipped": False,
                "skip_reason": "final_generation",
                "last_control_generation": None,
            }
        )

    def _finalize_experience(self, pending: _PendingExperience, next_state: ParetoState) -> None:
        """Close a pending transition and persist it to memory/logging sinks."""
        if self.experience_pool is None and self.experience_logger is None:
            return

        reward = compute_reward(state=pending.state, next_state=next_state, config=self.reward_config)
        record = ExperienceRecord(
            state=pending.state.to_dict(),
            action=pending.action.to_dict(),
            reward=reward,
            next_state=next_state.to_dict(),
            stage_id=pending.stage_id,
            wave_id=pending.wave_id,
            trigger_event_id=pending.trigger_event_id,
        )
        if self.experience_pool is not None:
            self.experience_pool.append(record)
        if self.experience_logger is not None:
            self.experience_logger.log(record)

    def _dynamic_context(self) -> dict[str, Any]:
        """Return DWTA dynamic context fields for generation-level logs."""
        live_cache = self.solver.dwta_live_cache
        if live_cache is None:
            return {
                "active_targets_count": None,
                "active_weapons_count": None,
                "cache_refresh_count": 0,
            }
        snapshot = live_cache.get_snapshot()
        return {
            "active_targets_count": int(snapshot.target_active_mask.sum()),
            "active_weapons_count": int(snapshot.weapon_active_mask.sum()),
            "cache_refresh_count": int(live_cache.refresh_count),
        }

    def _cache_refresh_count(self) -> int:
        """Read cache refresh count with static-mode fallback."""
        live_cache = self.solver.dwta_live_cache
        if live_cache is None:
            return 0
        return int(live_cache.refresh_count)

    def _resolve_stage_id(self, *, generation: int, wave_id: str | None) -> str:
        """Build a lightweight stage identifier for experience context."""
        if wave_id:
            return f"wave:{wave_id}:g{generation}"
        return f"static:g{generation}"


def compute_reward(*, state: ParetoState, next_state: ParetoState, config: RewardConfig) -> float:
    """Compute the lightweight reward defined by the MVP memory milestone.

    The reward favors hypervolume improvement and feasibility gains, while
    penalizing the next state's average constraint violation.
    """
    delta_hv = next_state.hv - state.hv
    delta_feasible_ratio = next_state.feasible_ratio - state.feasible_ratio
    return delta_hv + config.alpha * delta_feasible_ratio - config.beta * next_state.mean_cv



def _clip(value: float, low: float, high: float) -> float:
    """Clamp a value into the closed interval ``[low, high]``."""
    return max(low, min(high, value))


def _runtime_event_id(runtime_event: dict[str, Any], *, generation: int) -> str:
    """Build a stable event identifier for control-trigger bookkeeping."""
    wave_id = runtime_event.get("wave_id")
    if isinstance(wave_id, str) and wave_id:
        return wave_id
    event_type = runtime_event.get("event_type")
    if isinstance(event_type, str) and event_type:
        return f"g{generation}:{event_type}"
    return f"g{generation}:runtime_event"


def _resolve_wave_id(runtime_event: dict[str, Any]) -> str | None:
    """Return wave id when present and non-empty."""
    wave_id = runtime_event.get("wave_id")
    if isinstance(wave_id, str) and wave_id:
        return wave_id
    return None
