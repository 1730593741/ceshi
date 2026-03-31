"""Analyst 角色: 总结 Pareto 状态 与 近期经验 转换为 诊断信息."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from controller.control_semantics import ControlState
from infra.llm_client import LLMClient
from memory.experience_pool import ExperienceRecord
from sensing.pareto_state import ParetoState

logger = logging.getLogger(__name__)

# Key fields from state/action that carry real decision signal; rest is noise.
_EXPERIENCE_STATE_KEYS = (
    "feasible_ratio",
    "diversity_score",
    "delta_hv",
    "hv",
    "stagnation_len",
    "rank1_ratio",
)
_EXPERIENCE_ACTION_KEYS = (
    "mutation_prob",
    "crossover_prob",
    "control_state",
    "reason",
)


class AnalysisResult(BaseModel):
    """Structured diagnosis 直接作为 actuator 的策略决定 (merged Analyst+Strategist)."""

    control_state: ControlState
    reason: str
    rationale: str = ""
    convergence_signal: float
    diversity_signal: float
    feasibility_signal: float


class Analyst:
    """构建 diagnosis object 从 状态 与 近期经验，并直接输出控制策略."""

    def __init__(self, client: LLMClient, prompt_path: str = "llm/prompts/analyst.txt") -> None:
        self.client = client
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def analyze(self, *, state: ParetoState, recent_experiences: list[ExperienceRecord]) -> AnalysisResult:
        payload = {
            "state": state.to_dict(),
            "recent_experiences": [
                _compress_experience(record) for record in recent_experiences
            ],
        }
        response = self.client.generate_json(task="analyst", payload=payload, prompt_template=self.prompt_template)
        if not response.content:
            logger.warning("Analyst hold fallback due to llm error: %s", response.error)
            return _fallback_analysis(state=state, recent_experiences=recent_experiences)

        result = AnalysisResult.model_validate(response.content)
        if response.error:
            logger.warning("Analyst used fallback mode=%s: %s", response.mode_used, response.error)
        return result


def _compress_experience(record: ExperienceRecord) -> dict:
    """只保留对决策有信号量的关键字段，大幅压缩 experiences payload."""
    raw = record.to_dict()
    state = raw.get("state", {})
    action = raw.get("action", {})
    next_state = raw.get("next_state", {})
    return {
        "state": {k: state.get(k) for k in _EXPERIENCE_STATE_KEYS if k in state},
        "action": {k: action.get(k) for k in _EXPERIENCE_ACTION_KEYS if k in action},
        "reward": raw.get("reward"),
        "next_state": {k: next_state.get(k) for k in _EXPERIENCE_STATE_KEYS if k in next_state},
    }


def _fallback_analysis(*, state: ParetoState, recent_experiences: list[ExperienceRecord]) -> AnalysisResult:
    recent_rewards = [float(record.reward) for record in recent_experiences]
    reward_trend = (sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0

    if state.feasible_ratio < 0.6:
        control_state = ControlState.INCREASE_FEASIBILITY
        reason = "fallback_low_feasible_ratio"
        rationale = "Feasibility below threshold; prioritize repair and constraint recovery."
    elif state.diversity_score < 0.12:
        control_state = ControlState.INCREASE_DIVERSITY
        reason = "fallback_low_diversity"
        rationale = "Diversity too low; increase mutation and reduce crossover pressure."
    elif state.stagnation_len >= 3 or state.delta_hv <= 1e-6 or reward_trend < -1e-6:
        control_state = ControlState.INCREASE_CONVERGENCE
        reason = "fallback_stagnation_or_negative_recent_reward"
        rationale = "Stagnation or negative HV trend; strengthen convergence pressure."
    else:
        control_state = ControlState.MAINTAIN_BALANCE
        reason = "fallback_balanced"
        rationale = "Metrics within acceptable range; maintain current balance."

    return AnalysisResult(
        control_state=control_state,
        reason=reason,
        rationale=rationale,
        convergence_signal=float(max(0.0, min(1.0, state.rank1_ratio))),
        diversity_signal=float(max(0.0, min(1.0, state.diversity_score))),
        feasibility_signal=float(max(0.0, min(1.0, state.feasible_ratio))),
    )
