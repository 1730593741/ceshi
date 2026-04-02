"""用于测试 unified four-状态 控制 semantics 跨 LLM chain."""

from __future__ import annotations

from controller.control_semantics import ControlState
from infra.llm_client import LLMClient, LLMClientConfig
from llm.analyst import Analyst
from llm.strategist import Strategist
from memory.experience_pool import ExperienceRecord
from sensing.pareto_state import ParetoState


def test_analyst_output_uses_control_state_enum() -> None:
    analyst = Analyst(LLMClient(LLMClientConfig(mode="mock_llm")))
    state = ParetoState(
        generation=5,
        hv=0.1,
        delta_hv=0.0,
        feasible_ratio=0.4,
        rank1_ratio=0.3,
        mean_cv=0.8,
        diversity_score=0.2,
        crowding_entropy=0.3,
        d_dec=0.2,
        d_front=0.4,
        stagnation_len=2,
    )

    result = analyst.analyze(state=state, recent_experiences=[])

    assert isinstance(result.control_state, ControlState)
    assert result.control_state == ControlState.INCREASE_FEASIBILITY


def test_strategist_preserves_four_state_control_semantics() -> None:
    client = LLMClient(LLMClientConfig(mode="mock_llm"))
    strategist = Strategist(client)
    diagnosis = Analyst(client).analyze(
        state=ParetoState(
            generation=3,
            hv=0.4,
            delta_hv=0.01,
            feasible_ratio=0.9,
            rank1_ratio=0.5,
            mean_cv=0.1,
            diversity_score=0.02,
            crowding_entropy=0.2,
            d_dec=0.3,
            d_front=0.6,
            stagnation_len=0,
        ),
        recent_experiences=[],
    )

    decision = strategist.plan(diagnosis)

    assert decision.control_state in {
        ControlState.INCREASE_DIVERSITY,
        ControlState.INCREASE_CONVERGENCE,
        ControlState.INCREASE_FEASIBILITY,
        ControlState.MAINTAIN_BALANCE,
    }


def test_analyst_uses_recent_experience_outcome_signal() -> None:
    analyst = Analyst(LLMClient(LLMClientConfig(mode="mock_llm")))
    state = ParetoState(
        generation=7,
        hv=0.6,
        delta_hv=0.01,
        feasible_ratio=0.95,
        rank1_ratio=0.7,
        mean_cv=0.05,
        diversity_score=0.4,
        crowding_entropy=0.6,
        d_dec=0.4,
        d_front=0.6,
        stagnation_len=0,
    )
    recent_experiences = [
        ExperienceRecord(
            state={},
            action={"control_state": "maintain_balance"},
            reward=-0.3,
            next_state={},
        )
    ]

    result = analyst.analyze(state=state, recent_experiences=recent_experiences)

    assert result.control_state == ControlState.INCREASE_CONVERGENCE
