"""Strategist 角色 (deprecated: merged into Analyst).

StrategyDecision 数据类保留供 Actuator 接口使用。
Strategist LLM 调用已消除 —— Analyst 直接输出完整策略决定，
本模块仅提供从 AnalysisResult 构造 StrategyDecision 的零开销适配器。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from controller.control_semantics import ControlState
from llm.analyst import AnalysisResult

logger = logging.getLogger(__name__)


class StrategyDecision(BaseModel):
    """为 actuator 选择的高层控制状态."""

    control_state: ControlState
    rationale: str

    @classmethod
    def from_analysis(cls, diagnosis: AnalysisResult) -> "StrategyDecision":
        """将 AnalysisResult 直接转换为 StrategyDecision，不产生 LLM 网络往返."""
        return cls(
            control_state=diagnosis.control_state,
            rationale=diagnosis.rationale or diagnosis.reason,
        )


class Strategist:
    """(Deprecated) Strategist LLM 角色已合并入 Analyst.

    保留此类仅为向后兼容。LLMChainController 不再调用 plan()，
    而是直接从 AnalysisResult 构造 StrategyDecision。
    """

    def __init__(self, client: object = None, prompt_path: str = "llm/prompts/strategist.txt") -> None:
        logger.warning(
            "Strategist LLM role is deprecated and will not make any LLM calls. "
            "Analysis and strategy are now merged in the Analyst role."
        )
        self.client = client

    def plan(self, diagnosis: AnalysisResult) -> StrategyDecision:
        """(Deprecated) 直接从 diagnosis 透传，不调用 LLM."""
        return StrategyDecision.from_analysis(diagnosis)
