"""参考前沿 构建辅助函数 用于 可复现的 指标 evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sensing.hypervolume import ObjectivePoint


def dominates(a: ObjectivePoint, b: ObjectivePoint) -> bool:
    """返回 True 当 点 ``一个`` dominates 点 ``b`` (minimization)."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def nondominated(points: list[ObjectivePoint]) -> list[ObjectivePoint]:
    """筛选 非支配 点 并保持稳定且确定的顺序."""
    result: list[ObjectivePoint] = []
    for idx, point in enumerate(points):
        is_dominated = False
        for j, other in enumerate(points):
            if idx == j:
                continue
            if dominates(other, point):
                is_dominated = True
                break
        if not is_dominated:
            result.append(point)
    return result


@dataclass(frozen=True, slots=True)
class ReferenceFront:
    """参考前沿 与 its explicit provenance 元数据."""

    points: list[ObjectivePoint]
    source: str
    details: dict[str, Any]


def build_true_reference_front(points: list[ObjectivePoint], *, name: str) -> ReferenceFront:
    """构建 reference 从 已知真实 Pareto 前沿 点."""
    return ReferenceFront(
        points=nondominated(points),
        source="true_pareto_front",
        details={"name": name, "num_points": len(points)},
    )


def read_final_front_from_generation_log(path: Path) -> list[ObjectivePoint]:
    """从 generation_metrics JSONL 读取最终非支配前沿目标值.
    
        This 为 explicit 与 可复现的 because 点 为 recorded 在 该 运行 日志,
        then declared 作为 sourced 从 该 exact file path.
        """
    fronts = read_generation_fronts_from_generation_log(path)
    return fronts[-1] if fronts else []


def read_generation_fronts_from_generation_log(path: Path) -> list[list[ObjectivePoint]]:
    """从 generation_metrics JSONL 读取每一代的 rank-1 前沿目标值."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fronts: list[list[ObjectivePoint]] = []
    for row in rows:
        raw_front = row.get("rank1_objectives")
        if not isinstance(raw_front, list):
            continue
        parsed: list[ObjectivePoint] = []
        for point in raw_front:
            if isinstance(point, (list, tuple)):
                parsed.append(tuple(float(v) for v in point))
        fronts.append(nondominated(parsed))
    return fronts


def build_empirical_reference_front(method_to_generation_logs: dict[str, list[Path]]) -> ReferenceFront:
    """构建 empirical reference 前沿 从 matched runs 跨 compared 方法."""
    aggregated: list[ObjectivePoint] = []
    consumed: dict[str, list[str]] = {}
    for method, logs in method_to_generation_logs.items():
        consumed[method] = []
        for log_path in logs:
            consumed[method].append(str(log_path))
            aggregated.extend(read_final_front_from_generation_log(log_path))

    return ReferenceFront(
        points=nondominated(aggregated),
        source="empirical_matched_runs",
        details={
            "methods": sorted(method_to_generation_logs.keys()),
            "consumed_generation_logs": consumed,
            "num_aggregated_points": len(aggregated),
        },
    )


def build_intra_run_reference_front(generation_events: list[dict[str, Any]]) -> ReferenceFront:
    """构建 empirical reference 从 rank-1 前沿 跨 一个 单个 运行 timeline."""
    points: list[ObjectivePoint] = []
    for event in generation_events:
        raw = event.get("rank1_objectives")
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, (list, tuple)):
                points.append(tuple(float(v) for v in item))
    return ReferenceFront(
        points=nondominated(points),
        source="intra_run_union_front",
        details={"num_events": len(generation_events), "num_aggregated_points": len(points)},
    )
