"""Hypervolume 工具 用于 Pareto-状态 感知.

This module currently exposes 一个 稳定接口，并带有 一个 simplified 2-目标
implementation 到 keep 该 MVP 轻量 与 可替换.
"""

from __future__ import annotations

from collections.abc import Sequence

ObjectivePoint = tuple[float, ...]


class HypervolumeCalculator:
    """稳定且可替换的 hypervolume calculator 接口."""

    def compute(self, points: Sequence[ObjectivePoint], reference_point: ObjectivePoint) -> float:
        """计算 hypervolume 用于 一个 minimization 前沿 与 reference 点."""
        raise NotImplementedError


class SimplifiedHypervolumeCalculator(HypervolumeCalculator):
    """简化版 MVP hypervolume 实现.
    
        说明：
          - 当前仅支持两个目标。
          - 假设目标为最小化。
          - 在筛除支配点后累加矩形切片。
        """

    def compute(self, points: Sequence[ObjectivePoint], reference_point: ObjectivePoint) -> float:
        """计算 simplified 2D hypervolume.
        
                返回s 0.0 用于 空 inputs. Raises ValueError if dimensions 为 invalid.
                """
        if not points:
            return 0.0
        if len(reference_point) != 2:
            raise ValueError("Simplified hypervolume currently supports 2 objectives")

        filtered: list[tuple[float, float]] = []
        for point in points:
            if len(point) != 2:
                raise ValueError("Simplified hypervolume currently supports 2 objectives")
            p0, p1 = point
            r0, r1 = reference_point
            if p0 <= r0 and p1 <= r1:
                filtered.append((p0, p1))

        if not filtered:
            return 0.0

        nondominated = _filter_nondominated_2d(filtered)
        sorted_points = sorted(nondominated, key=lambda value: value[0])

        hv = 0.0
        current_y_limit = reference_point[1]
        for x, y in sorted_points:
            if y >= current_y_limit:
                continue
            width = max(0.0, reference_point[0] - x)
            height = current_y_limit - y
            hv += width * height
            current_y_limit = y
        return hv


def compute_hypervolume(points: Sequence[ObjectivePoint], reference_point: ObjectivePoint) -> float:
    """计算 hypervolume via 该 默认 simplified calculator."""
    return SimplifiedHypervolumeCalculator().compute(points, reference_point)


def _filter_nondominated_2d(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """筛选 非支配 点 用于 two-目标 minimization.

    修复说明：原实现使用 ``other != point`` 作为自身排除条件，当种群中存在
    重复点时，两个值相同的点互相都不被视为「其它个体」，导致重复点均进入
    非支配集。修复后改用索引比较（``j != i``），确保与自身以外的所有点
    正确比较，重复点只保留第一次出现的副本。
    """
    pts = list(points)
    result: list[tuple[float, float]] = []
    for i, point in enumerate(pts):
        dominated = False
        for j, other in enumerate(pts):
            if j == i:
                continue
            # ``other`` strictly dominates ``point`` for minimization:
            # other is no worse on both objectives AND strictly better on at least one.
            if other[0] <= point[0] and other[1] <= point[1] and (other[0] < point[0] or other[1] < point[1]):
                dominated = True
                break
        if not dominated:
            # Deduplicate: skip if an identical point was already added.
            if point not in result:
                result.append(point)
    return result
